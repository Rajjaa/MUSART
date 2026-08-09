"""Question templates: one natural-language question shape per relation.

A template is a question with ``[S]`` where the subject goes::

    publisher            -> Who is the publisher of [S]?
    architectural style  -> What is the architectural style of [S]?

Wording matters more than it looks. A stilted template ("What is the [S]'s
publisher?") depresses every model's score uniformly and turns a knowledge probe
into a phrasing probe, so the shipped CSV is hand-written.

A new corpus will surface relations the CSV does not cover. Rather than fail, or
silently use wording that would skew results, unknown relations are sent to an
LLM once, validated, and written back to the CSV with ``source=llm`` so a human
can review exactly what was invented before trusting the numbers.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

SUBJECT = "[S]"
FIELDNAMES = ["relation", "template", "source"]

PROMPT = textwrap.dedent(
    """\
    You are writing question templates for a knowledge-probing benchmark built
    from Wikidata relations.

    Write one natural English question that asks for the value of the relation
    "{relation}" of a subject. Use the literal placeholder [S] exactly once,
    where the subject's name goes.

    Rules:
    - Output only the question. No explanation, no quotes, no numbering.
    - It must be a single line ending in a question mark.
    - It must contain [S] exactly once.
    - Phrase it the way a person would ask, not as a slot-filling prompt.
      Good:  Who is the publisher of [S]?
      Good:  Where is [S] located?
      Bad:   What is the [S]'s publisher?

    Relation: {relation}
    Question:"""
)


def fallback(relation: str) -> str:
    return f"What is the {relation} of {SUBJECT}?"


def validate(template: str) -> Optional[str]:
    """Return an error message, or None when the template is usable."""
    if not template or not template.strip():
        return "empty"
    if "\n" in template.strip():
        return "more than one line"
    if template.count(SUBJECT) != 1:
        return f"contains {SUBJECT} {template.count(SUBJECT)} times, expected exactly 1"
    if not template.strip().endswith("?"):
        return "does not end in a question mark"
    if re.search(r"\{\w+\}|\[[A-RT-Z]\]", template):
        return "contains an unexpected placeholder"
    return None


def load(path: Path) -> Dict[str, Tuple[str, str]]:
    """relation -> (template, source). Missing file is an empty mapping."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: Dict[str, Tuple[str, str]] = {}
    for row in rows:
        relation = (row.get("relation") or "").strip()
        template = (row.get("template") or "").strip()
        if not relation or not template:
            continue
        out[relation] = (template, (row.get("source") or "manual").strip())
    return out


def save(path: Path, templates: Dict[str, Tuple[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIELDNAMES)
        for relation in sorted(templates):
            template, source = templates[relation]
            writer.writerow([relation, template, source])


# --- LLM generation -------------------------------------------------------


def _resolve_provider(provider: str) -> str:
    if provider != "auto":
        return provider
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def _ask_gemini(prompt: str, model: Optional[str]) -> str:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model or "gemini-3.1-flash-lite", contents=prompt
    )
    return (response.text or "").strip()


def _ask_anthropic(prompt: str, model: Optional[str]) -> str:
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model or "claude-sonnet-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in message.content if block.type == "text"
    ).strip()


def generate(relation: str, provider: str, model: Optional[str]) -> Tuple[str, str]:
    """Draft one template. Returns (template, source).

    Retries once on a validation failure, then falls back to generic wording --
    a benchmark build should never die because one relation was awkward to
    phrase, but the ``source`` column records exactly what happened.
    """
    ask = {"gemini": _ask_gemini, "anthropic": _ask_anthropic}.get(provider)
    if ask is None:
        return fallback(relation), "fallback"

    prompt = PROMPT.format(relation=relation)
    for attempt in (1, 2):
        try:
            candidate = ask(prompt, model).strip().strip('"')
        except Exception as exc:  # noqa: BLE001 - any API failure falls back
            log.warning("template generation for %r failed: %s", relation, exc)
            break
        problem = validate(candidate)
        if problem is None:
            return candidate, "llm"
        log.warning(
            "generated template for %r rejected (%s): %r", relation, problem, candidate
        )
        if attempt == 1:
            prompt = PROMPT.format(relation=relation) + (
                f"\n\nYour previous answer was rejected because it {problem}. "
                f"Try again."
            )
    return fallback(relation), "fallback"


def resolve(
    relations: Sequence[str],
    path: Path,
    provider: str = "auto",
    model: Optional[str] = None,
) -> Dict[str, str]:
    """Get a template for every relation, generating and caching what is missing."""
    known = load(path)
    missing = [r for r in dict.fromkeys(relations) if r not in known]

    if missing:
        resolved_provider = _resolve_provider(provider)
        if resolved_provider == "none":
            log.warning(
                "%d relation(s) have no template and no LLM is configured; using "
                "generic wording. Set GEMINI_API_KEY or ANTHROPIC_API_KEY, or write "
                "templates into %s, for better questions.",
                len(missing),
                path,
            )
        else:
            log.info(
                "generating %d template(s) with %s", len(missing), resolved_provider
            )
        for relation in missing:
            if resolved_provider == "none":
                known[relation] = (fallback(relation), "fallback")
            else:
                known[relation] = generate(relation, resolved_provider, model)

        save(path, known)
        invented = [(r, known[r][0], known[r][1]) for r in missing]
        log.warning(
            "wrote %d new template(s) to %s -- review these before publishing:\n%s",
            len(invented),
            path,
            "\n".join(f"  [{source}] {rel}: {tpl}" for rel, tpl, source in invented),
        )

    return {relation: known[relation][0] for relation in relations}


def render(template: str, subject_title: str) -> str:
    return template.replace(SUBJECT, subject_title)
