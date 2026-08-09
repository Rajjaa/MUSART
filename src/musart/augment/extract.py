"""Ask an LLM what an article says about each (subject, relation) pair.

One call per article-relation pair, over the whole article. ``max_char_buffer``
is set above the longest article on purpose: langextract's default (1000) splits
an article into roughly sixteen chunks and re-sends the prompt and examples with
every one, inflating cost about 3.4x for no gain in quality, while a single call
keeps the character-span grounding that the tier-1 filter depends on.

Each pair is written to its own file the moment it succeeds, so a run that dies
halfway -- on a rate limit, a quota, a dropped connection -- resumes by simply
being restarted. Nothing is ever half-written.
"""

from __future__ import annotations

import logging
import os
import random
import re
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from ..config import Config
from .link import slugify

log = logging.getLogger(__name__)

MAX_RETRIES = 5


def build_prompt(subject_title: str, relation: str, example_objects: Sequence[str]):
    """The extraction instruction, plus one worked example built from Wikidata.

    The example is what stops the model returning facts about other subjects
    mentioned in the article, which is the dominant failure mode without it.
    """
    import langextract as lx

    prompt = textwrap.dedent(
        f"""
        From the text, extract all items for the relation '{relation}' that are directly associated with the subject '{subject_title}'.
        Do not extract items related to other subjects.
        Each item should be a distinct extraction.
        Use the exact text for each extraction. Do not paraphrase."""
    )
    extractions = [
        lx.data.Extraction(extraction_class=relation, extraction_text=obj)
        for obj in example_objects
    ]
    joined = "' and '".join(example_objects)
    text = (
        f"For the subject '{subject_title}', some examples of the relation "
        f"'{relation}' are '{joined}'."
    )
    return prompt, [lx.data.ExampleData(text=text, extractions=extractions)]


def output_name(subject_title: str, relation: str) -> str:
    return f"{slugify(subject_title)}_{slugify(relation)}.jsonl"


def load_articles(path: Path) -> Dict[str, str]:
    import json

    out: Dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("title") and record.get("content"):
                out[record["title"]] = record["content"]
    return out


def run(
    cfg: Config,
    items: pd.DataFrame,
    run_dir: Path,
    limit: Optional[int] = None,
) -> Path:
    import langextract as lx

    api_key = os.environ.get("LANGEXTRACT_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "no API key: set LANGEXTRACT_API_KEY (or GEMINI_API_KEY) before "
            "running 'musart augment'"
        )

    articles = load_articles(cfg.work_dir / "articles.jsonl")
    if not articles:
        raise SystemExit(
            f"no article text at {cfg.work_dir / 'articles.jsonl'}; run "
            f"'musart fetch-articles' first"
        )
    log.info("%d article texts available", len(articles))

    run_dir.mkdir(parents=True, exist_ok=True)

    pairs = items[["subject_title", "relation", "ground_truth"]].drop_duplicates(
        subset=["subject_title", "relation"]
    )
    if limit:
        pairs = pairs.head(limit)

    todo = [
        row
        for row in pairs.itertuples(index=False)
        if not (run_dir / output_name(row.subject_title, row.relation)).exists()
    ]
    log.info("%d of %d pairs still to extract", len(todo), len(pairs))

    failures = 0
    for row in tqdm(todo, desc="extract"):
        text = articles.get(row.subject_title)
        if not text:
            log.warning("no article text for %r", row.subject_title)
            failures += 1
            continue

        examples = [form for forms in (row.ground_truth or {}).values() for form in forms]
        if not examples:
            log.warning(
                "no gold objects to build an example from for %r/%r",
                row.subject_title,
                row.relation,
            )
            failures += 1
            continue

        prompt, example_data = build_prompt(row.subject_title, row.relation, examples)

        backoff = 2.0
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = lx.extract(
                    text_or_documents=text,
                    prompt_description=prompt,
                    examples=example_data,
                    model_id=cfg.augment.model_id,
                    api_key=api_key,
                    debug=False,
                    max_workers=cfg.augment.max_workers,
                    batch_length=100,
                    max_char_buffer=cfg.augment.max_char_buffer,
                )
                lx.io.save_annotated_documents(
                    [result],
                    output_dir=run_dir,
                    output_name=output_name(row.subject_title, row.relation),
                )
                break
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                message = str(exc)
                # Quota exhaustion and a spending cap both surface as 429 but
                # mean different things, and neither is fixed by waiting longer.
                if "RESOURCE_EXHAUSTED" in message and (
                    "quota" in message.lower() or "cap" in message.lower()
                ):
                    raise SystemExit(
                        f"the API reports the account is out of budget, not "
                        f"merely rate limited -- extraction stopped. "
                        f"{len(todo)} pair(s) were pending. Original error: {exc}"
                    ) from exc
                if attempt == MAX_RETRIES:
                    log.error(
                        "giving up on %r/%r after %d attempts: %s",
                        row.subject_title,
                        row.relation,
                        MAX_RETRIES,
                        exc,
                    )
                    failures += 1
                    break
                wait = backoff + random.uniform(0, 1)
                log.warning("%s (attempt %d/%d), retrying in %.1fs", exc, attempt, MAX_RETRIES, wait)
                time.sleep(wait)
                backoff = min(backoff * 2, 120)

    if failures:
        log.warning("%d pair(s) produced no extraction file", failures)
    log.info("extractions in %s", run_dir)
    return run_dir
