"""Resolve extracted mentions to Wikidata entities using the article's own links.

The rule that matters: a mention is linked **only** when the article itself
anchors it. Matching a surface form against a global title index instead would
attach entities the article never points at -- "Mercury" in a chemistry article
resolving to the planet -- and those false links are worse than no link at all,
because they enter the gold answer set and silently reward wrong answers.

Four passes, most specific first. Mentions that survive all four unlinked are
**kept**, as ``TEXT:``-keyed surface forms rather than discarded, so a model
answering with the article's exact wording is credited instead of being counted
a false positive.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

TEXT_KEY_PREFIX = "TEXT:"


def slugify(text: str) -> str:
    """Filesystem-safe token for titles and relations."""
    return re.sub(r"[^\w.-]", "_", str(text))


def _norm(text: str) -> str:
    """Casefold and collapse whitespace, so anchors and mentions compare fairly."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


# --- tiering ---------------------------------------------------------------


def collect_extractions(run_dir: Path, fname2pair: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    """Read every per-pair extraction file into one tiered table.

    Tier 1 means langextract could align the extraction to a character span in
    the source text; tier 2 means it could not, and the model may have inferred
    or invented it. Only tier 1 is trustworthy enough for a gold answer.
    """
    # pathlib.glob sees dotfiles, which matters: articles titled ".hack//G.U."
    # and "...Thirteen Years Later" produce files a shell `ls *.jsonl` misses.
    files = [f for f in run_dir.glob("*.jsonl") if f.name != "extraction_results.jsonl"]
    if not files:
        raise SystemExit(f"no extraction files in {run_dir}")

    rows, unresolved = [], []
    for path in tqdm(sorted(files), desc="tiering"):
        pair = fname2pair.get(path.stem)
        if pair is None:
            unresolved.append(path.name)
            continue
        try:
            data = json.load(path.open(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            unresolved.append(path.name)
            continue
        subject_title, relation = pair
        for extraction in data.get("extractions") or []:
            rows.append(
                {
                    "source_file": path.name,
                    "entity": subject_title,
                    "property": relation,
                    "tier": 1 if extraction.get("char_interval") is not None else 2,
                    "extraction_text": extraction.get("extraction_text"),
                    "char_interval": extraction.get("char_interval"),
                    "extraction_class": extraction.get("extraction_class"),
                    "alignment_status": extraction.get("alignment_status"),
                }
            )

    if unresolved:
        log.warning(
            "%d file(s) could not be matched to a (title, relation) pair, e.g. %s",
            len(unresolved),
            unresolved[:3],
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"no extractions parsed from {run_dir}")
    frame["tier_description"] = frame["tier"].map({1: "grounded", 2: "inferred"})
    return frame


# --- linking ---------------------------------------------------------------


def build_html_link_map(html_path: Path) -> Dict[str, str]:
    """{link text -> target article title} for every /wiki/ link in one article.

    Parsing the article once and indexing all its links is what makes this
    affordable; resolving each mention by re-parsing the document does not scale.
    """
    from bs4 import BeautifulSoup

    try:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    out: Dict[str, str] = {}
    for link in BeautifulSoup(html, "html.parser").find_all("a"):
        href = link.get("href")
        if not (href and "/wiki/" in href):
            continue
        # .string is None when the anchor wraps markup (<a><i>Title</i></a>),
        # which would silently drop those links from the inventory.
        text = link.string or link.get_text(" ", strip=True)
        if not text:
            continue
        text = text.strip()
        if text and text not in out:
            title = urllib.parse.unquote(href.split("/wiki/")[1].split("#")[0])
            out[text] = title.replace("_", " ")
    return out


def index_anchors(link_map: Dict[str, str]):
    """Index one article's anchors for the four resolution passes."""
    by_text: Dict[str, str] = {}
    by_title: Dict[str, str] = {}
    token_index = defaultdict(list)
    for text, title in link_map.items():
        norm_text = _norm(text)
        if norm_text:
            by_text.setdefault(norm_text, title)
            # Any anchor containing a mention must contain the mention's first
            # token, so a token -> anchors index makes the containment pass cheap.
            for token in set(norm_text.split()):
                token_index[token].append((norm_text, title))
        norm_title = _norm(title)
        if norm_title:
            by_title.setdefault(norm_title, title)
    return by_text, by_title, token_index


def resolve_mention(
    text,
    raw_map,
    by_text,
    by_title,
    token_index,
    title2qid,
    allow_submention: bool,
):
    """Resolve one mention to (target title, method) using this article's anchors."""
    raw = str(text).strip()
    title = raw_map.get(raw)
    if title and title2qid.get(title):
        return title, "hyperlink_exact"

    norm = _norm(raw)
    if not norm:
        return None, None

    # Same anchor, different casing or spacing.
    title = by_text.get(norm)
    if title and title2qid.get(title):
        return title, "hyperlink_normalized"

    # The mention is the *target* of a piped anchor: the article links
    # [[Sega Saturn|Saturn]] and the extractor picked up "Sega Saturn".
    title = by_title.get(norm)
    if title and title2qid.get(title):
        return title, "hyperlink_title"

    # Short-form repeat mention: the article anchors "Barack Obama" on first
    # occurrence and later says just "Obama". Accepted only when every anchor
    # containing the mention at word boundaries resolves to the SAME entity --
    # otherwise "New York" inside both "New York City" and "New York City
    # Subway" would link to whichever happened to be seen first.
    if allow_submention and len(norm) >= 3:
        padded = f" {norm} "
        candidates = {}
        for anchor_norm, anchor_title in token_index.get(norm.split()[0], ()):
            if padded in f" {anchor_norm} ":
                qid = title2qid.get(anchor_title)
                if qid:
                    candidates[qid] = anchor_title
        if len(candidates) == 1:
            return next(iter(candidates.values())), "hyperlink_submention"

    return None, None


def link_mentions(
    frame: pd.DataFrame,
    html_dir: Path,
    title2qid: Dict[str, str],
    allow_submention: bool = True,
) -> pd.DataFrame:
    """Attach wikidata_qid / wikipedia_title to each extraction via hyperlinks."""
    qids, titles, methods = {}, {}, {}
    stats = defaultdict(int)

    for entity, group in tqdm(
        frame.groupby("entity", sort=False),
        desc="linking",
        total=frame["entity"].nunique(),
    ):
        # '/' is the only character the HTML cache sanitises out of titles.
        html_path = html_dir / f"{str(entity).replace('/', '_')}.html"
        if html_path.exists():
            page_links = build_html_link_map(html_path)
        else:
            page_links = {}
            stats["articles_without_html"] += 1
        by_text, by_title, token_index = index_anchors(page_links)

        resolved = {}
        for idx, text in group["extraction_text"].items():
            key = str(text).strip()
            if key not in resolved:
                resolved[key] = resolve_mention(
                    key, page_links, by_text, by_title, token_index,
                    title2qid, allow_submention,
                )
            target, method = resolved[key]
            qid = title2qid.get(target) if target else None
            qids[idx] = qid
            titles[idx] = target if qid else None
            methods[idx] = method if qid else None
            stats[method if qid else "unlinked"] += 1

    frame = frame.copy()
    frame["wikidata_qid"] = pd.Series(qids)
    frame["wikipedia_title"] = pd.Series(titles)
    frame["linking_method"] = pd.Series(methods)

    total = len(frame)
    linked = frame["wikidata_qid"].notna().sum()
    log.info("%d/%d extractions linked (%.1f%%)", linked, total, 100 * linked / max(total, 1))
    for method in (
        "hyperlink_exact",
        "hyperlink_normalized",
        "hyperlink_title",
        "hyperlink_submention",
    ):
        if stats[method]:
            log.info("  %-22s %7d", method, stats[method])
    log.info("  %-22s %7d (kept as surface forms)", "unlinked", stats["unlinked"])
    if stats["articles_without_html"]:
        log.warning("%d article(s) had no HTML", stats["articles_without_html"])
    return frame


# --- merge -----------------------------------------------------------------


def merge(
    items: pd.DataFrame,
    linked: pd.DataFrame,
    qid2title: Dict[str, str],
    qid2aliases: Dict[str, List[str]],
    drop_unlinked: bool = False,
) -> pd.DataFrame:
    """Fold linked entities and surface forms into each item's gold answers."""
    ok = linked[linked["wikidata_qid"].notna()]
    triples = (
        ok.groupby(["entity", "property"])
        .agg({"wikidata_qid": lambda s: list(dict.fromkeys(s))})
        .reset_index()
        .rename(
            columns={
                "entity": "subject_title",
                "property": "relation",
                "wikidata_qid": "wp_qids",
            }
        )
    )
    merged = items.merge(triples, on=["subject_title", "relation"], how="left")
    merged["wp_qids"] = merged["wp_qids"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    if drop_unlinked:
        merged["extra_surface_forms"] = [[] for _ in range(len(merged))]
    else:
        rest = linked[linked["wikidata_qid"].isna()]
        surface = (
            rest.groupby(["entity", "property"])
            .agg(
                {
                    "extraction_text": lambda s: list(
                        dict.fromkeys(
                            str(t).strip() for t in s if str(t).strip()
                        )
                    )
                }
            )
            .reset_index()
            .rename(
                columns={
                    "entity": "subject_title",
                    "property": "relation",
                    "extraction_text": "extra_surface_forms",
                }
            )
        )
        merged = merged.merge(surface, on=["subject_title", "relation"], how="left")
        merged["extra_surface_forms"] = merged["extra_surface_forms"].apply(
            lambda x: x if isinstance(x, list) else []
        )

    merged["extra_objects"] = merged.apply(
        lambda r: [q for q in r["wp_qids"] if q not in (r["ground_truth"] or {})],
        axis=1,
    )

    def augment(row):
        out = {
            qid: list(
                dict.fromkeys(qid2aliases.get(qid, []) + [qid2title.get(qid, qid)])
            )
            for qid in row["wp_qids"]
        }
        for qid, forms in (row["ground_truth"] or {}).items():
            if qid not in out:
                out[qid] = forms
        # An unlinked mention only earns its own gold-entity slot when no entity
        # already accepts that surface form; otherwise one gold object splits in
        # two and the model is charged a false negative for answering correctly.
        seen = {str(f).strip().lower() for forms in out.values() for f in forms}
        for form in row["extra_surface_forms"]:
            key = form.strip().lower()
            if key and key not in seen:
                out[f"{TEXT_KEY_PREFIX}{form}"] = [form]
                seen.add(key)
        return out

    merged["augmented_gt"] = merged.apply(augment, axis=1)

    before = merged["ground_truth"].apply(lambda d: len(d or {})).mean()
    after = merged["augmented_gt"].apply(len).mean()
    log.info("mean gold entities per item: %.2f -> %.2f", before, after)
    return merged
