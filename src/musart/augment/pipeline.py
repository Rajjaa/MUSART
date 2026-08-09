"""Run stage 4 end to end: extract, tier, link, merge."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .. import items as items_mod
from ..config import Config
from . import extract, link

log = logging.getLogger(__name__)


def _title_index(cfg: Config) -> Dict[str, str]:
    if cfg.title2qid is not None:
        return items_mod.load_title2qid(cfg.title2qid)
    path = cfg.work_dir / "title2qid.json"
    if path.exists():
        return items_mod.load_json_map(path)
    raise SystemExit("no title -> QID index; run 'musart collect'")


def run(cfg: Config, limit: Optional[int] = None, extract_only: bool = False) -> None:
    if not cfg.items_file.exists():
        raise SystemExit(f"no items at {cfg.items_file}; run 'musart build' first")
    items = pd.read_json(cfg.items_file, lines=True)
    log.info("%d items", len(items))

    run_dir = cfg.work_dir / f"extractions_{cfg.augment.model_id}"
    extract.run(cfg, items, run_dir, limit=limit)
    if extract_only:
        return

    # --- tier + filter ---
    pairs = items[["subject_title", "relation"]].drop_duplicates()
    fname2pair: Dict[str, tuple] = {}
    collisions = 0
    for title, relation in pairs.itertuples(index=False):
        key = f"{link.slugify(title)}_{link.slugify(relation)}"
        if key in fname2pair:
            collisions += 1
        fname2pair[key] = (title, relation)
    if collisions:
        log.warning(
            "%d filename collision(s) after slugifying; those pairs cannot be "
            "told apart and the last one wins",
            collisions,
        )

    extractions = link.collect_extractions(run_dir, fname2pair)
    log.info(
        "%d extractions from %d pair file(s)",
        len(extractions),
        extractions["source_file"].nunique(),
    )
    log.info("\n%s", extractions["tier_description"].value_counts().to_string())

    keep = extractions
    if cfg.augment.tier1_only:
        keep = keep[keep["tier"] == 1]
        keep = keep[keep["alignment_status"].isin(["match_exact"])]
    log.info("%d extraction(s) kept after the grounding filter", len(keep))
    if keep.empty:
        raise SystemExit("nothing left after filtering")

    # --- link ---
    title2qid = _title_index(cfg)
    qid2title = {qid: title for title, qid in title2qid.items()}
    linked = link.link_mentions(
        keep,
        cfg.work_dir / "html",
        title2qid,
        allow_submention=cfg.augment.submention_pass,
    )
    linked.to_csv(run_dir / "linked_results.csv", index=False)

    # --- merge ---
    needed = {q for q in linked["wikidata_qid"].dropna().unique()}
    qid2aliases = (
        items_mod.load_aliases(cfg.aliases_file, keep=needed)
        if cfg.aliases_file.exists()
        else {}
    )
    log.info("alias lists for %d/%d augmented entities", len(qid2aliases), len(needed))

    merged = link.merge(items, linked, qid2title, qid2aliases)

    out_path = cfg.work_dir / f"items_augmented_{cfg.augment.model_id}.jsonl"
    merged.to_json(out_path, orient="records", lines=True, force_ascii=False)
    log.info("wrote %s (%d items)", out_path, len(merged))
