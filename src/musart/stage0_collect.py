"""Stage 0: everything the pipeline needs about a corpus, from its titles.

Given article titles and their categories, this produces the four files the rest
of the pipeline reads::

    triples.csv       subject, relation, object, subject_label, object_label
    pid2label.json    property id -> English label
    title2qid.json    article title -> QID
    qid2aliases.jsonl QID -> aliases

The published MUSART was built from Wikidata dumps instead -- a 20 GB category
table and an 8 GB label dump, processed by shell scripts. That works, but it
makes the benchmark reproducible only by someone willing to download and grind
through those dumps. Going through the APIs takes a few minutes for a corpus of
this size and needs nothing but a network connection.

The trade-off is real and worth stating: the dumps are a frozen snapshot, the
APIs are live. Rebuilding MUSART through this stage today would not reproduce
the May 2025 data, because Wikidata has changed since. Use it to build a *new*
benchmark; use the dump-derived files to reproduce the published one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import corpus as corpus_mod
from . import wikidata
from .config import Config

log = logging.getLogger(__name__)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def run(cfg: Config, limit: Optional[int] = None) -> None:
    if cfg.corpus is None:
        raise SystemExit("set 'corpus' in the config before running collect")

    frame = corpus_mod.load(cfg.corpus)
    if limit:
        frame = frame.head(limit)
        log.info("limited to the first %d articles", len(frame))
    corpus_mod.write(frame, cfg.corpus_file)
    log.info("%d articles in %d categories", len(frame), frame["category"].nunique())

    client = wikidata.WikidataClient(
        cfg.cache_dir, contact=cfg.contact, delay=cfg.api_delay
    )

    # 1. Titles -> QIDs.
    titles = frame["title"].tolist()
    title2qid = client.titles_to_qids(titles)
    unresolved = len(titles) - len(title2qid)
    if unresolved:
        log.warning(
            "%d of %d titles did not resolve to a Wikidata item and will be "
            "dropped (deleted pages, disambiguation pages, or titles that no "
            "longer exist)",
            unresolved,
            len(titles),
        )
    _write_json(cfg.work_dir / "title2qid.json", title2qid)

    subjects = sorted(set(title2qid.values()))
    log.info("%d distinct subjects", len(subjects))

    # 2. Statements on each subject.
    claims = client.claims(subjects)
    triples: List[Dict[str, str]] = []
    for subject, pairs in claims.items():
        for pid, obj in pairs:
            triples.append({"subject": subject, "relation": pid, "object": obj})
    log.info("%d object-valued truthy triples", len(triples))
    if not triples:
        raise SystemExit(
            "no triples were collected -- check that the corpus titles are real "
            "English Wikipedia articles"
        )

    # 3. Labels and aliases for everything that appears in a triple.
    entities = sorted(
        {t["subject"] for t in triples} | {t["object"] for t in triples}
    )
    log.info("fetching labels and aliases for %d entities", len(entities))
    info = client.labels_and_aliases(entities)

    qid2label = {q: v["label"] for q, v in info.items() if v.get("label")}
    qid2aliases = {q: v["aliases"] for q, v in info.items() if v.get("aliases")}

    # 4. Property labels. Properties are entities too, so the same call works.
    pids = sorted({t["relation"] for t in triples})
    log.info("fetching labels for %d properties", len(pids))
    property_info = client.labels_and_aliases(pids)
    pid2label = {p: v["label"] for p, v in property_info.items() if v.get("label")}

    # --- write in the schemas the rest of the pipeline expects --------------

    table = pd.DataFrame(triples)
    # The label columns hold Python list literals, matching the dump scripts'
    # output, so a dump-derived triples file is a drop-in replacement.
    table["subject_label"] = table["subject"].apply(
        lambda q: repr([qid2label[q]] if q in qid2label else [])
    )
    table["object_label"] = table["object"].apply(
        lambda q: repr([qid2label[q]] if q in qid2label else [])
    )
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        cfg.work_dir / "triples.csv", sep="\t", header=False, index=False
    )

    _write_json(cfg.pid2label_file, pid2label)
    _write_json(cfg.work_dir / "qid2label.json", qid2label)

    with cfg.aliases_file.open("w", encoding="utf-8") as fh:
        for qid in sorted(qid2aliases):
            fh.write(
                json.dumps({"qid": qid, "alias": qid2aliases[qid]}, ensure_ascii=False)
                + "\n"
            )

    log.info(
        "wrote %s (%d triples), %s (%d properties), %s (%d entities with aliases)",
        cfg.work_dir / "triples.csv",
        len(table),
        cfg.pid2label_file,
        len(pid2label),
        cfg.aliases_file,
        len(qid2aliases),
    )
