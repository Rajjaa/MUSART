"""Command line entry point.

    musart collect     --config c.yaml   titles + categories -> Wikidata
    musart saliency    --config c.yaml   triples -> salient relations
    musart select      --config c.yaml   salient relations -> item set
    musart build       --config c.yaml   items -> questions + gold answers
    musart augment     --config c.yaml   items -> article-grounded gold answers
    musart repro-check --config c.yaml --against <jsonl>

Each stage reads what the one before it wrote, so they can be run separately and
resumed. ``musart run`` does stages 1-3 in sequence.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import config as config_mod
from . import corpus as corpus_mod
from . import items as items_mod
from . import stage1_saliency, stage2_select, stage3_build

log = logging.getLogger("musart")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _load_corpus(cfg: config_mod.Config) -> pd.DataFrame:
    """The normalised corpus, rebuilt from the spec if stage 0 has not run."""
    if cfg.corpus_file.exists():
        return corpus_mod.read(cfg.corpus_file)
    if cfg.corpus is None:
        raise SystemExit(
            f"no corpus: {cfg.corpus_file} does not exist and no 'corpus' is set "
            f"in the config"
        )
    return corpus_mod.load(cfg.corpus)


def _qid2category(cfg: config_mod.Config) -> Dict[str, str]:
    """subject QID -> category, via the corpus and the title index."""
    corpus = _load_corpus(cfg)
    if cfg.title2qid is not None:
        title2qid = items_mod.load_title2qid(cfg.title2qid)
    elif (cfg.work_dir / "title2qid.json").exists():
        title2qid = items_mod.load_json_map(cfg.work_dir / "title2qid.json")
    else:
        raise SystemExit(
            "no title -> QID index. Run 'musart collect', or set 'title2qid' in "
            "the config to a CSV with qid,title columns."
        )

    mapping: Dict[str, str] = {}
    for title, category in zip(corpus["title"], corpus["category"]):
        qid = title2qid.get(title)
        if qid is not None:
            mapping[qid] = category
    if not mapping:
        raise SystemExit(
            "no corpus article resolved to a QID -- check that the titles in "
            f"{cfg.corpus} match those in the title index"
        )
    log.info(
        "%d of %d corpus articles resolved to a QID", len(mapping), len(corpus)
    )
    return mapping


def _pid2label(cfg: config_mod.Config) -> Dict[str, str]:
    if not cfg.pid2label_file.exists():
        raise SystemExit(
            f"no property labels at {cfg.pid2label_file}. Run 'musart collect', "
            f"or set 'pid2label' in the config."
        )
    return items_mod.load_json_map(cfg.pid2label_file)


def _base_items(cfg: config_mod.Config, with_labels: bool):
    """Triples -> filtered item table. Shared by stages 1 and 2."""
    if not cfg.triples_file.exists():
        raise SystemExit(
            f"no triples at {cfg.triples_file}. Run 'musart collect', or set "
            f"'triples' in the config."
        )
    log.info("reading triples from %s", cfg.triples_file)
    triples = items_mod.read_triples(cfg.triples_file)
    log.info("%d triples", len(triples))

    qid2category = _qid2category(cfg)
    pid2label = _pid2label(cfg)

    table = items_mod.build(triples, qid2category, pid2label)
    log.info("%d items before filtering", len(table))

    qid2labels = items_mod.qid_to_labels(triples) if with_labels else {}
    return table, triples, qid2labels, pid2label


def cmd_saliency(cfg: config_mod.Config, args) -> None:
    table, _, _, pid2label = _base_items(cfg, with_labels=False)
    generic = items_mod.load_generic_relations(cfg.saliency.generic_relations)
    table = items_mod.apply_common_filters(
        table,
        cfg.saliency.min_category_items,
        cfg.saliency.min_relation_items,
        generic,
        on_stage=lambda stage, frame: log.info(
            "after dropping %s: %d items", stage, len(frame)
        ),
    )
    log.info(
        "graph: %d subjects, %d relations, %d categories",
        table["subject"].nunique(),
        table["relation"].nunique(),
        table["category"].nunique(),
    )
    result = stage1_saliency.run(
        table,
        pid2label,
        threshold=cfg.saliency.threshold,
        top_k=cfg.saliency.top_k,
        n_iter=cfg.saliency.n_iter,
        damping_factor=cfg.saliency.damping_factor,
        solver=cfg.saliency.solver,
    )
    stage1_saliency.write(result, cfg.cat2relations_file)
    total = int(result["salient_relations"].apply(len).sum())
    log.info(
        "wrote %s: %d categories, %d salient (category, relation) pairs",
        cfg.cat2relations_file,
        len(result),
        total,
    )


def cmd_select(cfg: config_mod.Config, args) -> None:
    if not cfg.cat2relations_file.exists():
        raise SystemExit(
            f"no salient relations at {cfg.cat2relations_file}. Run "
            f"'musart saliency' first."
        )
    table, triples, qid2labels, _ = _base_items(cfg, with_labels=True)
    generic = items_mod.load_generic_relations(cfg.saliency.generic_relations)
    table = items_mod.apply_common_filters(
        table,
        cfg.saliency.min_category_items,
        cfg.saliency.min_relation_items,
        generic,
    )

    qid2title = _title_index(cfg)
    table = items_mod.attach_labels(table, qid2labels, qid2title)

    salient = stage1_saliency.salient_map(
        stage1_saliency.read(cfg.cat2relations_file)
    )
    selected, funnel, ranges, cardinality = stage2_select.run(
        table,
        salient,
        categories=cfg.select.categories,
        min_relation_items_post=cfg.select.min_relation_items_post,
        min_multi_valued=cfg.select.min_multi_valued,
        exclude_pairs=cfg.select.exclude_pairs,
    )

    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    selected.to_json(cfg.selected_file, orient="records", lines=True, force_ascii=False)
    funnel.to_csv(cfg.work_dir / "funnel.csv", index=False)
    ranges.to_csv(cfg.work_dir / "cat_relation_range.csv", index=False)
    cardinality.to_csv(cfg.work_dir / "relation_cardinality.csv", index=False)

    print(funnel.to_string(index=False), file=sys.stderr)
    log.info(
        "wrote %s: %d items, %d categories, %d relations, %d subjects",
        cfg.selected_file,
        len(selected),
        selected["category"].nunique(),
        selected["relation_label"].nunique(),
        selected["subject"].nunique(),
    )


def _title_index(cfg: config_mod.Config) -> Dict[str, str]:
    """QID -> page title."""
    if cfg.title2qid is not None:
        title2qid = items_mod.load_title2qid(cfg.title2qid)
    elif (cfg.work_dir / "title2qid.json").exists():
        title2qid = items_mod.load_json_map(cfg.work_dir / "title2qid.json")
    else:
        raise SystemExit("no title -> QID index; run 'musart collect'")
    return {qid: title for title, qid in title2qid.items()}


def cmd_build(cfg: config_mod.Config, args) -> None:
    if not cfg.selected_file.exists():
        raise SystemExit(
            f"no selected items at {cfg.selected_file}. Run 'musart select' first."
        )
    selected = pd.read_json(cfg.selected_file, lines=True)
    cardinality_path = cfg.work_dir / "relation_cardinality.csv"
    if not cardinality_path.exists():
        raise SystemExit(
            f"no {cardinality_path}; re-run 'musart select' to produce it."
        )
    cardinality = pd.read_csv(cardinality_path)

    qid2title = _title_index(cfg)
    triples = items_mod.read_triples(cfg.triples_file)
    qid2labels = items_mod.qid_to_labels(triples)

    objects = sorted({obj for objs in selected["object"] for obj in objs})
    if not cfg.aliases_file.exists():
        raise SystemExit(
            f"no aliases at {cfg.aliases_file}. Run 'musart collect', or set "
            f"'aliases' in the config."
        )
    log.info("loading aliases for %d objects from %s", len(objects), cfg.aliases_file)
    qid2aliases = items_mod.load_aliases(cfg.aliases_file, keep=objects)

    result = stage3_build.run(
        selected,
        cardinality,
        qid2aliases,
        qid2title,
        qid2labels,
        template_path=cfg.build.templates,
        template_llm=cfg.build.template_llm,
        template_model=cfg.build.template_model,
        n_icl=cfg.build.n_icl,
        seed=cfg.build.seed,
    )
    stage3_build.write(result, cfg.items_file)
    log.info(
        "wrote %s: %d items, %d categories, %d relations",
        cfg.items_file,
        len(result),
        result["category"].nunique(),
        result["relation"].nunique(),
    )


def cmd_collect(cfg: config_mod.Config, args) -> None:
    from . import stage0_collect

    stage0_collect.run(cfg, limit=args.limit)


def cmd_augment(cfg: config_mod.Config, args) -> None:
    from .augment import pipeline

    pipeline.run(cfg, limit=args.limit, extract_only=args.extract_only)


def cmd_fetch_articles(cfg: config_mod.Config, args) -> None:
    from .augment import fetch_articles

    fetch_articles.run(cfg, limit=args.limit)


def cmd_run(cfg: config_mod.Config, args) -> None:
    cmd_saliency(cfg, args)
    cmd_select(cfg, args)
    cmd_build(cfg, args)


def cmd_repro_check(cfg: config_mod.Config, args) -> None:
    from . import repro

    ok = repro.compare(cfg.items_file, Path(args.against))
    raise SystemExit(0 if ok else 1)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="musart", description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="fetch Wikidata for the corpus")
    p.add_argument("--limit", type=int, help="only the first N articles (smoke test)")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("saliency", help="rank relations by category")
    p.set_defaults(func=cmd_saliency)

    p = sub.add_parser("select", help="apply the selection funnel")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("build", help="questions, gold answers, exemplars")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("run", help="saliency + select + build")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("fetch-articles", help="download article text and HTML")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_fetch_articles)

    p = sub.add_parser("augment", help="article-grounded gold answers (needs an API key)")
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--extract-only",
        action="store_true",
        help="run extraction but not linking",
    )
    p.set_defaults(func=cmd_augment)

    p = sub.add_parser("repro-check", help="diff the built items against a reference")
    p.add_argument("--against", required=True, help="reference items jsonl")
    p.set_defaults(func=cmd_repro_check)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = config_mod.load(args.config)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
