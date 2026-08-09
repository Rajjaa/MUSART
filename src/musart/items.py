"""Turn a triples table into the item table both stage 1 and stage 2 work on.

An *item* is one (subject, relation) pair with all of its objects collected --
that is, one question and its complete gold answer set. Everything downstream
counts items, not triples.

Stage 1 and stage 2 build the same table from the same file and apply the same
three opening filters; only what they do afterwards differs. Keeping that shared
prefix in one place is what makes their counts comparable.
"""

from __future__ import annotations

import ast
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

TRIPLE_COLUMNS = ["subject", "relation", "object", "subject_label", "object_label"]


def _parse_labels(value) -> List[str]:
    """Labels are stored as a Python list literal, the shape the dump scripts emit."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [x for x in parsed if isinstance(x, str)]


def read_triples(path: Path) -> pd.DataFrame:
    """Read the tab-separated triples table.

    Columns, no header: subject, relation, object, subject_label, object_label.
    The two label columns hold Python list literals. A 3-column file (no labels)
    is also accepted -- stage 1 does not need labels.
    """
    df = pd.read_csv(path, delimiter="\t", header=None, dtype=str)
    if df.shape[1] == 3:
        df.columns = TRIPLE_COLUMNS[:3]
        df["subject_label"] = [[] for _ in range(len(df))]
        df["object_label"] = [[] for _ in range(len(df))]
    elif df.shape[1] >= 5:
        df = df.iloc[:, :5]
        df.columns = TRIPLE_COLUMNS
        df["subject_label"] = df["subject_label"].apply(_parse_labels)
        df["object_label"] = df["object_label"].apply(_parse_labels)
    else:
        raise ValueError(
            f"{path}: expected 3 or 5 tab-separated columns, found {df.shape[1]}"
        )
    # Deduplicate the label lists but keep row order stable.
    df["subject_label"] = df["subject_label"].apply(lambda l: list(dict.fromkeys(l)))
    df["object_label"] = df["object_label"].apply(lambda l: list(dict.fromkeys(l)))
    return df


def qid_to_labels(triples: pd.DataFrame) -> Dict[str, List[str]]:
    """Collect every label seen for a QID, whether it appeared as subject or object."""
    out: Dict[str, List[str]] = defaultdict(list)
    for subj, obj, s_labels, o_labels in zip(
        triples["subject"],
        triples["object"],
        triples["subject_label"],
        triples["object_label"],
    ):
        out[subj].extend(s_labels)
        out[obj].extend(o_labels)
    return {k: list(dict.fromkeys(v)) for k, v in out.items() if v}


def load_generic_relations(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"generic relations file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.lstrip().startswith("#")]


def load_json_map(path: Path) -> Dict[str, str]:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def load_title2qid(path: Path) -> Dict[str, str]:
    """Read a ``qid,title`` CSV into title -> QID.

    The dump-derived index stores the QID as a bare integer; a ``Q`` prefix is
    added when it is missing so either shape works.
    """
    df = pd.read_csv(path, dtype=str)
    missing = {"qid", "title"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing column(s) {', '.join(sorted(missing))}; "
            f"found {', '.join(df.columns)}"
        )
    df = df.dropna(subset=["qid", "title"])
    qids = df["qid"].apply(lambda q: q if q.startswith("Q") else f"Q{q}")
    return dict(zip(df["title"], qids))


def load_aliases(path: Path, keep: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """Read a ``qid,alias`` JSONL into QID -> aliases, optionally filtered.

    The unfiltered file is hundreds of megabytes and only a few tens of
    thousands of entities are ever asked about, so ``keep`` is worth passing.
    """
    wanted = set(keep) if keep is not None else None
    out: Dict[str, List[str]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = record.get("qid")
            if qid is None or (wanted is not None and qid not in wanted):
                continue
            alias = record.get("alias") or []
            out[qid] = alias if isinstance(alias, list) else [alias]
    return out


def build(
    triples: pd.DataFrame,
    qid2category: Dict[str, str],
    pid2label: Dict[str, str],
) -> pd.DataFrame:
    """Group triples into items and attach category and relation label.

    Subjects outside the corpus, and relations with no English label, are
    dropped -- neither can produce an answerable question.
    """
    items = (
        triples.groupby(by=["subject", "relation"])
        .agg({"object": "unique"})
        .reset_index()
    )
    items["category"] = items["subject"].map(qid2category)
    items = items[items["category"].notna()]
    items["relation_label"] = items["relation"].map(pid2label)
    items = items[items["relation_label"].notna()]
    return items.reset_index(drop=True)


def attach_labels(
    items: pd.DataFrame,
    qid2labels: Dict[str, List[str]],
    qid2title: Dict[str, str],
) -> pd.DataFrame:
    """Add the surface forms stage 2 and stage 3 need.

    ``object_label`` is a list parallel to ``object``; an entry is None when the
    QID has neither a label nor a page title, which is what the
    unresolvable-object filter looks for.
    """
    items = items.copy()
    items["subject_title"] = items["subject"].map(qid2title)
    items["object_title"] = items["object"].apply(
        lambda qids: [qid2title.get(q) for q in qids]
    )
    items["subject_label"] = items["subject"].apply(lambda q: qid2labels.get(q, []))
    items["object_label"] = items["object"].apply(
        lambda qids: [qid2labels.get(q, qid2title.get(q)) for q in qids]
    )
    items["cardinality"] = items["object"].apply(len)
    return items


def apply_common_filters(
    items: pd.DataFrame,
    min_category_items: int,
    min_relation_items: int,
    generic_relations: Sequence[str],
    on_stage=None,
) -> pd.DataFrame:
    """The three filters stage 1 and stage 2 both apply, in the same order.

    Note the ordering: rare categories go first, so the relation counts that
    decide the second filter are computed over the surviving categories only.
    """
    counts = items["category"].value_counts()
    rare_categories = counts.index[counts < min_category_items]
    items = items[~items["category"].isin(rare_categories)]
    if on_stage:
        on_stage(f"categories with < {min_category_items} items", items)

    counts = items["relation_label"].value_counts()
    rare_relations = counts.index[counts < min_relation_items]
    items = items[~items["relation_label"].isin(rare_relations)]
    if on_stage:
        on_stage(f"relations with < {min_relation_items} items", items)

    items = items[~items["relation_label"].isin(list(generic_relations))]
    if on_stage:
        on_stage("generic relations", items)

    return items.reset_index(drop=True)


def summarise(items: pd.DataFrame) -> Dict[str, int]:
    return {
        "items": len(items),
        "subjects": items["subject"].nunique(),
        "relations": items["relation_label"].nunique(),
        "categories": items["category"].nunique(),
        "triples": int(items["object"].apply(len).sum()) if len(items) else 0,
    }
