"""Normalise a corpus spec into one row per article.

Two input shapes are accepted:

* **Nested JSON**, the shape Wikipedia's Good Articles index has and the one
  ``assets/good_articles.json`` ships::

      {"Sports and recreation": {"Football": ["Pelé", ...], "Motorsport": [...]}}

  Nesting may be any depth; the leaves are lists of article titles.

* **Flat CSV** with ``title`` and ``category`` columns.

Both become ``corpus.jsonl`` with ``title``, ``category_path`` and ``category``.

``category`` is the level of the hierarchy the benchmark treats as a domain. For
a nested spec that is the *second* level: the top level is a broad heading
("Sports and recreation")
that groups domains too dissimilar to share a relation vocabulary, while the
second is specific enough that "what makes a relation salient here" is a
meaningful question ("Football", "Motorsport").
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pandas as pd

log = logging.getLogger(__name__)

# Wikipedia's own index nests everything under a "Contents" bucket that is
# navigation scaffolding, not a topic.
DROP_CATEGORIES = {"Contents"}


def _walk(node: Any, path: List[str]) -> Iterator[Dict[str, Any]]:
    if isinstance(node, dict):
        for key, child in node.items():
            yield from _walk(child, path + [key])
    elif isinstance(node, list):
        for title in node:
            if isinstance(title, str):
                yield {"title": title, "category_path": path}


def from_nested_json(path: Path) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8") as fh:
        hierarchy = json.load(fh)
    rows = list(_walk(hierarchy, []))
    if not rows:
        raise ValueError(f"{path}: no article titles found")
    df = pd.DataFrame(rows)
    # Second level where it exists, first level otherwise, so a one-level spec
    # still works.
    df["category"] = df["category_path"].apply(lambda p: p[1] if len(p) > 1 else p[0])
    return df


def from_csv(path: Path) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: empty")
    missing = {"title", "category"} - set(rows[0])
    if missing:
        raise ValueError(
            f"{path}: missing column(s) {', '.join(sorted(missing))}; "
            f"found {', '.join(rows[0])}"
        )
    df = pd.DataFrame([{"title": r["title"], "category": r["category"]} for r in rows])
    df["category_path"] = df["category"].apply(lambda c: [c])
    return df


def load(path: Path) -> pd.DataFrame:
    """Read either corpus shape and return normalised rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"corpus spec not found: {path}")
    df = from_csv(path) if path.suffix.lower() == ".csv" else from_nested_json(path)

    df = df[~df["category"].isin(DROP_CATEGORIES)]
    df = df[df["title"].astype(bool)]

    # An article can be filed under several categories -- "The X-Files" is both
    # Film and Television, "Species" is both Film and Biology and medicine. The
    # benchmark needs one domain per subject, and the last listing wins.
    #
    # That is an arbitrary tie-break, and it decides which domain the article's
    # questions end up in, so it is reported rather than applied in silence.
    ambiguous = df["title"].duplicated(keep=False)
    if ambiguous.any():
        titles = df.loc[ambiguous, "title"].nunique()
        examples = (
            df[ambiguous]
            .groupby("title")["category"]
            .apply(lambda c: "/".join(c))
            .head(3)
        )
        log.warning(
            "%d article(s) are filed under more than one category; keeping the "
            "last listing for each. Examples: %s",
            titles,
            "; ".join(f"{t} -> {c}" for t, c in examples.items()),
        )
    df = df.drop_duplicates(subset="title", keep="last").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path}: every article was dropped as navigation scaffolding")
    return df[["title", "category_path", "category"]]


def write(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def read(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)
