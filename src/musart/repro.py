"""Compare a built dataset against a reference, field by field.

Used to check that a change to the pipeline did not change the data. Items are
matched on (subject, relation, category), not on row order.

``icl_examples`` is reported but never counted as a failure: exemplar selection
is randomised, so two builds agree on it only when they used the same seed and
the same candidate pool. Every other field must match exactly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

log = logging.getLogger(__name__)

KEY = ["subject", "relation", "category"]
SOFT_FIELDS = {"icl_examples"}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Reference files vary in whether they name the field ground_truth or
    # ground_truth_forms, and category or second_category.
    df = df.rename(
        columns={
            "second_category": "category",
            "relation_label": "relation",
            "ground_truth_forms": "ground_truth",
        }
    )
    for column in ("object",):
        if column in df.columns:
            df[column] = df[column].apply(
                lambda v: sorted(v) if isinstance(v, (list, tuple)) else v
            )
    if "ground_truth" in df.columns:
        df["ground_truth"] = df["ground_truth"].apply(
            lambda d: {k: sorted(set(v)) for k, v in d.items()}
            if isinstance(d, dict)
            else d
        )
    return df


def compare(built_path: Path, reference_path: Path) -> bool:
    built = _normalise(pd.read_json(built_path, lines=True))
    reference = _normalise(pd.read_json(reference_path, lines=True))

    print(f"built     {built_path}: {len(built):,} items")
    print(f"reference {reference_path}: {len(reference):,} items")

    ok = True
    for name, frame in (("built", built), ("reference", reference)):
        print(
            f"  {name:9} {frame['category'].nunique()} categories, "
            f"{frame['relation'].nunique()} relations, "
            f"{frame['subject'].nunique()} subjects"
        )

    if len(built) != len(reference):
        print(f"FAIL item count differs by {len(built) - len(reference):+,}")
        ok = False

    built_keys = set(map(tuple, built[KEY].values))
    reference_keys = set(map(tuple, reference[KEY].values))
    only_built = built_keys - reference_keys
    only_reference = reference_keys - built_keys
    if only_built or only_reference:
        ok = False
        print(f"FAIL {len(only_built):,} items only in built, "
              f"{len(only_reference):,} only in reference")
        for key in list(only_built)[:5]:
            print(f"       only built: {key}")
        for key in list(only_reference)[:5]:
            print(f"   only reference: {key}")

    shared = sorted(
        (set(built.columns) & set(reference.columns)) - set(KEY)
    )
    left = built.set_index(KEY).sort_index()
    right = reference.set_index(KEY).sort_index()
    common = left.index.intersection(right.index)
    left, right = left.loc[common], right.loc[common]

    for field in shared:
        differing = [
            key
            for key, a, b in zip(common, left[field], right[field])
            if json.dumps(a, sort_keys=True, default=str)
            != json.dumps(b, sort_keys=True, default=str)
        ]
        if not differing:
            print(f"  OK   {field}: identical across {len(common):,} items")
            continue
        share = len(differing) / max(len(common), 1)
        if field in SOFT_FIELDS:
            print(
                f"  note {field}: differs on {len(differing):,} of {len(common):,} "
                f"items ({share:.1%}) -- exemplar selection is randomised, "
                f"compared separately"
            )
            continue
        ok = False
        print(f"FAIL {field}: differs on {len(differing):,} of {len(common):,} items")
        for key in differing[:3]:
            print(f"       {key}")
            print(f"         built     {str(left.loc[key, field])[:160]}")
            print(f"         reference {str(right.loc[key, field])[:160]}")

    print("PASS" if ok else "FAIL")
    return ok
