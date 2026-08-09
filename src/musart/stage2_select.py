"""Stage 2: the funnel from salient relations down to the final item set.

Being salient is necessary but not sufficient. A relation also has to be
*askable*: frequent enough to be worth a domain-level question, consistent
enough in arity that an answer can be graded, and pointing at objects that
resolve to something a model could plausibly say.

The stages, in order, with MUSART's counts:

    salient join            153,290 -> 108,443
    category whitelist      108,443 ->  60,770
    relation frequency       60,770 ->  55,753
    cardinality              55,753 ->  22,775
    unresolvable objects     22,775 ->  22,756
    (category, relation)     22,756 ->  21,775

Every step writes a row to ``funnel.csv``, which is the table to quote when
describing the construction.
"""

from __future__ import annotations

import difflib
import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import items as items_mod

log = logging.getLogger(__name__)


class Funnel:
    """Records what each filter cost, so the drop is attributable."""

    def __init__(self) -> None:
        self.rows: List[Dict] = []

    def record(self, stage: str, frame: pd.DataFrame) -> None:
        summary = items_mod.summarise(frame)
        previous = self.rows[-1]["items"] if self.rows else None
        summary["stage"] = stage
        summary["dropped"] = (previous - summary["items"]) if previous is not None else 0
        self.rows.append(summary)

    def to_frame(self) -> pd.DataFrame:
        cols = ["stage", "items", "dropped", "subjects", "relations", "categories", "triples"]
        return pd.DataFrame(self.rows)[cols]


def check_whitelist(requested: Sequence[str], available: Sequence[str]) -> List[str]:
    """Warn about whitelist entries that match nothing, with a spelling suggestion.

    A whitelist is a hand-edited list of strings checked against category names
    the author is not looking at directly. An entry that matches nothing drops a
    whole domain and changes nothing else, so without this check a single
    misspelling is indistinguishable from a deliberate omission.
    """
    available_set = set(available)
    unmatched = [c for c in requested if c not in available_set]
    for name in unmatched:
        close = difflib.get_close_matches(name, available_set, n=1, cutoff=0.7)
        hint = f"; did you mean {close[0]!r}?" if close else ""
        log.warning("category %r in the whitelist matches no data%s", name, hint)
    return unmatched


def classify_cardinality(
    items: pd.DataFrame, min_multi_valued: int
) -> pd.DataFrame:
    """Split relations into single-valued, reliably multi-valued, and neither.

    A relation is gradeable when its arity is predictable. ``date of birth`` is
    always one value; ``cast member`` is reliably many. The dangerous case is in
    between -- a relation that is usually single but occasionally multi gives a
    model no way to know how many answers are wanted, and penalises it either
    way. Those are dropped.
    """
    multi = items.groupby("relation_label")["multi_valued"].agg(Counter)
    frame = multi.reset_index()
    frame.columns = ["relation_label", "multi_valued"]
    # Counter[1] is how many items of this relation had more than one object.
    frame["enough_multi_valued"] = frame["multi_valued"].apply(
        lambda c: 1 if c.get(1, 0) >= min_multi_valued else 0
    )
    # No item of this relation ever had more than one object.
    frame["single_valued"] = frame["multi_valued"].apply(
        lambda c: 1 if 1 not in c else 0
    )
    return frame


def category_relation_range(items: pd.DataFrame) -> pd.DataFrame:
    """How many distinct objects each (category, relation) pair ever takes.

    A pair with a handful of possible answers is not a knowledge probe -- it is
    a multiple-choice question a model can guess. Emitted as a table rather than
    applied automatically, since where to draw the line is a judgement call that
    depends on the corpus.
    """
    rows = []
    for (category, relation), group in items.groupby(["category", "relation_label"]):
        objects = [obj for objs in group["object"] for obj in objs]
        counts = Counter(objects)
        rows.append(
            {
                "category": category,
                "relation": relation,
                "items": len(group),
                "distinct_objects": len(counts),
                "top_objects": "; ".join(f"{o} ({n})" for o, n in counts.most_common(5)),
            }
        )
    return pd.DataFrame(rows).sort_values("distinct_objects").reset_index(drop=True)


def run(
    items: pd.DataFrame,
    salient: Dict[str, List[str]],
    categories: Optional[Sequence[str]] = None,
    min_relation_items_post: int = 105,
    min_multi_valued: int = 100,
    exclude_pairs: Sequence[Tuple[str, str]] = (),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the funnel.

    Returns (selected items, funnel report, range table, cardinality table).

    ``items`` must already have the common filters and labels attached.
    """
    funnel = Funnel()
    funnel.record("input", items)

    # 1. Salient relations only.
    is_salient = [
        relation in salient.get(category, ())
        for category, relation in zip(items["category"], items["relation_label"])
    ]
    items = items[pd.Series(is_salient, index=items.index)]
    funnel.record("salient relations", items)

    # 2. Category whitelist.
    if categories:
        check_whitelist(categories, items["category"].unique())
        items = items[items["category"].isin(list(categories))]
        funnel.record("category whitelist", items)

    # 3. Relation frequency, recomputed after the whitelist: a relation salient
    #    only in a category we dropped should not keep its old count.
    counts = items["relation_label"].value_counts()
    frequent = counts.index[counts >= min_relation_items_post]
    items = items[items["relation_label"].isin(frequent)]
    funnel.record(f"relations with < {min_relation_items_post} items", items)

    # 4. Cardinality.
    items = items.copy()
    items["cardinality"] = items["object"].apply(len)
    items["multi_valued"] = (items["cardinality"] > 1).astype(int)
    cardinality = classify_cardinality(items, min_multi_valued)
    keep = set(
        cardinality[cardinality["single_valued"] == 1]["relation_label"]
    ) | set(cardinality[cardinality["enough_multi_valued"] == 1]["relation_label"])
    items = items[items["relation_label"].isin(keep)]
    funnel.record("ambiguous cardinality", items)

    # 5. Objects that resolve to no surface form at all cannot be answered.
    items = items[items["object_label"].apply(all)]
    funnel.record("unresolvable objects", items)

    # Stage 3 needs to know which relations to draw multi-valued exemplars for.
    # Classified before the pair exclusion, so that dropping a pair cannot flip
    # a relation's class and change the few-shot prompts as a side effect.
    cardinality = classify_cardinality(items, min_multi_valued)

    # 6. Hand-excluded (category, relation) pairs.
    if exclude_pairs:
        excluded = {tuple(p) for p in exclude_pairs}
        mask = [
            (category, relation) not in excluded
            for category, relation in zip(items["category"], items["relation_label"])
        ]
        items = items[pd.Series(mask, index=items.index)]
        funnel.record("excluded pairs", items)

    items = items.reset_index(drop=True)

    # Counter objects do not serialise; the two flags carry all stage 3 needs.
    cardinality = cardinality[
        ["relation_label", "single_valued", "enough_multi_valued"]
    ].reset_index(drop=True)

    return items, funnel.to_frame(), category_relation_range(items), cardinality
