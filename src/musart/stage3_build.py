"""Stage 3: turn selected items into askable questions with gradeable answers.

Three things happen here:

* **Gold answers.** An object is a QID; a model answers with a string. Every
  surface form Wikidata knows for that entity -- aliases, labels, page title --
  is collected, so a model saying "Capcom" is not marked wrong because the
  canonical label is "Capcom Co., Ltd.".
* **Questions.** The relation's template with the subject substituted in.
* **In-context exemplars.** Five worked examples per (category, relation) for
  the few-shot prompt, drawn from the same distribution as the item itself.

Exemplar selection shuffles the candidate items, so ``build.seed`` determines
which examples a build produces.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

from . import templates as templates_mod

log = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "subject",
    "object",
    "subject_title",
    "category",
    "relation",
    "question",
    "icl_examples",
    "ground_truth",
]


def ground_truth_forms(
    objects: Sequence[str],
    qid2aliases: Dict[str, List[str]],
    qid2title: Dict[str, str],
    qid2labels: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Every string that should count as naming this object."""
    result: Dict[str, List[str]] = defaultdict(list)
    for qid in objects:
        result[qid].extend(qid2aliases.get(qid, []))
        if qid in qid2title:
            result[qid].append(qid2title[qid])
        if qid in qid2labels:
            result[qid].extend(qid2labels[qid])
    return dict(result)


def _answer_forms(
    objects: Sequence[str],
    qid2title: Dict[str, str],
    qid2labels: Dict[str, List[str]],
) -> Optional[List[str]]:
    """One display string per object, preferring the page title.

    Returns None when any object is a Wikipedia portal -- portals are navigation
    pages, not entities, and reading like an answer in a prompt would teach the
    model the wrong output shape.
    """
    forms = []
    for qid in objects:
        if qid in qid2title:
            if "Portal:" in qid2title[qid]:
                return None
            forms.append(qid2title[qid])
        elif qid in qid2labels and qid2labels[qid]:
            forms.append(qid2labels[qid][0])
    return forms


def select_exemplars(
    items: pd.DataFrame,
    multi_valued_relations: Set[str],
    single_valued_relations: Set[str],
    qid2title: Dict[str, str],
    qid2labels: Dict[str, List[str]],
    n_icl: int = 5,
    seed: int = 0,
) -> Dict[tuple, List[Dict]]:
    """Pick up to ``n_icl`` worked examples per (category, relation).

    Two rules shape the choice. For a multi-valued relation only items that
    actually have several objects are eligible, so the examples demonstrate that
    several answers are wanted. And an item is skipped when every one of its
    objects already appeared in an earlier example, which stops five exemplars
    from being five ways of saying "Nintendo" and gives the model a sense of the
    answer space instead.
    """
    exemplars: Dict[tuple, List[Dict]] = defaultdict(list)

    for category in items["category"].unique():
        in_category = items[items["category"] == category]
        for relation in in_category["relation_label"].unique():
            is_multi = relation in multi_valued_relations
            if not is_multi and relation not in single_valued_relations:
                # Neither class: stage 2 should have dropped it. No exemplars
                # rather than misleading ones.
                continue

            candidates = in_category[in_category["relation_label"] == relation]
            candidates = candidates.sample(frac=1, random_state=seed)

            used_objects: Set[str] = set()
            key = (category, relation)
            for _, row in candidates.iterrows():
                if is_multi and row["cardinality"] == 1:
                    continue
                objects = list(row["object"])
                if all(obj in used_objects for obj in objects):
                    continue
                used_objects.update(objects)
                forms = _answer_forms(objects, qid2title, qid2labels)
                if forms is None:
                    continue
                exemplars[key].append({"question": row["question"], "answer": forms})
                if len(exemplars[key]) >= n_icl:
                    break

    return exemplars


def format_exemplars(examples: Sequence[Dict]) -> str:
    """Question then answers, one per line, blank line between examples."""
    prompt = ""
    for example in examples:
        prompt += f"{example['question']}\n"
        prompt += "\n".join(example["answer"])
        prompt += "\n\n"
    return prompt


def run(
    items: pd.DataFrame,
    cardinality: pd.DataFrame,
    qid2aliases: Dict[str, List[str]],
    qid2title: Dict[str, str],
    qid2labels: Dict[str, List[str]],
    template_path,
    template_llm: str = "auto",
    template_model: Optional[str] = None,
    n_icl: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    items = items.copy()

    items["ground_truth"] = items["object"].apply(
        lambda objs: ground_truth_forms(objs, qid2aliases, qid2title, qid2labels)
    )

    relations = sorted(items["relation_label"].unique())
    rel2template = templates_mod.resolve(
        relations, template_path, provider=template_llm, model=template_model
    )
    items["question"] = [
        templates_mod.render(rel2template[relation], title)
        for relation, title in zip(items["relation_label"], items["subject_title"])
    ]

    if "cardinality" not in items.columns:
        items["cardinality"] = items["object"].apply(len)

    multi = set(
        cardinality[cardinality["enough_multi_valued"] == 1]["relation_label"]
    )
    single = set(cardinality[cardinality["single_valued"] == 1]["relation_label"])
    exemplars = select_exemplars(
        items, multi, single, qid2title, qid2labels, n_icl=n_icl, seed=seed
    )

    short = [
        f"{cat}/{rel}"
        for (cat, rel), examples in exemplars.items()
        if len(examples) < n_icl
    ]
    if short:
        log.warning(
            "%d (category, relation) pair(s) yielded fewer than %d exemplars: %s",
            len(short),
            n_icl,
            ", ".join(sorted(short)[:10]) + (" ..." if len(short) > 10 else ""),
        )

    items["icl_examples"] = [
        format_exemplars(exemplars.get((category, relation), []))
        for category, relation in zip(items["category"], items["relation_label"])
    ]

    # The item table carries both the property id and its label under
    # 'relation' and 'relation_label'. The published schema exposes the label
    # as 'relation', so the id column goes before the rename collides with it.
    items = items.drop(columns=["relation"]).rename(
        columns={"relation_label": "relation"}
    )
    return items[OUTPUT_COLUMNS].reset_index(drop=True)


def write(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
