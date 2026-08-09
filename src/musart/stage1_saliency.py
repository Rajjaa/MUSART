"""Stage 1: which relations are salient for which category.

The question this stage answers is *not* "which relations occur here" -- raw
frequency is dominated by relations that occur everywhere (``country``,
``instance of``). It is "which relations are characteristic of this kind of
article": ``spore print color`` for Biology, ``pole position`` for Motorsport.

The method is personalized PageRank over a graph that alternates between
relations, subjects and categories::

    relation --> subject --> category

Seeding the walk at one relation and letting it settle concentrates mass on the
categories whose articles that relation actually describes. A relation used by a
handful of mushroom articles sends nearly all of its mass to Biology; a relation
used everywhere spreads thin and clears no category's bar.

Categories are sinks (no outgoing edges), so they accumulate the walk's mass and
their scores are directly comparable across relations.

Selection is a single threshold on that score: a relation is salient for a
category when the category scores above ``saliency.threshold`` in that
relation's ranking. Because categories are sinks, the scale is comparable
across relations, so one global cut behaves sensibly for every category.

Inspect ``cat2relations.jsonl`` after a first run and adjust the threshold
before going further -- it is the one parameter that decides how selective the
benchmark is, and re-running this stage is cheap.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sknetwork.data import from_edge_list
from sknetwork.ranking import PageRank
from tqdm import tqdm

log = logging.getLogger(__name__)


def build_graph(items: pd.DataFrame):
    """Edges: subject -> category, relation -> subject.

    Directed and unweighted. A subject appears once per relation that uses it,
    but ``from_edge_list`` collapses duplicates, so a subject's out-edge to its
    category is not weighted by how many relations point at it.
    """
    edges: List[Tuple[str, str]] = []
    for subject, relation, category in zip(
        items["subject"], items["relation"], items["category"]
    ):
        edges.append((subject, category))
        edges.append((relation, subject))
    graph = from_edge_list(edges, weighted=False, directed=True)
    return graph


def rank_categories(
    items: pd.DataFrame,
    pid2label: Dict[str, str],
    top_k: int = 30,
    n_iter: int = 50,
    damping_factor: float = 0.999,
    solver: str = "piteration",
    progress: bool = True,
) -> Dict[str, List[Tuple[str, float]]]:
    """Run one personalized PageRank per relation; collect the categories it hits.

    Returns category -> [(relation label, score), ...] sorted by score desc.
    """
    graph = build_graph(items)
    adjacency = graph.adjacency
    names = graph.names
    name_to_index = {name: i for i, name in enumerate(names)}

    categories = set(items["category"].unique())
    cat2relations: Dict[str, List[Tuple[str, float]]] = {}

    relations = items["relation"].unique()
    for pid in tqdm(relations, desc="pagerank", disable=not progress):
        pagerank = PageRank(
            n_iter=n_iter, damping_factor=damping_factor, solver=solver
        )
        scores = pagerank.fit_predict(adjacency, {name_to_index[pid]: 1})
        ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        label = pid2label.get(pid)
        if label is None:
            continue
        for name, score in ranked[:top_k]:
            if name in categories:
                cat2relations.setdefault(name, []).append((label, round(float(score), 4)))

    for relations_for_cat in cat2relations.values():
        relations_for_cat.sort(key=lambda x: x[1], reverse=True)
    return cat2relations


def relation_posterior(
    items: pd.DataFrame, ranked: List[Tuple[str, float]], category: str
) -> List[Tuple[str, float]]:
    """Reweight PageRank by how common the relation is *within* the category.

    Diagnostic only -- MUSART selected on the raw PageRank score, and this
    column is emitted so a new corpus can be inspected both ways before
    committing to a threshold. It is not used for selection.
    """
    prior = (
        items.groupby("category")["relation_label"].value_counts(normalize=True).to_dict()
    )
    priors = np.array([prior.get((category, rel), 0.0) for rel, _ in ranked])
    scores = np.array([score for _, score in ranked])
    weighted = scores * priors
    total = weighted.sum()
    if total <= 0:
        return [(rel, 0.0) for rel, _ in ranked]
    weighted = weighted / total
    out = [(rel, round(float(score), 4)) for (rel, _), score in zip(ranked, weighted)]
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def run(
    items: pd.DataFrame,
    pid2label: Dict[str, str],
    threshold: float = 0.1,
    top_k: int = 30,
    n_iter: int = 50,
    damping_factor: float = 0.999,
    solver: str = "piteration",
    progress: bool = True,
) -> pd.DataFrame:
    """Full stage: rank, score, cut.

    ``items`` must already have the common filters applied.
    """
    ranked = rank_categories(
        items,
        pid2label,
        top_k=top_k,
        n_iter=n_iter,
        damping_factor=damping_factor,
        solver=solver,
        progress=progress,
    )

    rows = []
    for category, relations in ranked.items():
        salient = [rel for rel, score in relations if score > threshold]
        rows.append(
            {
                "category": category,
                "relations": [list(t) for t in relations],
                "relation_posterior": [
                    list(t) for t in relation_posterior(items, relations, category)
                ],
                "salient_relations": salient,
            }
        )
    df = pd.DataFrame(rows).sort_values("category").reset_index(drop=True)

    empty = df[df["salient_relations"].apply(len) == 0]["category"].tolist()
    if empty:
        log.warning(
            "%d categor%s no relation above threshold %.3f and will contribute "
            "nothing downstream: %s",
            len(empty),
            "ies have" if len(empty) > 1 else "y has",
            threshold,
            ", ".join(sorted(empty)[:10]) + (" ..." if len(empty) > 10 else ""),
        )
    return df


def write(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def read(path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def salient_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    """category -> salient relation labels.

    Accepts a file written by this stage, or one storing the same mapping under
    ``cat_wise_mapping`` as (relation, score) pairs.
    """
    if "salient_relations" in df.columns:
        return dict(zip(df["category"], df["salient_relations"]))
    if "cat_wise_mapping" in df.columns:
        return {
            cat: [pair[0] for pair in mapping]
            for cat, mapping in zip(df["category"], df["cat_wise_mapping"])
        }
    raise ValueError(
        "cat2relations file has neither 'salient_relations' nor 'cat_wise_mapping'; "
        f"columns are {list(df.columns)}"
    )
