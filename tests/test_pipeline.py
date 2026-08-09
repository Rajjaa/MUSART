"""Unit tests on synthetic fixtures.

The MUSART regression (``musart repro-check``) is the real proof the pipeline is
faithful, but it needs the published inputs and takes minutes. These cover the
individual decisions in isolation, including the two that produced real defects
in the shipped dataset: a whitelist entry that matched nothing, and an exemplar
sampler that was not seeded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from musart import corpus, items as items_mod, stage1_saliency, stage2_select, stage3_build
from musart import templates as templates_mod
from musart.augment import link
from musart.wikidata import _merge_aliases, _pick_label, truthy_entity_claims


# --- corpus ----------------------------------------------------------------


def test_nested_corpus_uses_second_level_as_category(tmp_path):
    spec = tmp_path / "c.json"
    spec.write_text(json.dumps({"Arts": {"Film": ["A", "B"], "Music": ["C"]}}))
    frame = corpus.load(spec)
    assert set(frame["category"]) == {"Film", "Music"}
    assert frame.loc[frame["title"] == "A", "category_path"].iloc[0] == ["Arts", "Film"]


def test_corpus_drops_navigation_and_keeps_last_category(tmp_path):
    spec = tmp_path / "c.json"
    spec.write_text(
        json.dumps({"Top": {"Contents": ["Nav"], "Film": ["X"], "TV": ["X"]}})
    )
    frame = corpus.load(spec)
    assert "Nav" not in set(frame["title"])
    # A title under two categories keeps the last, matching the notebook's
    # dict-assignment order. Getting this backwards costs 15 items on MUSART.
    assert frame.loc[frame["title"] == "X", "category"].iloc[0] == "TV"


def test_flat_csv_corpus(tmp_path):
    spec = tmp_path / "c.csv"
    spec.write_text("title,category\nA,Film\nB,Music\n")
    assert set(corpus.load(spec)["category"]) == {"Film", "Music"}


def test_csv_missing_column_is_rejected(tmp_path):
    spec = tmp_path / "c.csv"
    spec.write_text("title,topic\nA,Film\n")
    with pytest.raises(ValueError, match="category"):
        corpus.load(spec)


# --- wikidata --------------------------------------------------------------


def test_truthy_prefers_preferred_rank():
    claims = {
        "P1": [
            _statement("Q1", "normal"),
            _statement("Q2", "preferred"),
            _statement("Q3", "deprecated"),
        ]
    }
    assert truthy_entity_claims(claims) == [("P1", "Q2")]


def test_truthy_keeps_all_normal_when_none_preferred():
    claims = {"P1": [_statement("Q1", "normal"), _statement("Q2", "normal")]}
    assert truthy_entity_claims(claims) == [("P1", "Q1"), ("P1", "Q2")]


def test_truthy_skips_non_entity_values():
    claims = {
        "P1": [
            {
                "rank": "normal",
                "mainsnak": {
                    "snaktype": "value",
                    "datavalue": {"type": "string", "value": "hello"},
                },
            }
        ],
        "P2": [{"rank": "normal", "mainsnak": {"snaktype": "novalue"}}],
    }
    assert truthy_entity_claims(claims) == []


def test_label_falls_back_to_mul():
    """Wikidata moved shared spellings to 'mul'; reading only 'en' loses them."""
    assert _pick_label({"mul": {"value": "Capcom"}}) == "Capcom"
    assert _pick_label({"en": {"value": "A"}, "mul": {"value": "B"}}) == "A"
    assert _pick_label({}) is None


def test_aliases_merge_en_and_mul_without_duplicates():
    merged = _merge_aliases(
        {
            "en": [{"value": "CAPCOM"}, {"value": "Capcom Inc"}],
            "mul": [{"value": "Capcom Inc"}, {"value": "Capcom Co."}],
        }
    )
    assert merged == ["CAPCOM", "Capcom Inc", "Capcom Co."]


def _statement(qid: str, rank: str):
    return {
        "rank": rank,
        "mainsnak": {
            "snaktype": "value",
            "datavalue": {"type": "wikibase-entityid", "value": {"id": qid}},
        },
    }


# --- filters ---------------------------------------------------------------


def _items(rows):
    frame = pd.DataFrame(rows)
    frame["object"] = frame["object"].apply(list)
    return frame


def test_common_filters_drop_rare_and_generic():
    rows = []
    for i in range(5):
        rows.append({"subject": f"Q{i}", "relation": "P1", "relation_label": "genre",
                     "category": "Film", "object": ["Qa"]})
    rows.append({"subject": "Q9", "relation": "P2", "relation_label": "rare",
                 "category": "Film", "object": ["Qb"]})
    rows.append({"subject": "Q8", "relation": "P3", "relation_label": "instance of",
                 "category": "Film", "object": ["Qc"]})
    out = items_mod.apply_common_filters(_items(rows), 1, 2, ["instance of"])
    assert set(out["relation_label"]) == {"genre"}


def test_cardinality_classes():
    rows = []
    for i in range(3):
        rows.append({"relation_label": "single", "multi_valued": 0})
    for i in range(4):
        rows.append({"relation_label": "multi", "multi_valued": 1})
    # Mostly single but occasionally multi: ungradeable, so neither class.
    rows += [{"relation_label": "mixed", "multi_valued": 0}] * 5
    rows += [{"relation_label": "mixed", "multi_valued": 1}]
    table = stage2_select.classify_cardinality(pd.DataFrame(rows), min_multi_valued=2)
    by = table.set_index("relation_label")
    assert by.loc["single", "single_valued"] == 1
    assert by.loc["multi", "enough_multi_valued"] == 1
    assert by.loc["mixed", "single_valued"] == 0
    assert by.loc["mixed", "enough_multi_valued"] == 0


def test_whitelist_reports_typo_with_suggestion(caplog):
    """The 'Films'/'Film' defect: one unmatched string cost a whole domain."""
    with caplog.at_level("WARNING"):
        unmatched = stage2_select.check_whitelist(["Films", "Football"], ["Film", "Football"])
    assert unmatched == ["Films"]
    assert "did you mean 'Film'" in caplog.text


def test_whitelist_silent_when_everything_matches(caplog):
    with caplog.at_level("WARNING"):
        assert stage2_select.check_whitelist(["Film"], ["Film", "TV"]) == []
    assert caplog.text == ""


# --- saliency --------------------------------------------------------------


def test_pagerank_ranks_a_relations_own_category_first():
    """A relation used only by Biology articles must score Biology highest."""
    rows = []
    for i in range(10):
        rows.append({"subject": f"B{i}", "relation": "P_bio",
                     "relation_label": "spore print colour", "category": "Biology",
                     "object": ["Qx"]})
        rows.append({"subject": f"F{i}", "relation": "P_film",
                     "relation_label": "director", "category": "Film",
                     "object": ["Qy"]})
    ranked = stage1_saliency.rank_categories(
        _items(rows),
        {"P_bio": "spore print colour", "P_film": "director"},
        progress=False,
    )
    assert ranked["Biology"][0][0] == "spore print colour"
    assert ranked["Film"][0][0] == "director"


# --- templates -------------------------------------------------------------


@pytest.mark.parametrize(
    "template,problem",
    [
        ("Who published [S]?", None),
        ("Who published it?", "[S]"),
        ("[S] and [S]?", "[S]"),
        ("Who published [S].", "question mark"),
        ("line one\nline two [S]?", "one line"),
        ("", "empty"),
    ],
)
def test_template_validation(template, problem):
    result = templates_mod.validate(template)
    if problem is None:
        assert result is None
    else:
        assert result is not None and problem in result


def test_resolve_falls_back_and_caches(tmp_path):
    path = tmp_path / "t.csv"
    templates_mod.save(path, {"publisher": ("Who published [S]?", "manual")})
    resolved = templates_mod.resolve(["publisher", "colour"], path, provider="none")
    assert resolved["publisher"] == "Who published [S]?"
    assert "[S]" in resolved["colour"]
    # The invented template is written back so a human can review it.
    assert templates_mod.load(path)["colour"][1] == "fallback"


def test_render_substitutes_subject():
    assert templates_mod.render("Who published [S]?", "Doom") == "Who published Doom?"


# --- exemplars -------------------------------------------------------------


def _exemplar_items():
    rows = []
    for i in range(10):
        rows.append(
            {
                "subject": f"Q{i}",
                "category": "Film",
                "relation_label": "director",
                "object": [f"D{i}"],
                "cardinality": 1,
                "question": f"Who directed film {i}?",
            }
        )
    return pd.DataFrame(rows)


def test_exemplar_selection_is_deterministic_under_a_seed():
    frame = _exemplar_items()
    titles = {f"D{i}": f"Director {i}" for i in range(10)}
    first = stage3_build.select_exemplars(frame, set(), {"director"}, titles, {}, 3, seed=1)
    second = stage3_build.select_exemplars(frame, set(), {"director"}, titles, {}, 3, seed=1)
    assert first == second
    other = stage3_build.select_exemplars(frame, set(), {"director"}, titles, {}, 3, seed=2)
    assert len(other[("Film", "director")]) == 3


def test_exemplars_skip_portal_pages():
    frame = _exemplar_items()
    titles = {f"D{i}": "Portal:Film" for i in range(10)}
    picked = stage3_build.select_exemplars(frame, set(), {"director"}, titles, {}, 3, seed=0)
    assert picked.get(("Film", "director"), []) == []


def test_exemplars_do_not_repeat_the_same_answer():
    frame = _exemplar_items()
    frame["object"] = [["SAME"] for _ in range(len(frame))]
    titles = {"SAME": "Same Person"}
    picked = stage3_build.select_exemplars(frame, set(), {"director"}, titles, {}, 5, seed=0)
    assert len(picked[("Film", "director")]) == 1


def test_format_exemplars_shape():
    text = stage3_build.format_exemplars(
        [{"question": "Q1?", "answer": ["A", "B"]}, {"question": "Q2?", "answer": ["C"]}]
    )
    assert text == "Q1?\nA\nB\n\nQ2?\nC\n\n"


def test_ground_truth_forms_union():
    forms = stage3_build.ground_truth_forms(
        ["Q1"], {"Q1": ["alias"]}, {"Q1": "Title"}, {"Q1": ["label"]}
    )
    assert forms["Q1"] == ["alias", "Title", "label"]


# --- linking ---------------------------------------------------------------


def _anchors(link_map):
    return link.index_anchors(link_map)


def test_link_exact_and_normalised():
    link_map = {"Capcom": "Capcom"}
    by_text, by_title, tokens = _anchors(link_map)
    t2q = {"Capcom": "Q14428"}
    assert link.resolve_mention("Capcom", link_map, by_text, by_title, tokens, t2q, True) == (
        "Capcom", "hyperlink_exact")
    assert link.resolve_mention("  capcom ", link_map, by_text, by_title, tokens, t2q, True) == (
        "Capcom", "hyperlink_normalized")


def test_link_by_piped_anchor_target():
    """The article links [[Sega Saturn|Saturn]]; the model extracted the target."""
    link_map = {"Saturn": "Sega Saturn"}
    by_text, by_title, tokens = _anchors(link_map)
    t2q = {"Sega Saturn": "Q200912"}
    assert link.resolve_mention(
        "Sega Saturn", link_map, by_text, by_title, tokens, t2q, True
    ) == ("Sega Saturn", "hyperlink_title")


def test_submention_resolves_only_when_unambiguous():
    t2q = {"Barack Obama": "Q76", "New York City": "Q60", "New York City Subway": "Q7733"}
    link_map = {"Barack Obama": "Barack Obama"}
    by_text, by_title, tokens = _anchors(link_map)
    assert link.resolve_mention(
        "Obama", link_map, by_text, by_title, tokens, t2q, True
    ) == ("Barack Obama", "hyperlink_submention")

    # Two different entities contain "New York", so it stays unlinked.
    ambiguous = {"New York City": "New York City", "New York City Subway": "New York City Subway"}
    by_text, by_title, tokens = _anchors(ambiguous)
    assert link.resolve_mention(
        "New York", ambiguous, by_text, by_title, tokens, t2q, True
    ) == (None, None)


def test_submention_can_be_disabled():
    link_map = {"Barack Obama": "Barack Obama"}
    by_text, by_title, tokens = _anchors(link_map)
    assert link.resolve_mention(
        "Obama", link_map, by_text, by_title, tokens, {"Barack Obama": "Q76"},
        False,
    ) == (None, None)


def test_unanchored_mention_is_never_linked():
    """The whole point: no global title-index fallback."""
    by_text, by_title, tokens = _anchors({})
    assert link.resolve_mention(
        "Capcom", {}, by_text, by_title, tokens, {"Capcom": "Q14428"}, True
    ) == (None, None)


def test_merge_keeps_unlinked_forms_but_not_duplicates():
    items = pd.DataFrame(
        [{"subject_title": "Doom", "relation": "publisher",
          "ground_truth": {"Q1": ["id Software"]}}]
    )
    linked = pd.DataFrame(
        [
            {"entity": "Doom", "property": "publisher", "extraction_text": "GT Interactive",
             "wikidata_qid": "Q2"},
            # Already an accepted form of Q1, so it must not become its own key.
            {"entity": "Doom", "property": "publisher", "extraction_text": "id Software",
             "wikidata_qid": None},
            {"entity": "Doom", "property": "publisher", "extraction_text": "Bethesda",
             "wikidata_qid": None},
        ]
    )
    merged = link.merge(items, linked, {"Q2": "GT Interactive"}, {"Q2": ["GTI"]})
    gt = merged["augmented_gt"].iloc[0]
    assert gt["Q2"] == ["GTI", "GT Interactive"]
    assert gt["Q1"] == ["id Software"]
    assert f"{link.TEXT_KEY_PREFIX}Bethesda" in gt
    assert f"{link.TEXT_KEY_PREFIX}id Software" not in gt
