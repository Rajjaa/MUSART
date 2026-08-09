# MUSART

Build a multi-domain LM-as-KB benchmark from a set of Wikipedia articles.

Given article titles and the domains they belong to, this produces a question
set with alias-aware gold answers and few-shot exemplars: *"Who is the publisher
of Mega Man X2?"* graded against every surface form Wikidata knows for Capcom.

MUSART itself — 21,775 questions over 11 domains and 37 relations — is one
config file here. Point the pipeline at your own articles and it builds a
benchmark for your domains instead.

---

## Install

```bash
pip install -e .                        # stages 0-3
pip install -e '.[augment,templates]'   # + stage 4 and template generation
```

Python 3.9+. Stages 0–3 need no API key.

## Quickstart on your own corpus

```bash
cp configs/template.yaml my.yaml            # point `corpus` at your articles
musart --config my.yaml collect             # titles -> Wikidata triples, labels, aliases
musart --config my.yaml run                 # saliency -> selection -> questions
```

Your corpus is either a nested JSON of `{heading: {domain: [titles]}}` — the
shape `assets/good_articles.json` has — or a CSV with `title,category` columns.
That is the only input. `collect` resolves the titles to Wikidata items and
fetches everything else from the Wikimedia APIs, caching every batch so an
interrupted run resumes.

The defaults are tuned for ~25,000 articles. For a smaller corpus, lower
`min_category_items`, `min_relation_items` and `min_relation_items_post`, or the
filters will empty it.

## The four stages

| stage | command | what it decides |
|---|---|---|
| 0 | `collect` | which entities and statements the corpus contains |
| 1 | `saliency` | which relations are *characteristic* of each domain |
| 2 | `select` | which of those are frequent and gradeable enough to ask |
| 3 | `build` | how to phrase each question and what counts as a right answer |
| 4 | `augment` | what the *article* says, beyond what Wikidata records |

### 1. Saliency

Raw frequency picks the wrong relations: `country` and `instance of` are
everywhere and characterise nothing. Instead, personalized PageRank runs over a
graph that alternates `relation -> subject -> category`, seeded at one relation
at a time. Categories are sinks, so they accumulate the walk's mass, and a
relation used only by mushroom articles sends nearly all of its mass to Biology
while a relation used everywhere spreads thin and clears no bar. Relations
scoring above `threshold` for a domain are that domain's salient set.

### 2. Selection

Salient is necessary, not sufficient. A relation also has to be frequent enough
to be worth asking, and predictable enough in arity to grade: a relation that is
usually single-valued but occasionally multi-valued gives a model no way to know
how many answers are wanted and penalises it either way. Those are dropped.

`select` writes `funnel.csv`, which reports what each filter cost — that is the
table to quote when describing the construction — and `cat_relation_range.csv`,
which shows how many distinct objects each (domain, relation) pair takes. A pair
with three possible answers is a guessing game, not a knowledge probe; that file
is how you find them.

### 3. Build

Gold answers collect every alias, label and page title for each object, so
"Capcom" is not marked wrong against "Capcom Co., Ltd.". Questions come from a
per-relation template CSV; relations with no template are sent to an LLM once,
validated, and appended with `source=llm` for review. Exemplars are drawn per
(domain, relation), skipping items whose objects have all been used already so
five examples are not five ways of saying "Nintendo".

### 4. Augment (opt-in, costs money)

Wikidata is incomplete: an article can state plainly that a game was published
by Capcom while Wikidata records only Nintendo, and a model answering "Capcom"
is marked wrong for being right. This stage reads each article and adds what it
says.

```bash
musart --config my.yaml fetch-articles
export LANGEXTRACT_API_KEY=...
musart --config my.yaml augment
```

Mentions are linked to entities **only via the article's own hyperlinks**.
Matching surface forms against a global title index instead would attach
entities the article never points at — "Mercury" in a chemistry article
resolving to the planet — and those false links are worse than no link, because
they enter the gold set and reward wrong answers. Mentions that stay unlinked
are **kept** as `TEXT:`-keyed surface forms rather than discarded, so a model
answering in the article's own words is credited.

MUSART's run was 21,662 article×relation pairs, 75,998 extractions, projected at
roughly $26 on `gemini-3.1-flash-lite`. Each pair is written the moment it
succeeds, so the run is resumable by restarting it.

---

## Data

The benchmark and everything needed to rebuild it are in a single 15 MB
download:

**[musart-data.tar.gz](https://osf.io/t843u/overview?view_only=811a08a286c34528a674e8cd725770ca)**
— open the link, then download the file from *Files*.

```
sha256  0cf8e97af7bac82f520d8ba00e6a3d18f9dd530b93ce7fbcc0df38e722de7b84
```

```bash
tar -xzf musart-data.tar.gz     # from the root of this repository
```

That creates `inputs/`, which is where `configs/musart.yaml` and the
`repro-check` command already point — no path editing.

| file | what it is |
|---|---|
| `inputs/musart_published.jsonl` | the benchmark: 21,775 items, 11 domains, 37 relations, 8,407 subjects |
| `inputs/musart_published_augmented.jsonl` | the same items with article-grounded objects added to the gold answers |
| `inputs/ga_triples_with_labels.csv` | Wikidata statements over the corpus subjects |
| `inputs/enwiki_qid_title.csv` | Wikipedia title ↔ QID index |
| `inputs/wikidata_aliases.jsonl` | English aliases per QID |
| `inputs/pid2label.json` | property id → English label |

The last four are a May 2025 snapshot, included because Wikidata is live:
`musart collect` fetches current data and builds a valid benchmark, but not this
one. The title index and alias file are restricted to entities appearing in the
corpus triples — the full versions are ~860 MB between them and are ~99%
irrelevant here. The restriction is by membership in the triples file rather
than in the final dataset, so every intermediate stage sees what it would
otherwise have seen.

Wikidata content is CC0; Wikipedia titles and text are CC BY-SA 4.0.

## Reproducing MUSART

With `inputs/` in place:

```bash
musart --config configs/musart.yaml run
musart --config configs/musart.yaml repro-check --against inputs/musart_published.jsonl
```

This reproduces the funnel exactly:

| stage | items | | relations | domains |
|---|---:|---|---:|---:|
| input | 153,290 | | 286 | 50 |
| salient relations | **108,443** | | 286 | 41 |
| domain whitelist | **60,770** | | 208 | 12 |
| relation frequency ≥ 105 | **55,753** | | 105 | 12 |
| cardinality | **22,775** | | 38 | 11 |
| unresolvable objects | **22,756** | | 38 | 11 |
| excluded pairs | **21,775** | | **37** | **11** |

8,407 subjects, 44,722 triples. `repro-check` confirms `object`, `question`,
`subject_title` and `ground_truth` are identical across all 21,775 items, and
the regenerated `cat2relations.jsonl` matches the published one byte for byte —
all 50 domains, every PageRank score to 4dp.

`icl_examples` is compared separately and reported rather than asserted:
exemplar selection is randomised, so it matches only when the same `build.seed`
and the same candidate pool are used.

---

## Layout

```
configs/musart.yaml      reproduces the published dataset
configs/template.yaml    commented starting point for a new corpus
assets/                  GA hierarchy, 121 question templates, relation blocklist
src/musart/              stage0_collect, stage1_saliency, stage2_select,
                         stage3_build, templates, wikidata, augment/
tests/                   33 unit tests: pytest tests
```

Outputs land in `work_dir` and are not tracked by git.

## Citation

If you use MUSART or this pipeline, please cite the accompanying paper.
