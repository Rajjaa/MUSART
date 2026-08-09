"""Pipeline configuration.

One YAML file drives every stage. ``configs/musart.yaml`` is the preset that
reproduces the shipped MUSART dataset; ``configs/template.yaml`` is the starting
point for a new corpus.

Every filter threshold in the pipeline is a field here. Nothing about MUSART's
particular categories, relations or thresholds is hardcoded in the stage code.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"


@dataclass
class SaliencyConfig:
    """Stage 1: which relations are salient for which category."""

    # Categories with fewer items than this are dropped before the graph is
    # built -- a category needs enough mass for PageRank to say anything.
    min_category_items: int = 100
    # Same idea for relations.
    min_relation_items: int = 50
    # Relation labels excluded outright (Wikimedia housekeeping, taxonomy links,
    # demographics). Path to a newline-delimited file.
    generic_relations: Path = ASSETS / "generic_relations.txt"
    # How many nodes of each relation's personalized-PageRank ranking to inspect
    # for categories.
    top_k: int = 30
    # PageRank parameters. A high damping factor keeps the walk close to its
    # seed relation, which is what makes the scores discriminative between
    # categories rather than reflecting overall graph structure.
    #
    # 'piteration' runs a fixed number of power iterations; 'lanczos' solves
    # for the exact stationary distribution and is independent of n_iter.
    n_iter: int = 50
    damping_factor: float = 0.999
    solver: str = "piteration"
    # A relation is salient for a category when the category scores above this
    # in the relation's personalized PageRank. Raising it yields fewer, more
    # domain-specific relations; lowering it broadens the benchmark.
    threshold: float = 0.1


@dataclass
class SelectConfig:
    """Stage 2: the funnel from salient relations down to the final item set."""

    # Categories to keep. None keeps every category that survives the filters.
    #
    # A whitelist is an editorial choice rather than a data-driven one. Entries
    # that match nothing are reported with a close-match suggestion rather than
    # ignored, so a misspelling does not quietly remove a domain.
    categories: Optional[List[str]] = None
    # After the salient join and the category whitelist, relations still need
    # this many items to be worth asking about.
    min_relation_items_post: int = 105
    # A relation is kept when it is either reliably single-valued (no item has
    # more than one object) or has at least this many genuinely multi-valued
    # items. Relations in between are ambiguous to grade and are dropped.
    min_multi_valued: int = 100
    # (category, relation) pairs to drop, for pairs whose object range is so
    # small the question is trivial. Inspect cat_relation_range.csv to choose.
    exclude_pairs: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class BuildConfig:
    """Stage 3: questions, gold answers, in-context exemplars."""

    # relation -> question template CSV. Templates use [S] for the subject.
    templates: Path = ASSETS / "musart_templates.csv"
    # How to produce templates for relations the CSV does not cover.
    #   gemini | anthropic  ask an LLM, then cache to the CSV for review
    #   none                use the generic fallback wording
    #   auto                gemini or anthropic if a key is present, else none
    template_llm: str = "auto"
    template_model: Optional[str] = None
    # Exemplars per (category, relation) in the few-shot prompt.
    n_icl: int = 5
    # Exemplar selection shuffles the candidate items; this seed makes a build
    # reproducible.
    seed: int = 0


@dataclass
class AugmentConfig:
    """Stage 4 (opt-in): article-grounded gold answers."""

    model_id: str = "gemini-3.1-flash-lite"
    # langextract chunk size. The default (1000) splits an article into ~16
    # chunks and re-sends the prompt with each, inflating cost ~3.4x. A value
    # above the longest article means one call per article and keeps span
    # grounding intact.
    max_char_buffer: int = 200000
    # Keep only extractions langextract could ground to a character span in the
    # source text. Tier-2 (ungrounded) extractions are the ones that drift.
    tier1_only: bool = True
    # Resolve a short repeat mention ("Capcom") to an entity linked by its full
    # form earlier in the same article.
    submention_pass: bool = True
    max_workers: int = 8


@dataclass
class Config:
    """The whole pipeline."""

    # Corpus spec: nested {category: {subcategory: [titles]}} JSON, or a CSV
    # with title,category columns.
    corpus: Optional[Path] = None
    # Where every stage reads and writes.
    work_dir: Path = Path("output")
    # Skip stage 0 and use these instead of fetching from the Wikimedia APIs.
    # Each is optional and independent.
    #   triples     tab-separated, no header:
    #               subject, relation, object, subject_label, object_label
    #   title2qid   CSV with qid,title columns (the Wikipedia<->Wikidata index)
    #   aliases     JSONL with qid,alias columns, alias being a list
    #   pid2label   JSON mapping property id -> English label
    triples: Optional[Path] = None
    title2qid: Optional[Path] = None
    aliases: Optional[Path] = None
    pid2label: Optional[Path] = None
    # Contact address sent in the User-Agent on Wikimedia API calls, as their
    # etiquette policy asks. Set this to a real address before running a large
    # corpus -- Wikimedia may throttle unidentified clients harder.
    contact: str = "musart-pipeline"
    # Seconds between Wikimedia API calls. Raise it if stage 0 hits HTTP 429.
    api_delay: float = 0.5

    saliency: SaliencyConfig = field(default_factory=SaliencyConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    build: BuildConfig = field(default_factory=BuildConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    # ---- derived paths -------------------------------------------------

    @property
    def corpus_file(self) -> Path:
        return self.work_dir / "corpus.jsonl"

    @property
    def triples_file(self) -> Path:
        return self.triples or (self.work_dir / "triples.csv")

    @property
    def pid2label_file(self) -> Path:
        return self.pid2label or (self.work_dir / "pid2label.json")

    @property
    def qid2title_file(self) -> Path:
        return self.work_dir / "qid2title.json"

    @property
    def qid2label_file(self) -> Path:
        return self.work_dir / "qid2label.json"

    @property
    def aliases_file(self) -> Path:
        return self.aliases or (self.work_dir / "qid2aliases.jsonl")

    @property
    def cat2relations_file(self) -> Path:
        return self.work_dir / "cat2relations.jsonl"

    @property
    def selected_file(self) -> Path:
        return self.work_dir / "selected_items.jsonl"

    @property
    def items_file(self) -> Path:
        return self.work_dir / "items.jsonl"

    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache"


_SECTIONS = {
    "saliency": SaliencyConfig,
    "select": SelectConfig,
    "build": BuildConfig,
    "augment": AugmentConfig,
}

# Fields that are paths, and so are resolved relative to the config file.
_PATH_FIELDS = {
    "corpus",
    "work_dir",
    "triples",
    "title2qid",
    "aliases",
    "pid2label",
    "generic_relations",
    "templates",
}


def _coerce(cls, data: Dict[str, Any], base: Path):
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"unknown key(s) in {cls.__name__}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )
    kwargs = {}
    for name, value in data.items():
        if value is not None and name in _PATH_FIELDS:
            path = Path(value)
            value = path if path.is_absolute() else (base / path)
        kwargs[name] = value
    return cls(**kwargs)


def load(path: Path) -> Config:
    """Read a YAML config. Relative paths resolve against the config's folder."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent

    sections = {}
    for name, cls in _SECTIONS.items():
        sections[name] = _coerce(cls, raw.pop(name, None) or {}, base)

    cfg = _coerce(Config, raw, base)
    for name, value in sections.items():
        setattr(cfg, name, value)

    # exclude_pairs comes out of YAML as lists; make them hashable tuples.
    cfg.select.exclude_pairs = [tuple(p) for p in cfg.select.exclude_pairs]
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.saliency.threshold < 0:
        raise ValueError("saliency.threshold must be >= 0")
    if not 0 < cfg.saliency.damping_factor < 1:
        raise ValueError("saliency.damping_factor must be in (0, 1)")
    if cfg.saliency.top_k < 1:
        raise ValueError("saliency.top_k must be >= 1")
    if cfg.build.n_icl < 0:
        raise ValueError("build.n_icl must be >= 0")
    if cfg.build.template_llm not in {"auto", "none", "gemini", "anthropic"}:
        raise ValueError(
            f"build.template_llm must be auto|none|gemini|anthropic, "
            f"got {cfg.build.template_llm!r}"
        )
    for pair in cfg.select.exclude_pairs:
        if len(pair) != 2:
            raise ValueError(
                f"select.exclude_pairs entries must be [category, relation], got {pair!r}"
            )
