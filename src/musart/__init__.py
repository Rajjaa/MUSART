"""Build a multi-domain LM-as-KB benchmark from a set of Wikipedia articles.

The pipeline runs in four stages, each a CLI subcommand:

    collect   article titles + categories -> Wikidata triples, labels, aliases
    saliency  triples                     -> salient relations per category
    select    salient relations           -> the item set that survives the funnel
    build     items                       -> questions, gold answers, ICL exemplars

Stages 1-3 are offline and deterministic. Stage 4 (``augment``) adds
article-grounded objects to the gold answers and needs an LLM API key.

MUSART itself is the worked example: ``configs/musart.yaml`` reproduces the
shipped dataset. See README.md for what that config does and does not reproduce.
"""

__version__ = "1.0.0"
