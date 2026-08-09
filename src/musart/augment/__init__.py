"""Stage 4 (opt-in): article-grounded gold answers.

Wikidata is incomplete. An article can state plainly that a game was published
by Capcom while Wikidata records only Nintendo, and a model answering "Capcom"
is then marked wrong for being right. This stage reads each article and adds
what it says to the gold answer set.

Three modules, run in order by ``pipeline.py``:

    fetch_articles  download plain text and HTML for the corpus
    extract         ask an LLM what the article says about each relation
    link            resolve each extracted mention to a Wikidata entity

This stage needs an LLM API key and costs money; stages 1-3 do not. MUSART's own
run was 21,662 article x relation pairs producing 75,998 extractions, projected
at roughly $26 on gemini-3.1-flash-lite.
"""
