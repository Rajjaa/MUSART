"""Download article plain text and HTML for the corpus.

Two representations are needed and they are not interchangeable. Extraction
reads **plain text**, because that is what the LLM should see and what
langextract aligns character spans against. Linking reads **HTML**, because the
whole point of the linking step is that the article's own hyperlinks decide
which entity a mention refers to, and stripping the markup throws exactly that
away.

Both are written once per article and skipped on re-run, so an interrupted fetch
resumes.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

from .. import corpus as corpus_mod
from ..config import Config

log = logging.getLogger(__name__)

REST_HTML = "https://en.wikipedia.org/api/rest_v1/page/html/{title}"
EXTRACT_API = "https://en.wikipedia.org/w/api.php"


def _safe_name(title: str) -> str:
    """Match the linker's expectation: only '/' is sanitised out of titles."""
    return title.replace("/", "_")


def fetch_html(session: requests.Session, title: str) -> Optional[str]:
    url = REST_HTML.format(title=requests.utils.quote(title.replace(" ", "_"), safe=""))
    response = session.get(url, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text


def fetch_text(session: requests.Session, titles: List[str]) -> Dict[str, str]:
    """Plain-text extracts, up to 20 titles per request (the API's cap here)."""
    response = session.get(
        EXTRACT_API,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": "|".join(titles),
            "format": "json",
            "formatversion": "2",
        },
        timeout=60,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])
    return {
        page["title"]: page["extract"]
        for page in pages
        if page.get("extract")
    }


def run(cfg: Config, limit: Optional[int] = None) -> None:
    frame = (
        corpus_mod.read(cfg.corpus_file)
        if cfg.corpus_file.exists()
        else corpus_mod.load(cfg.corpus)
    )
    if limit:
        frame = frame.head(limit)

    html_dir = cfg.work_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    text_path = cfg.work_dir / "articles.jsonl"

    session = requests.Session()
    session.headers["User-Agent"] = f"musart-pipeline/1.0 ({cfg.contact})"

    titles = frame["title"].tolist()

    # --- plain text ---
    have: Dict[str, str] = {}
    if text_path.exists():
        with text_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                have[record["title"]] = record["content"]
        log.info("%d article texts already downloaded", len(have))

    todo = [t for t in titles if t not in have]
    if todo:
        with text_path.open("a", encoding="utf-8") as fh:
            for start in tqdm(range(0, len(todo), 20), desc="text"):
                batch = todo[start : start + 20]
                try:
                    for title, content in fetch_text(session, batch).items():
                        fh.write(
                            json.dumps(
                                {"title": title, "content": content},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    fh.flush()
                except requests.RequestException as exc:
                    log.warning("text batch failed (%s), continuing", exc)
                time.sleep(cfg.api_delay)

    # --- HTML ---
    missing = [t for t in titles if not (html_dir / f"{_safe_name(t)}.html").exists()]
    log.info("%d of %d articles need HTML", len(missing), len(titles))
    failures = 0
    for title in tqdm(missing, desc="html"):
        try:
            html = fetch_html(session, title)
        except requests.RequestException as exc:
            log.warning("HTML fetch failed for %r: %s", title, exc)
            failures += 1
            continue
        if html is None:
            failures += 1
            continue
        # Written whole then moved, so an interrupted run never leaves a
        # truncated file that the linker would silently treat as link-free.
        path = html_dir / f"{_safe_name(title)}.html"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(html, encoding="utf-8")
        tmp.replace(path)
        time.sleep(cfg.api_delay)

    if failures:
        log.warning(
            "%d article(s) had no HTML; their mentions can only be kept as "
            "surface forms, never linked to an entity",
            failures,
        )
    log.info("text -> %s, html -> %s", text_path, html_dir)
