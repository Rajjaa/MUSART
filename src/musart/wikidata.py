"""Batched, cached, resumable access to the Wikimedia APIs.

Stage 0 needs four things from Wikimedia: the QID behind each article title, the
statements on each of those entities, English labels and aliases for everything
mentioned, and labels for the properties involved. All of it comes through this
client.

Three properties matter for a pipeline someone else will run:

* **Batched.** The APIs take 50 ids per request, and using that fully is the
  difference between minutes and hours.
* **Cached.** Every batch's *extracted* result is written to disk under a key
  derived from its contents. Re-running skips the network entirely.
* **Polite.** Requests are serial, carry a descriptive User-Agent as Wikimedia's
  etiquette policy asks, and back off on 429 and 5xx rather than hammering.

Caching the extracted result rather than the raw response keeps the cache small
-- full claim JSON for a 27,000-article corpus is gigabytes; the triples it
yields are a few megabytes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

BATCH_SIZE = 50
MAX_RETRIES = 8


def batched(items: Sequence[str], size: int = BATCH_SIZE) -> Iterable[List[str]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


class WikidataClient:
    def __init__(
        self,
        cache_dir: Path,
        contact: str = "musart-pipeline",
        delay: float = 0.5,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            f"musart-pipeline/1.0 ({contact}) python-requests"
        )

    # --- plumbing ---------------------------------------------------------

    def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params = {**params, "format": "json", "formatversion": "2"}
        backoff = 2.0
        for attempt in range(1, MAX_RETRIES + 1):
            wait = None
            try:
                response = self.session.get(url, params=params, timeout=60)
                if response.status_code in (429, 500, 502, 503, 504):
                    # Wikimedia says how long to wait when it throttles; obeying
                    # that is both faster and better behaved than guessing.
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait = min(float(retry_after), 300.0)
                    raise requests.HTTPError(f"HTTP {response.status_code}")
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"{url} failed after {MAX_RETRIES} attempts: {exc}. "
                        f"If this is HTTP 429 the corpus is being fetched faster "
                        f"than Wikimedia allows -- raise the client delay "
                        f"(WikidataClient(delay=...)). Everything fetched so far "
                        f"is cached, so re-running resumes rather than restarting."
                    ) from exc
                if wait is None:
                    wait = backoff
                    backoff = min(backoff * 2, 120)
                log.warning(
                    "%s (attempt %d/%d), retrying in %.0fs",
                    exc,
                    attempt,
                    MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue

            if "error" in payload:
                raise RuntimeError(f"{url}: {payload['error']}")
            time.sleep(self.delay)
            return payload
        raise AssertionError("unreachable")

    def _cached_batches(
        self,
        name: str,
        keys: Sequence[str],
        fetch: Callable[[List[str]], Dict[str, Any]],
        progress: bool = True,
    ) -> Dict[str, Any]:
        """Run ``fetch`` over batches of ``keys``, reusing anything on disk.

        Cache files are named by a hash of the batch contents, so changing the
        corpus only re-fetches the batches that actually changed.
        """
        out: Dict[str, Any] = {}
        batches = list(batched(sorted(set(keys))))
        hits = 0
        for batch in tqdm(batches, desc=name, disable=not progress):
            digest = hashlib.sha256("\n".join(batch).encode("utf-8")).hexdigest()[:24]
            path = self.cache_dir / f"{name}_{digest}.json"
            if path.exists():
                try:
                    out.update(json.loads(path.read_text(encoding="utf-8")))
                    hits += 1
                    continue
                except json.JSONDecodeError:
                    log.warning("corrupt cache file %s, refetching", path)
            result = fetch(batch)
            # Written whole, then moved, so an interrupted run never leaves a
            # half-written cache entry that would be trusted next time.
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(result), encoding="utf-8")
            tmp.replace(path)
            out.update(result)
        if hits:
            log.info("%s: %d/%d batches from cache", name, hits, len(batches))
        return out

    # --- titles -> QIDs ---------------------------------------------------

    def titles_to_qids(self, titles: Sequence[str], progress: bool = True) -> Dict[str, str]:
        """Map article titles to QIDs.

        The API normalises and follows redirects, and reports both, so the
        result is keyed by the title that was asked for rather than the one
        Wikipedia settled on.
        """

        def fetch(batch: List[str]) -> Dict[str, str]:
            payload = self._get(
                WIKIPEDIA_API,
                {
                    "action": "query",
                    "prop": "pageprops",
                    "ppprop": "wikibase_item",
                    "redirects": "1",
                    "titles": "|".join(batch),
                },
            )
            query = payload.get("query", {})
            # Chase requested title -> normalised -> redirect target.
            alias: Dict[str, str] = {}
            for entry in query.get("normalized", []):
                alias[entry["from"]] = entry["to"]
            for entry in query.get("redirects", []):
                alias[entry["from"]] = entry["to"]

            resolved: Dict[str, str] = {}
            for page in query.get("pages", []):
                qid = (page.get("pageprops") or {}).get("wikibase_item")
                if qid:
                    resolved[page.get("title")] = qid

            out: Dict[str, str] = {}
            for title in batch:
                seen = set()
                current = title
                while current in alias and current not in seen:
                    seen.add(current)
                    current = alias[current]
                if current in resolved:
                    out[title] = resolved[current]
            return out

        return self._cached_batches("title2qid", titles, fetch, progress)

    # --- entities ---------------------------------------------------------

    def _entities(self, batch: List[str], props: str) -> Dict[str, Any]:
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": props,
        }
        if "labels" in props or "aliases" in props:
            # Wikidata has migrated labels that are spelled the same in every
            # Latin-script language to the 'mul' pseudo-language. Asking for
            # 'en' alone returns nothing for those entities -- Q42 comes back
            # with 75 languages and no English label at all. Both codes are
            # requested, and languagefallback resolves 'mul' into 'en'.
            params["languages"] = "en|mul"
            params["languagefallback"] = "1"
        payload = self._get(WIKIDATA_API, params)
        return payload.get("entities", {}) or {}

    def claims(
        self, qids: Sequence[str], progress: bool = True
    ) -> Dict[str, List[Tuple[str, str]]]:
        """QID -> [(property id, object QID), ...], truthy statements only."""

        def fetch(batch: List[str]) -> Dict[str, List[Tuple[str, str]]]:
            entities = self._entities(batch, "claims")
            return {
                qid: truthy_entity_claims(entity.get("claims") or {})
                for qid, entity in entities.items()
                if "missing" not in entity
            }

        return self._cached_batches("claims", qids, fetch, progress)

    def labels_and_aliases(
        self, qids: Sequence[str], progress: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """QID -> {"label": str | None, "aliases": [str, ...]}."""

        def fetch(batch: List[str]) -> Dict[str, Dict[str, Any]]:
            entities = self._entities(batch, "labels|aliases")
            out: Dict[str, Dict[str, Any]] = {}
            for qid, entity in entities.items():
                if "missing" in entity:
                    continue
                out[qid] = {
                    "label": _pick_label(entity.get("labels") or {}),
                    "aliases": _merge_aliases(entity.get("aliases") or {}),
                }
            return out

        return self._cached_batches("labels", qids, fetch, progress)


def _pick_label(labels: Dict[str, Any]) -> Optional[str]:
    """English label, falling back to the multilingual one."""
    for code in ("en", "mul"):
        value = (labels.get(code) or {}).get("value")
        if value:
            return value
    return None


def _merge_aliases(aliases: Dict[str, Any]) -> List[str]:
    """Union of the English and multilingual alias lists.

    Neither is a superset of the other: "Douglas Noel Adams" is only under
    'mul', while Capcom's "Capcom Entertainment Inc" is only under 'en'.
    Reading one and not the other loses gold answers and turns correct model
    responses into false negatives.
    """
    merged: List[str] = []
    for code in ("en", "mul"):
        for entry in aliases.get(code) or []:
            value = entry.get("value")
            if value and value not in merged:
                merged.append(value)
    return merged


def truthy_entity_claims(claims: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract (property, object QID) pairs, applying Wikidata's truthy rule.

    Truthy means: within a property, if any statement is ranked *preferred*,
    only those count; otherwise the *normal*-ranked ones do. *Deprecated* never
    counts. This is what a knowledge base would answer with, as opposed to every
    historical or disputed value ever recorded.

    Only object-valued statements survive -- dates, quantities and strings are
    not entities and have no aliases to grade an answer against.
    """
    out: List[Tuple[str, str]] = []
    for pid, statements in claims.items():
        ranks = {s.get("rank") for s in statements}
        wanted = "preferred" if "preferred" in ranks else "normal"
        for statement in statements:
            if statement.get("rank") != wanted:
                continue
            snak = statement.get("mainsnak") or {}
            if snak.get("snaktype") != "value":
                continue
            datavalue = snak.get("datavalue") or {}
            if datavalue.get("type") != "wikibase-entityid":
                continue
            object_id = (datavalue.get("value") or {}).get("id")
            if object_id:
                out.append((pid, object_id))
    return out
