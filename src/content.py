"""Trusted-content layer: real peer-reviewed and preprint literature.

OpenAlex (peer-reviewed) is the primary source; arXiv (preprints) tops up thin
results and is always labelled as not-yet-peer-reviewed. Every function fails
soft: on a network or parse error it logs and returns [], so a flaky API never
crashes the pipeline. Identical searches are cached briefly.
"""
import re
import time
import functools
import requests
import xml.etree.ElementTree as ET

from . import config

log = config.get_logger("content")
_HEADERS = {"User-Agent": config.USER_AGENT}


def _clean_query(q: str) -> str:
    """OpenAlex treats * and ? as wildcards, so a natural question ('...cathodes?')
    returns a 400. Reduce the query to plain terms (letters, digits, spaces, hyphens)."""
    return re.sub(r"[^\w\s-]", " ", q or "").strip()


def _abstract_from_inverted(inv) -> str:
    """OpenAlex returns abstracts as an inverted index {word: [positions]}; rebuild the text."""
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))[: config.ABSTRACT_CHARS]


def _ttl_cache(ttl: int = config.CACHE_TTL):
    """Tiny time-boxed cache so repeated searches within `ttl` seconds skip the network."""
    def deco(fn):
        cache = {}

        @functools.wraps(fn)
        def wrap(*a, **k):
            key = (a, tuple(sorted(k.items())))
            now = time.time()
            if key in cache and now - cache[key][0] < ttl:
                return cache[key][1]
            val = fn(*a, **k)
            cache[key] = (now, val)
            return val

        return wrap
    return deco


@_ttl_cache()
def search_openalex(query: str, k: int = config.OPENALEX_K) -> list:
    """Return up to k peer-reviewed works from OpenAlex. Fails soft to []."""
    params = {
        "search": _clean_query(query),
        "per-page": k,
        "select": "title,authorships,publication_year,doi,primary_location,"
                  "abstract_inverted_index,cited_by_count",
    }
    try:
        r = requests.get("https://api.openalex.org/works", params=params,
                         headers=_HEADERS, timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("OpenAlex query failed: %s", e)
        return []
    out = []
    for w in r.json().get("results", []):
        loc = (w.get("primary_location") or {}).get("source") or {}
        out.append({
            "title": w.get("title"),
            "authors": [a["author"]["display_name"] for a in w.get("authorships", [])[:3]],
            "year": w.get("publication_year"),
            "venue": loc.get("display_name"),
            "doi": w.get("doi"),
            "citations": w.get("cited_by_count"),
            "abstract": _abstract_from_inverted(w.get("abstract_inverted_index")),
            "source": "OpenAlex (peer-reviewed)",
        })
    return out


@_ttl_cache()
def search_arxiv(query: str, k: int = config.ARXIV_K) -> list:
    """Return up to k arXiv preprints. Fails soft to []."""
    params = {"search_query": "all:" + _clean_query(query), "max_results": k}
    try:
        r = requests.get("http://export.arxiv.org/api/query", params=params,
                         headers=_HEADERS, timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()
        entries = ET.fromstring(r.text).findall("a:entry", {"a": "http://www.w3.org/2005/Atom"})
    except (requests.RequestException, ET.ParseError) as e:
        log.warning("arXiv query failed: %s", e)
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in entries:
        out.append({
            "title": (e.findtext("a:title", "", ns) or "").strip(),
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)][:3],
            "year": (e.findtext("a:published", "", ns) or "")[:4],
            "venue": "arXiv (preprint)",
            "doi": e.findtext("a:id", "", ns),
            "abstract": (e.findtext("a:summary", "", ns) or "").strip()[: config.ABSTRACT_CHARS],
            "source": "arXiv (preprint)",
        })
    return out


def retrieve(query: str) -> list:
    """Peer-reviewed first; top up with preprints only when results are thin."""
    papers = search_openalex(query)
    if len(papers) < config.MIN_PEER_REVIEWED:
        papers = papers + search_arxiv(query)
    log.info("retrieved %d papers for %r", len(papers), query[:60])
    return papers
