#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import re
import sys
import time
import gzip
import tarfile
import io
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Iterable, Optional

from html import unescape as html_unescape
from html.parser import HTMLParser


DOCS_LLMS_TXT = "https://docs.polymarket.com/llms.txt"
POLYMARKET_HOME = "https://polymarket.com"
CLOB_ORIGIN = "https://clob.polymarket.com"
GAMMA_ORIGIN = "https://gamma-api.polymarket.com"
NPM_CLOB_CLIENT_REGISTRY = "https://registry.npmjs.org/@polymarket/clob-client"
PYPI_CLOB_CLIENT_JSON = "https://pypi.org/pypi/py-clob-client/json"

# Default "useful" API hosts (exclude polymarket.com webapp internals).
DEFAULT_PUBLIC_HOSTS = {
    "gamma-api.polymarket.com",
    "clob.polymarket.com",
    "bridge.polymarket.com",
    "data-api.polymarket.com",
    "relayer-v2.polymarket.com",
    "user-pnl-api.polymarket.com",
    "builders.polymarket.com",
    "status.polymarket.com",
    "recovery.polymarket.com",
    "matic-recovery.polymarket.com",
    "worldcoin.polymarket.com",
}

DEFAULT_EXCLUDED_HOSTS = {
    "docs.polymarket.com",
    "help.polymarket.com",
    "images.polymarket.com",
    "news.polymarket.com",
    "learn.polymarket.com",
    "embed.polymarket.com",
}


@dataclasses.dataclass(frozen=True)
class Evidence:
    source_url: str
    context: str


@dataclasses.dataclass(frozen=True)
class Endpoint:
    raw: str  # may be absolute or relative
    normalized: str  # normalized absolute or relative
    host: str  # empty if unknown (relative-only)
    path: str  # empty if unknown
    method: str  # "GET"/"POST"/... or ""
    description: str  # short human description (may be empty)
    source_kind: str  # "docs" | "js" | "seed"
    evidence: Evidence


def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_text(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _write_text(p: Path, text: str) -> None:
    _safe_mkdir(p.parent)
    p.write_text(text, encoding="utf-8")


def _write_json(p: Path, obj) -> None:
    _safe_mkdir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_urlish(raw: str) -> str:
    s = raw.strip()

    # Trim wrapping quotes/backticks
    s = s.strip("\"'`")

    # Trim trailing punctuation that often sticks to URLs in markdown/text
    s = s.rstrip(").,;:]}>")

    # Unescape common JS escapes
    s = s.replace("\\u002F", "/").replace("\\/", "/")
    s = s.replace("&amp;", "&")
    # Normalize JS template literals like "...${x}..." into a stable placeholder.
    s = re.sub(r"\$\{[^}]+\}", "{var}", s)
    # If extraction grabbed extra code, cut at common delimiters.
    for delim in (",", "{", "}", "$", " ", "\n", "\t"):
        if delim in s:
            s = s.split(delim, 1)[0]
    s = s.rstrip(").,;:]}>")
    return s


def _parse_host_path(url_or_path: str) -> tuple[str, str]:
    """
    Returns (host, path). host == "" for relative endpoints.
    """
    s = url_or_path
    if s.startswith("http://") or s.startswith("https://"):
        u = urllib.parse.urlparse(s)
        return (u.netloc, u.path or "/")
    if s.startswith("/"):
        return ("", s)
    return ("", "")


def _source_origin(source_url: str) -> str:
    try:
        u = urllib.parse.urlparse(source_url)
    except ValueError:
        return ""
    if u.scheme and u.netloc:
        return f"{u.scheme}://{u.netloc}"
    return ""

def _source_host(source_url: str) -> str:
    try:
        return (urllib.parse.urlparse(source_url).netloc or "").lower()
    except ValueError:
        return ""


def _defrag(s: str) -> str:
    try:
        base, _frag = urllib.parse.urldefrag(s)
        return base
    except ValueError:
        return s


def _is_probably_domainish_first_segment(path: str) -> bool:
    """
    Filters out strings like "/api.etherscan.io/api" which are not real relative endpoints.
    """
    if not path.startswith("/"):
        return False
    seg = path.split("/", 2)[1]
    if seg in (".well-known",):
        return False
    return "." in seg


def _pick_base_for_relative_path(rel_path: str, source_url: str, text: str) -> str:
    """
    Decide which origin to attach a relative path to.

    - For JS bundles and site pages: join to the page origin.
    - For docs pages: prefer joining to clob/gamma if that host is referenced on the page.
      (Docs often show endpoints as "GET /book" rather than full URLs.)
    """
    host = _source_host(source_url)
    origin = _source_origin(source_url)

    if host != "docs.polymarket.com":
        return origin

    t = (text or "").lower()
    clob_hint = ("clob.polymarket.com" in t) or (CLOB_ORIGIN in t)
    gamma_hint = ("gamma-api.polymarket.com" in t) or (GAMMA_ORIGIN in t)

    clobish = bool(
        re.match(
            r"^/(book|order|orders|trades?|fills?|balances?|positions?|tickers?|prices?|quote|quotes|cancel|cancellations|auth|apikey|api-key|keys?)\b",
            rel_path,
            flags=re.I,
        )
    )
    gammaish = bool(re.match(r"^/(events?|markets?|comments?|tags?|series|search|price|prices)\b", rel_path, flags=re.I))

    if clob_hint and gamma_hint:
        if clobish and not gammaish:
            return CLOB_ORIGIN
        if gammaish and not clobish:
            return GAMMA_ORIGIN
        # ambiguous: keep relative (don't guess)
        return ""

    if clob_hint:
        return CLOB_ORIGIN
    if gamma_hint:
        return GAMMA_ORIGIN
    return ""


def _looks_like_endpoint(s: str) -> bool:
    if not s:
        return False
    if s.startswith("http://") or s.startswith("https://"):
        return True
    if s.startswith("/"):
        # Avoid lots of false positives: require a slash-path that looks API-ish
        # (still allow /markets, /events, /orders, /api, /v1, etc.)
        return bool(re.match(r"^/(api|v\d+|markets?|events?|orders?|trades?|positions?|books?|orderbook|auth|user|users|portfolio)\b", s))
    return False


def _extract_full_urls(text: str) -> list[str]:
    # Very permissive then we filter.
    candidates = re.findall(r"https?://[^\s\"'<>\\),{}:]+", text)
    return candidates


def _extract_schemeless_polymarket_urls(text: str) -> list[str]:
    # e.g. "gamma-api.polymarket.com/events" (without https://)
    candidates = re.findall(r"(?<!://)\b((?:[A-Za-z0-9-]+\.)*polymarket\.com/[^\s\"'<>\\),{}]+)", text)
    out: list[str] = []
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        out.append("https://" + c)
    return out


def _extract_relative_paths(text: str) -> list[str]:
    # Capture "/something" tokens, but not things like "/_next/static/..."
    candidates = re.findall(
        r"(?<![A-Za-z0-9_])/(?:api|v\d+|markets?|events?|orders?|order|trades?|positions?|books?|book|orderbook|auth|user|users|portfolio|comments?)"
        r"[^\s\"'<>\\),{}:]*",
        text,
    )
    # Drop obvious static assets
    out: list[str] = []
    for c in candidates:
        if c.startswith("/_next/") or c.startswith("/static/") or c.startswith("/assets/"):
            continue
        if _is_probably_domainish_first_segment(c):
            continue
        out.append(c)
    return out


def _infer_method_near(text: str, needle: str) -> str:
    """
    Try to infer HTTP method from nearby JS/JSON/markdown snippet.
    """
    i = text.find(needle)
    if i < 0:
        return ""
    window = text[max(0, i - 200) : i + 200]
    m = re.search(r"\bmethod\s*:\s*['\"](GET|POST|PUT|PATCH|DELETE)['\"]", window, flags=re.I)
    if m:
        return m.group(1).upper()
    # common fetch shorthand: axios.post("..."), client.get("...")
    m2 = re.search(r"\b(get|post|put|patch|delete)\s*\(\s*['\"]" + re.escape(needle) + r"['\"]", window, flags=re.I)
    if m2:
        return m2.group(1).upper()
    # docs / curl-style: "POST /order" or "GET /book"
    m3 = re.search(r"\b(GET|POST|PUT|PATCH|DELETE)\b\s+`?" + re.escape(needle) + r"`?\b", window, flags=re.I)
    if m3:
        return m3.group(1).upper()
    return ""


def _parse_llms_index(llms_text: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    """
    Parse https://docs.polymarket.com/llms.txt (markdown).

    Returns:
      - docs_urls: list of https://docs.polymarket.com/... URLs
      - meta_by_url: { url: { "title": str, "summary": str } }
    """
    docs_urls: list[str] = []
    meta_by_url: dict[str, dict[str, str]] = {}

    md_re = re.compile(r"^\s*-\s*\[(?P<title>[^\]]+)\]\((?P<url>https?://[^)]+)\)\s*(?::\s*(?P<summary>.*))?$")

    for line in llms_text.splitlines():
        m = md_re.match(line.strip())
        if not m:
            continue
        url = _defrag(_normalize_urlish(m.group("url")))
        if url.startswith("http://docs.polymarket.com/"):
            url = url.replace("http://", "https://", 1)
        if not url.startswith("https://docs.polymarket.com/"):
            continue

        title = (m.group("title") or "").strip()
        summary = (m.group("summary") or "").strip()

        docs_urls.append(url)
        meta_by_url.setdefault(url, {"title": "", "summary": ""})
        if title and not meta_by_url[url]["title"]:
            meta_by_url[url]["title"] = title
        if summary and not meta_by_url[url]["summary"]:
            meta_by_url[url]["summary"] = summary

    # Fallback: extract any docs urls even if line format changes.
    for u in _extract_full_urls(llms_text) + _extract_schemeless_polymarket_urls(llms_text):
        u = _defrag(_normalize_urlish(u))
        if u.startswith("http://docs.polymarket.com/"):
            u = u.replace("http://", "https://", 1)
        if u.startswith("https://docs.polymarket.com/"):
            docs_urls.append(u)

    seen = set()
    out: list[str] = []
    for u in docs_urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out, meta_by_url


class Fetcher:
    def __init__(self, timeout_s: int = 20, max_retries: int = 3, user_agent: str = "endpoint-catalog-bot/1.0"):
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.user_agent = user_agent

    def get(self, url: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    },
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read()
                    if (resp.headers.get("Content-Encoding") or "").lower() == "gzip":
                        raw = gzip.decompress(raw)
                    enc = (resp.headers.get_content_charset() or "utf-8").strip()
                    # Some servers return "utf-8" without declaring; be tolerant.
                    return raw.decode(enc, errors="replace")
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(0.6 * attempt)
        raise RuntimeError(f"failed to fetch {url}: {last_err}")

    def get_bytes(self, url: str) -> bytes:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "*/*",
                    },
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read()
                    if (resp.headers.get("Content-Encoding") or "").lower() == "gzip":
                        raw = gzip.decompress(raw)
                    return raw
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(0.6 * attempt)
        raise RuntimeError(f"failed to fetch bytes {url}: {last_err}")

class _TextAndScriptsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.script_srcs: list[str] = []
        self._skip_text_depth = 0  # inside <script>/<style>

    def handle_starttag(self, tag: str, attrs) -> None:
        t = tag.lower()
        if t in ("script", "style"):
            self._skip_text_depth += 1
        if t == "script":
            for k, v in attrs:
                if k.lower() == "src" and v:
                    self.script_srcs.append(v)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("script", "style") and self._skip_text_depth > 0:
            self._skip_text_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_text_depth > 0:
            return
        if data and data.strip():
            self.text_parts.append(data)

    def get_text(self) -> str:
        return "\n".join(self.text_parts)


def _extract_from_html(html: str) -> tuple[list[str], list[str]]:
    p = _TextAndScriptsParser()
    p.feed(html)
    text = html_unescape(p.get_text())

    # Extract from both visible text and raw HTML attributes.
    full = _extract_full_urls(text) + _extract_full_urls(html) + _extract_schemeless_polymarket_urls(text) + _extract_schemeless_polymarket_urls(html)
    rel = _extract_relative_paths(text) + _extract_relative_paths(html)
    return (full, rel)


def _extract_from_text(text: str) -> tuple[list[str], list[str]]:
    return (_extract_full_urls(text) + _extract_schemeless_polymarket_urls(text), _extract_relative_paths(text))


def _collect_docs_pages(fetcher: Fetcher) -> list[str]:
    llms = fetcher.get(DOCS_LLMS_TXT)
    urls, _meta = _parse_llms_index(llms)
    return urls


def _collect_polymarket_js_urls(fetcher: Fetcher) -> list[str]:
    html = fetcher.get(POLYMARKET_HOME)
    p = _TextAndScriptsParser()
    p.feed(html)
    scripts = []
    for src in p.script_srcs:
        src = (src or "").strip()
        if not src:
            continue
        scripts.append(urllib.parse.urljoin(POLYMARKET_HOME, src))
    # de-dupe
    out = []
    seen = set()
    for u in scripts:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _scan_tar_gz_text_files(
    *,
    archive_bytes: bytes,
    allowed_suffixes: tuple[str, ...],
    max_file_bytes: int = 2_000_000,
) -> dict[str, str]:
    """
    Returns { member_name: decoded_text } for text-like members.
    """
    out: dict[str, str] = {}
    bio = io.BytesIO(archive_bytes)
    with tarfile.open(fileobj=bio, mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name or ""
            if not name.endswith(allowed_suffixes):
                continue
            if m.size is not None and m.size > max_file_bytes:
                continue
            f = tf.extractfile(m)
            if not f:
                continue
            raw = f.read(max_file_bytes + 1)
            if len(raw) > max_file_bytes:
                continue
            try:
                out[name] = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
    return out


def _collect_sdk_endpoints_npm_clob_client(fetcher: Fetcher) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Returns (sources, items) where items are (source_url, text).
    """
    reg = json.loads(fetcher.get(NPM_CLOB_CLIENT_REGISTRY))
    ver = (reg.get("dist-tags") or {}).get("latest")
    if not ver:
        raise RuntimeError("npm registry: missing dist-tags.latest")
    vinfo = (reg.get("versions") or {}).get(ver) or {}
    tarball = ((vinfo.get("dist") or {}).get("tarball") or "").strip()
    if not tarball:
        raise RuntimeError("npm registry: missing tarball url")

    tgz = fetcher.get_bytes(tarball)
    files = _scan_tar_gz_text_files(archive_bytes=tgz, allowed_suffixes=(".js", ".ts", ".mjs", ".cjs", ".json"))
    sources = [tarball]
    items: list[tuple[str, str]] = []
    for name, text in files.items():
        items.append((f"{tarball}#{name}", text))
    return sources, items


def _collect_sdk_endpoints_pypi_py_clob_client(fetcher: Fetcher) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Returns (sources, items) where items are (source_url, text).
    """
    meta = json.loads(fetcher.get(PYPI_CLOB_CLIENT_JSON))
    urls = meta.get("urls") or []
    sdist = ""
    for u in urls:
        if (u.get("packagetype") or "") == "sdist" and (u.get("url") or "").endswith((".tar.gz", ".tgz")):
            sdist = u.get("url") or ""
            break
    if not sdist and urls:
        # fall back to first url
        sdist = (urls[0].get("url") or "").strip()
    if not sdist:
        raise RuntimeError("pypi: missing sdist url")

    tgz = fetcher.get_bytes(sdist)
    files = _scan_tar_gz_text_files(archive_bytes=tgz, allowed_suffixes=(".py", ".json", ".md", ".txt"))
    sources = [sdist]
    items: list[tuple[str, str]] = []
    for name, text in files.items():
        items.append((f"{sdist}#{name}", text))
    return sources, items


def _filter_to_polymarketish(urls: Iterable[str]) -> list[str]:
    out: list[str] = []
    for u in urls:
        s = _normalize_urlish(u)
        if not s.startswith("http://") and not s.startswith("https://"):
            continue
        try:
            host = urllib.parse.urlparse(s).netloc.lower()
        except ValueError:
            # Some scraped strings look like URLs but aren't valid.
            continue
        if not host or not re.match(r"^[a-z0-9.-]+$", host):
            continue
        if host.startswith("-") or host.endswith("-"):
            continue
        if host == "polymarket.com" or host.endswith(".polymarket.com"):
            out.append(s)
            continue
    return out


def _make_endpoint_items(
    *,
    source_url: str,
    source_kind: str,
    text: str,
    full_urls: Iterable[str],
    rel_paths: Iterable[str],
    docs_meta_by_url: dict[str, dict[str, str]] | None = None,
    force_origin: str | None = None,
    max_evidence_len: int = 240,
) -> list[Endpoint]:
    items: list[Endpoint] = []
    origin = _source_origin(source_url)
    docs_meta_by_url = docs_meta_by_url or {}

    def add(raw: str) -> None:
        norm = _defrag(_normalize_urlish(raw))
        if not _looks_like_endpoint(norm):
            return
        if "${" in norm:
            return
        # If it's a relative endpoint, attach it to a best-guess origin.
        if norm.startswith("/"):
            base = force_origin or _pick_base_for_relative_path(norm, source_url, text) or origin
            if base:
                norm = urllib.parse.urljoin(base, norm)
        host, path = _parse_host_path(norm)
        method = _infer_method_near(text, raw) or _infer_method_near(text, norm)
        # Skip broken template fragments.
        if "{var}" in norm and norm.count("{var}") >= 2:
            return
        meta = docs_meta_by_url.get(source_url) or {}
        desc = (meta.get("summary") or meta.get("title") or "").strip()
        ctx = norm
        if len(ctx) > max_evidence_len:
            ctx = ctx[: max_evidence_len - 3] + "..."
        items.append(
            Endpoint(
                raw=raw,
                normalized=norm,
                host=host,
                path=path,
                method=method,
                description=desc,
                source_kind=source_kind,
                evidence=Evidence(source_url=source_url, context=ctx),
            )
        )

    for u in full_urls:
        add(u)
    for p in rel_paths:
        add(p)
    return items


def _key(ep: Endpoint) -> str:
    # method may be unknown; keep method in key only if present
    return f"{ep.method or '*'} {ep.normalized}"


def _is_doc_like_path(path: str) -> bool:
    # These are documentation sections, not API endpoints.
    return any(
        seg in (path or "")
        for seg in (
            "/api-reference/",
            "/market-data/",
            "/market-makers/",
            "/trading/",
            "/concepts/",
            "/builders/",
            "/resources/",
        )
    )


def _heuristic_description(ep: Endpoint) -> str:
    host = (ep.host or "").lower()
    path = ep.path or ""
    method = (ep.method or "").upper()

    def m(desc: str) -> str:
        # Keep it short and consistent.
        return desc.strip().rstrip(".") + "."

    if host == "gamma-api.polymarket.com":
        if path == "/events":
            return m("List events")
        if path.startswith("/events/slug"):
            return m("Get event by slug")
        if path.startswith("/events/"):
            return m("Get event by id")
        if path == "/markets":
            return m("List markets")
        if path.startswith("/markets/"):
            return m("Get market by id")
        if path == "/tags":
            return m("List tags")
        if path == "/comments":
            return m("List or create comments")
        if path.startswith("/comments/"):
            return m("Get or manage a comment")
        if path.startswith("/search"):
            return m("Search markets, events, and profiles")
        return m("Gamma API endpoint")

    if host == "clob.polymarket.com":
        if path == "/book":
            return m("Get order book summary for a token_id")
        if path == "/books":
            return m("Get order books for multiple token_ids")
        if path == "/midpoint":
            return m("Get midpoint price for a token_id")
        if path == "/orderbook-history":
            return m("Get order book history for an asset_id (time range)")
        if path == "/order":
            if method == "POST":
                return m("Create a new order")
            if method == "DELETE":
                return m("Cancel a single order")
            return m("Create or cancel an order")
        if path == "/orders":
            if method == "GET":
                return m("List user orders")
            if method == "DELETE":
                return m("Cancel multiple orders")
            return m("List or cancel orders")
        if path.startswith("/auth/"):
            return m("Authentication / API key flow")
        if "cancel" in path:
            return m("Cancel orders")
        if "trade" in path or path.startswith("/trades"):
            return m("Trades data")
        return m("CLOB API endpoint")

    if host == "bridge.polymarket.com":
        if path == "/quote":
            return m("Get bridge quote")
        if path == "/supported-assets":
            return m("List supported bridge assets")
        if path.startswith("/status"):
            return m("Get bridge transaction status")
        if path == "/deposit":
            return m("Create deposit addresses / start deposit")
        if path == "/withdraw":
            return m("Create withdrawal addresses / start withdrawal")
        return m("Bridge API endpoint")

    if host.endswith(".polymarket.com"):
        return m("Polymarket endpoint (purpose not in docs index)")

    return m("Endpoint (purpose unknown)")


def _is_specific_docs_source(url: str) -> bool:
    """
    True if the docs page is likely dedicated to a specific API endpoint.
    """
    if not url.startswith("https://docs.polymarket.com/"):
        return False
    p = urllib.parse.urlparse(url).path
    # Exclude "general" pages that mention many endpoints.
    if p.endswith(("/api-reference/rate-limits.md", "/api-reference/clients-sdks.md", "/api-reference/introduction.md")):
        return False
    return ("/api-reference/" in p) or ("/trading/bridge/" in p) or p.startswith("/builders/")


def _best_description(ep: Endpoint) -> str:
    """
    Prefer endpoint-specific docs descriptions; otherwise fall back to heuristics.
    """
    d = (ep.description or "").strip()
    if d and ep.source_kind == "probe":
        return d if d.endswith(".") else d + "."
    if d and ep.source_kind == "docs" and _is_specific_docs_source(ep.evidence.source_url):
        # Ensure it ends with a period for consistent rendering.
        return d if d.endswith(".") else d + "."
    return _heuristic_description(ep)


def _filter_endpoints(
    endpoints: list[Endpoint],
    *,
    all_hosts: bool,
    include_webapp: bool,
    include_staging: bool,
    include_hosts: set[str],
    exclude_hosts: set[str],
) -> list[Endpoint]:
    if all_hosts:
        return endpoints

    allowed = set(DEFAULT_PUBLIC_HOSTS) | set(include_hosts)
    if include_webapp:
        allowed.add("polymarket.com")
    if include_staging:
        allowed.add("clob-staging.polymarket.com")

    out: list[Endpoint] = []
    for ep in endpoints:
        host = (ep.host or "").lower()
        if not host:
            continue
        if host in DEFAULT_EXCLUDED_HOSTS:
            continue
        if host in exclude_hosts:
            continue
        if host not in allowed:
            continue
        if _is_doc_like_path(ep.path or ""):
            continue
        out.append(ep)
    return out


def _drop_methodless_if_methodful(endpoints: list[Endpoint]) -> list[Endpoint]:
    """
    If we have both:
      - "* https://host/path" (unknown method)
      - "GET/POST/... https://host/path"
    keep only the method-specific ones.
    """
    buckets: dict[str, list[Endpoint]] = {}
    for ep in endpoints:
        buckets.setdefault(ep.normalized, []).append(ep)

    out: list[Endpoint] = []
    for _norm, lst in buckets.items():
        if any(e.method for e in lst):
            out.extend([e for e in lst if e.method])
        else:
            out.extend(lst)
    return out


def _http_status(fetcher: Fetcher, url: str) -> int:
    """
    Returns HTTP status code, or -1 on network/unknown errors.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": fetcher.user_agent, "Accept": "*/*"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=fetcher.timeout_s) as resp:
            return int(getattr(resp, "status", 200))
    except urllib.error.HTTPError as e:
        return int(getattr(e, "code", 0) or 0)
    except Exception:
        return -1


def _probe_clob_read_endpoints(fetcher: Fetcher) -> list[Endpoint]:
    """
    Best-effort discovery for undocumented CLOB read endpoints.
    We only send GET requests with dummy params and treat non-404 as "exists".
    """
    candidates: list[tuple[str, str, dict[str, str]]] = [
        (
            "/orderbook-history",
            "Order book history (time range).",
            {"asset_id": "0", "startTs": "0", "endTs": "1", "limit": "1", "offset": "0"},
        ),
    ]
    out: list[Endpoint] = []
    for path, desc, params in candidates:
        base_url = urllib.parse.urljoin(CLOB_ORIGIN, path)
        url = base_url
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        status = _http_status(fetcher, url)
        if status in (404, -1):
            continue
        host, p = _parse_host_path(base_url)
        out.append(
            Endpoint(
                raw=url,
                normalized=base_url,
                host=host,
                path=p,
                method="GET",
                description=desc,
                source_kind="probe",
                evidence=Evidence(source_url=url, context=f"probe_status={status}"),
            )
        )
    return out


def _load_seen(cache_path: Path) -> set[str]:
    s = _read_text(cache_path)
    if not s:
        return set()
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return set(str(x) for x in obj)
    except Exception:
        pass
    return set()


def _save_seen(cache_path: Path, keys: Iterable[str]) -> None:
    _write_json(cache_path, sorted(set(keys)))


def _render_md(endpoints: list[Endpoint], meta: dict) -> str:
    # Group by host; relative endpoints go under "(relative)"
    groups: dict[str, list[Endpoint]] = {}
    for ep in endpoints:
        h = ep.host or "(relative)"
        groups.setdefault(h, []).append(ep)

    lines: list[str] = []
    lines.append("# Polymarket Endpoints Catalog")
    lines.append("")
    lines.append("Generated file. Grouped endpoint list with evidence sources.")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **generated_at**: `{meta.get('generated_at')}`")
    if meta.get("filters"):
        lines.append(f"- **filters**: `{json.dumps(meta.get('filters'), ensure_ascii=False)}`")
    lines.append(f"- **sources**: {len(meta.get('sources', []))}")
    lines.append(f"- **total_endpoints**: {len(endpoints)}")
    lines.append("")
    lines.append("## Hosts")
    lines.append("")
    for host in sorted(groups.keys()):
        lines.append(f"- `{host}`: {len(groups[host])} endpoints")
    lines.append("")

    for host in sorted(groups.keys()):
        lines.append(f"## {host}")
        lines.append("")
        # Sort by path then method
        eps = sorted(groups[host], key=lambda e: (e.path or e.normalized, e.method or ""))
        for ep in eps:
            m = ep.method or "*"
            desc = _best_description(ep)
            lines.append(f"- **{m}** `{ep.normalized}` — {desc}")
            lines.append(f"  - evidence: `{ep.evidence.source_url}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_diff_md(added: list[str], removed: list[str], meta: dict) -> str:
    lines: list[str] = []
    lines.append("# Polymarket Endpoints Diff")
    lines.append("")
    lines.append(f"- **generated_at**: `{meta.get('generated_at')}`")
    lines.append(f"- **added**: {len(added)}")
    lines.append(f"- **removed**: {len(removed)}")
    lines.append("")

    lines.append("## Added")
    lines.append("")
    if added:
        for a in added:
            lines.append(f"- `{a}`")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Removed")
    lines.append("")
    if removed:
        for r in removed:
            lines.append(f"- `{r}`")
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Fetch Polymarket endpoints from docs + website JS bundles.")
    ap.add_argument("--out-dir", default="out", help="Output directory (default: out)")
    ap.add_argument("--cache-dir", default=".cache", help="Cache directory (default: .cache)")
    ap.add_argument("--no-cache", action="store_true", help="Disable cache/diff output")
    ap.add_argument("--seed-url", action="append", default=[], help="Extra URL to crawl (repeatable)")
    ap.add_argument("--all-hosts", action="store_true", help="Do not filter hosts (include everything found)")
    ap.add_argument("--include-webapp", action="store_true", help="Include polymarket.com internal webapp endpoints")
    ap.add_argument("--include-staging", action="store_true", help="Include staging hosts (e.g. clob-staging)")
    ap.add_argument("--include-host", action="append", default=[], help="Include an extra host (repeatable)")
    ap.add_argument("--exclude-host", action="append", default=[], help="Exclude a host (repeatable)")
    ap.add_argument("--scan-sdks", action="store_true", help="Scan official SDK packages for endpoints (slower)")
    ap.add_argument("--probe-clob", action="store_true", help="Probe for undocumented CLOB read endpoints (safe GETs)")
    ap.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds (default: 20)")
    ap.add_argument("--max-pages", type=int, default=400, help="Max docs pages to crawl (default: 400)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    cache_seen = cache_dir / "seen_endpoint_keys.json"

    fetcher = Fetcher(timeout_s=args.timeout)

    sources: list[str] = []
    endpoints: list[Endpoint] = []
    docs_meta_by_url: dict[str, dict[str, str]] = {}

    # 1) Docs pages from llms.txt
    try:
        llms = fetcher.get(DOCS_LLMS_TXT)
        docs_pages, docs_meta_by_url = _parse_llms_index(llms)
        docs_pages = docs_pages[: args.max_pages]
        sources.extend(docs_pages)
    except Exception as e:
        print(f"[warn] unable to load docs index {DOCS_LLMS_TXT}: {e}", file=sys.stderr)
        docs_pages = []

    # 2) Polymarket homepage scripts (JS bundles)
    try:
        js_urls = _collect_polymarket_js_urls(fetcher)
        sources.extend(js_urls)
    except Exception as e:
        print(f"[warn] unable to load {POLYMARKET_HOME} scripts: {e}", file=sys.stderr)
        js_urls = []

    # 3) Extra seed URLs
    for u in args.seed_url:
        sources.append(u)

    # 4) Optional: scan official SDKs (helps catch undocumented endpoints)
    sdk_items: list[tuple[str, str]] = []
    if args.scan_sdks:
        try:
            s, items = _collect_sdk_endpoints_npm_clob_client(fetcher)
            sources.extend(s)
            sdk_items.extend(items)
        except Exception as e:
            print(f"[warn] sdk scan (npm clob-client) failed: {e}", file=sys.stderr)
        try:
            s, items = _collect_sdk_endpoints_pypi_py_clob_client(fetcher)
            sources.extend(s)
            sdk_items.extend(items)
        except Exception as e:
            print(f"[warn] sdk scan (pypi py-clob-client) failed: {e}", file=sys.stderr)

    # Fetch + extract
    seen_norm: set[str] = set()

    def add_items(items: Iterable[Endpoint]) -> None:
        nonlocal endpoints
        for it in items:
            k = _key(it)
            if k in seen_norm:
                continue
            seen_norm.add(k)
            endpoints.append(it)

    # Crawl docs pages
    for url in docs_pages:
        try:
            html = fetcher.get(url)
        except Exception:
            continue
        full, rel = _extract_from_html(html)
        full = _filter_to_polymarketish(full)
        add_items(
            _make_endpoint_items(
                source_url=url,
                source_kind="docs",
                text=html,
                full_urls=full,
                rel_paths=rel,
                docs_meta_by_url=docs_meta_by_url,
            )
        )

    # Crawl JS bundles
    for url in js_urls:
        try:
            js = fetcher.get(url)
        except Exception:
            continue
        full, rel = _extract_from_text(js)
        full = _filter_to_polymarketish(full)
        add_items(
            _make_endpoint_items(
                source_url=url,
                source_kind="js",
                text=js,
                full_urls=full,
                rel_paths=rel,
                docs_meta_by_url=docs_meta_by_url,
            )
        )

    # Crawl seed URLs (best-effort: html/text)
    for url in args.seed_url:
        try:
            body = fetcher.get(url)
        except Exception:
            continue
        full, rel = _extract_from_text(body)
        full = _filter_to_polymarketish(full)
        add_items(
            _make_endpoint_items(
                source_url=url,
                source_kind="seed",
                text=body,
                full_urls=full,
                rel_paths=rel,
                docs_meta_by_url=docs_meta_by_url,
            )
        )

    # Crawl SDK package contents
    for src_url, text in sdk_items:
        full, rel = _extract_from_text(text)
        full = _filter_to_polymarketish(full)
        add_items(
            _make_endpoint_items(
                source_url=src_url,
                source_kind="sdk",
                text=text,
                full_urls=full,
                rel_paths=rel,
                docs_meta_by_url=docs_meta_by_url,
                force_origin=CLOB_ORIGIN,
            )
        )

    # Optional: probe endpoints
    if args.probe_clob:
        try:
            add_items(_probe_clob_read_endpoints(fetcher))
        except Exception as e:
            print(f"[warn] probe failed: {e}", file=sys.stderr)

    # Final sort stable
    endpoints = _drop_methodless_if_methodful(endpoints)
    include_hosts = {h.strip().lower() for h in (args.include_host or []) if h.strip()}
    exclude_hosts = {h.strip().lower() for h in (args.exclude_host or []) if h.strip()}
    endpoints_filtered = _filter_endpoints(
        endpoints,
        all_hosts=bool(args.all_hosts),
        include_webapp=bool(args.include_webapp),
        include_staging=bool(args.include_staging),
        include_hosts=include_hosts,
        exclude_hosts=exclude_hosts,
    )
    endpoints_sorted = sorted(endpoints_filtered, key=lambda e: (e.host or "", e.path or e.normalized, e.method or "", e.normalized))

    meta = {
        "generated_at": _now_iso(),
        "sources": sorted(set(sources)),
        "filters": {
            "all_hosts": bool(args.all_hosts),
            "include_webapp": bool(args.include_webapp),
            "include_staging": bool(args.include_staging),
            "include_host": sorted(include_hosts),
            "exclude_host": sorted(exclude_hosts),
        },
        "counts": {
            "sources_total": len(sources),
            "endpoints_total": len(endpoints_sorted),
        },
        "build_id": _sha1(_now_iso() + str(len(endpoints_sorted))),
    }

    # Write outputs
    md_path = out_dir / "polymarket_endpoints.md"
    json_path = out_dir / "polymarket_endpoints.json"
    diff_path = out_dir / "polymarket_endpoints_diff.md"

    _write_text(md_path, _render_md(endpoints_sorted, meta))

    json_obj = {
        "meta": meta,
        "endpoints": [
            {
                "normalized": e.normalized,
                "raw": e.raw,
                "method": e.method,
                "host": e.host,
                "path": e.path,
                "description": _best_description(e),
                "source_kind": e.source_kind,
                "evidence": dataclasses.asdict(e.evidence),
            }
            for e in endpoints_sorted
        ],
    }
    _write_json(json_path, json_obj)

    if not args.no_cache:
        prev = _load_seen(cache_seen)
        cur = set(_key(e) for e in endpoints_sorted)
        added = sorted(cur - prev)
        removed = sorted(prev - cur)
        _write_text(diff_path, _render_diff_md(added, removed, meta))
        _save_seen(cache_seen, cur)

    print(f"wrote: {md_path}")
    print(f"wrote: {json_path}")
    if not args.no_cache:
        print(f"wrote: {diff_path}")
        print(f"cache: {cache_seen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

