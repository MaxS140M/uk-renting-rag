"""Download GOV.UK renting guidance listed in sources.txt and save it as clean JSON documents.

Uses the GOV.UK Content API (https://www.gov.uk/api/content/<path>), which returns each page's
main content as structured JSON, so there is no navigation, cookie banner or footer to strip.

Usage:
    python scripts/download_docs.py            # download pages not already in data/raw/
    python scripts/download_docs.py --refresh  # re-download everything
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = ROOT / "scripts" / "sources.txt"
RAW_DIR = ROOT / "data" / "raw"

API_BASE = "https://www.gov.uk/api/content"
SITE_BASE = "https://www.gov.uk"
USER_AGENT = "uk-renting-rag/0.1 (portfolio project; +https://github.com/MaxS140M/uk-renting-rag)"
REQUEST_DELAY_S = 1.0  # pause between requests, to be polite to GOV.UK
MAX_RETRIES = 4
TIMEOUT_S = 30

log = logging.getLogger("download_docs")


class FetchError(Exception):
    """Raised when a page cannot be fetched after all retries."""


# --- HTTP -------------------------------------------------------------------------------------


def fetch_content(path: str) -> dict:
    """Fetch one Content API item, retrying transient failures with exponential backoff."""
    url = API_BASE + path
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                data = json.load(response)
            time.sleep(REQUEST_DELAY_S)
            return data
        except urllib.error.HTTPError as err:
            # 4xx (except rate limiting) will not succeed on retry, so fail immediately.
            if err.code != 429 and err.code < 500:
                raise FetchError(f"HTTP {err.code} for {url}") from err
            retry_after = err.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
            reason = f"HTTP {err.code}"
        except (urllib.error.URLError, TimeoutError) as err:
            wait = 2**attempt
            reason = str(err)
        if attempt == MAX_RETRIES:
            raise FetchError(f"{reason} for {url} after {MAX_RETRIES} attempts")
        log.warning(
            "%s for %s, retrying in %.0fs (attempt %d/%d)", reason, path, wait, attempt, MAX_RETRIES
        )
        time.sleep(wait)
    raise AssertionError("unreachable")


# --- HTML to text -----------------------------------------------------------------------------

BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "ul",
    "ol",
    "table",
    "tr",
    "blockquote",
    "br",
    "details",
    "summary",
    "dl",
    "dt",
    "dd",
}
HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 4, "h6": 4}
SKIP_TAGS = {"script", "style", "svg", "button"}


class _TextExtractor(HTMLParser):
    """Convert GOV.UK body HTML to plain text, keeping headings as Markdown '#' lines."""

    def __init__(self, heading_offset: int) -> None:
        super().__init__(convert_charrefs=True)
        self.heading_offset = heading_offset
        self.blocks: list[str] = []
        self.current: list[str] = []
        self.prefix = ""
        self.skip_depth = 0

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self.current)).strip()
        if text:
            self.blocks.append(self.prefix + text)
        self.current = []
        self.prefix = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self.skip_depth += 1
        elif tag in HEADING_TAGS:
            self._flush()
            level = min(HEADING_TAGS[tag] + self.heading_offset, 4)
            self.prefix = "#" * level + " "
        elif tag == "li":
            self._flush()
            self.prefix = "- "
        elif tag in ("td", "th"):
            if self.current:
                self.current.append(" | ")
        elif tag in BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag in HEADING_TAGS or tag == "li" or tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.current.append(data)

    def text(self) -> str:
        self._flush()
        # Blank line between blocks, but keep consecutive list items on adjacent lines.
        out: list[str] = []
        for block in self.blocks:
            if out and not (block.startswith("- ") and out[-1].startswith("- ")):
                out.append("")
            out.append(block)
        return "\n".join(out)


def html_to_text(html: str, heading_offset: int = 0) -> str:
    """Return clean plain text for a fragment of GOV.UK body HTML."""
    parser = _TextExtractor(heading_offset)
    parser.feed(html)
    parser.close()
    return parser.text()


# --- Content API item to document -------------------------------------------------------------


def extract_text(item: dict) -> str:
    """Build the full document text for a Content API item, handling each page format."""
    details = item.get("details", {})
    schema = item.get("schema_name")

    if schema == "guide":
        # Multi-part guide: every part is in the same response. Each part title becomes a
        # top-level heading, and headings inside a part are pushed down one level.
        sections = [
            f"# {part['title']}\n\n{html_to_text(part['body'], heading_offset=1)}"
            for part in details.get("parts", [])
        ]
        return "\n\n".join(sections)

    if schema == "publication":
        # Publications are a short landing page; the real content is in linked HTML
        # publications. Fall back to the landing page body if the content is only a PDF.
        children = [
            c
            for c in item.get("links", {}).get("children", [])
            if c.get("document_type") == "html_publication"
        ]
        if children:
            parts = []
            for child in children:
                child_item = fetch_content(child["base_path"])
                body = html_to_text(child_item["details"]["body"], heading_offset=1)
                parts.append(f"# {child_item['title']}\n\n{body}")
            return "\n\n".join(parts)
        log.warning(
            "%s has no HTML publication (likely PDF only); using landing page text",
            item["base_path"],
        )

    body = details.get("body", "")
    if isinstance(body, list):  # some formats return body as [{content_type, content}]
        body = next((b["content"] for b in body if b.get("content_type") == "text/html"), "")
    return html_to_text(body)


def doc_id_for(path: str) -> str:
    """Stable, filesystem-safe document ID: the last segment of the GOV.UK path."""
    return path.rstrip("/").rsplit("/", 1)[-1]


def build_document(path: str) -> dict:
    """Fetch a GOV.UK page and return it as a document dict ready to save."""
    item = fetch_content(path)
    if item.get("schema_name") == "redirect":
        raise FetchError(f"{path} is a redirect; update sources.txt to its destination")
    if item.get("withdrawn_notice"):
        log.warning("%s is marked as withdrawn on GOV.UK; its content may be out of date", path)

    text = extract_text(item)
    if not text.strip():
        raise FetchError(f"{path} returned no text")

    return {
        "doc_id": doc_id_for(path),
        "title": item["title"].strip(),
        "url": SITE_BASE + path,
        "text": text,
        "date_retrieved": datetime.now(UTC).date().isoformat(),
        "last_updated": item.get("public_updated_at"),
        "document_type": item.get("document_type"),
        "licence": "Open Government Licence v3.0",
    }


# --- CLI --------------------------------------------------------------------------------------


def read_sources(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download pages even if they already exist in data/raw/",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

    paths = read_sources(SOURCES_FILE)
    ids = [doc_id_for(p) for p in paths]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        log.error("Duplicate doc_ids in sources.txt: %s", sorted(duplicates))
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded, skipped, failed = 0, 0, []
    for n, path in enumerate(paths, start=1):
        out_file = RAW_DIR / f"{doc_id_for(path)}.json"
        if out_file.exists() and not args.refresh:
            skipped += 1
            continue
        try:
            doc = build_document(path)
        except FetchError as err:
            log.error("[%d/%d] FAILED %s: %s", n, len(paths), path, err)
            failed.append(path)
            continue
        out_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        downloaded += 1
        log.info(
            "[%d/%d] saved %s (%d words)", n, len(paths), doc["doc_id"], len(doc["text"].split())
        )

    log.info(
        "Done: %d downloaded, %d skipped (already present), %d failed",
        downloaded,
        skipped,
        len(failed),
    )
    for path in failed:
        log.error("Failed: %s", path)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
