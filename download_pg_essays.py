#!/usr/bin/env python3
"""
Download all Paul Graham essay text from https://paulgraham.com/articles.html

Saves each essay as essays/<slug>.txt with title on the first line, then the body.
Usage: python download_pg_essays.py
"""

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://paulgraham.com"
INDEX_URL = f"{BASE_URL}/articles.html"
OUT_DIR = Path(__file__).resolve().parent / "essays"
SKIP_HREFS = {"index.html", "articles.html", "rss.html"}


def get_essay_urls(index_url: str = INDEX_URL) -> list[tuple[str, str, str]]:
    """Fetch index page and return list of (title, url, slug) for each essay."""
    r = requests.get(index_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.endswith(".html"):
            continue
        # Resolve relative URLs
        if href.startswith("http"):
            if "paulgraham.com" not in href:
                continue
            url = href
            slug = href.split("/")[-1].replace(".html", "")
        else:
            if href in SKIP_HREFS:
                continue
            url = f"{BASE_URL}/{href}" if not href.startswith("/") else f"{BASE_URL}{href}"
            slug = href.split("/")[-1].replace(".html", "")
        title = (a.get_text() or "").strip() or slug
        out.append((title, url, slug))
    # Dedupe by URL, keep first occurrence
    seen = set()
    unique = []
    for title, url, slug in out:
        if url in seen:
            continue
        seen.add(url)
        unique.append((title, url, slug))
    return unique


def fetch_essay_text(url: str) -> tuple[str, str]:
    """Fetch essay page and return (title, body_text)."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Try to get title from <title> or first heading
    title = ""
    if soup.title:
        title = soup.title.get_text().strip()
        # Often "Essay Name --" or "Essay Name | Paul Graham"
        title = re.sub(r"\s*[|–-].*$", "", title).strip()
    for tag in ("h1", "h2"):
        el = soup.find(tag)
        if el:
            t = el.get_text().strip()
            if t and len(t) > 2:
                title = t
                break

    # Main content: PG's site often has content in the body or in a table cell.
    # Prefer the largest text block that looks like an essay (many spaces/paragraphs).
    body = soup.body
    if not body:
        body = soup
    text = body.get_text(separator="\n", strip=True)
    # Normalize multiple newlines to double newline (paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return title, text


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching index from {INDEX_URL} ...")
    essays = get_essay_urls()
    print(f"Found {len(essays)} essays.")

    for i, (title, url, slug) in enumerate(essays, 1):
        path = OUT_DIR / f"{slug}.txt"
        if path.exists():
            print(f"[{i}/{len(essays)}] Skip (exists): {slug}.txt")
            continue
        try:
            title_fetched, body = fetch_essay_text(url)
            if title_fetched:
                title = title_fetched
            content = f"{title}\n\n{body}"
            path.write_text(content, encoding="utf-8")
            print(f"[{i}/{len(essays)}] Saved: {slug}.txt ({len(body)} chars)")
        except Exception as e:
            print(f"[{i}/{len(essays)}] Error {slug}: {e}")
        time.sleep(0.5)  # Be nice to the server

    print(f"Done. Essays in {OUT_DIR}")


if __name__ == "__main__":
    main()
