"""
Scrape Federal Reserve text documents.

Minimum version:
- FOMC statements only.

Output:
- data/raw/fed_text_documents.parquet

Fields:
- document_date
- document_type
- title
- url
- raw_text
- clean_text

Lookahead-bias rule:
- This script stores the actual document publication date.
- Downstream feature construction must only use rows where:
    document_date <= feature_date
"""

from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.federalreserve.gov"
START_DATE = pd.Timestamp("2010-01-01")
END_DATE = pd.Timestamp(date.today())

OUTPUT_PATH = Path("data/raw/fed_text_documents.parquet")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 FedNarrativeAlpha research scraper "
        "(educational use; contact: GitHub portfolio project)"
    )
}

REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 0.25


def fetch_soup(url: str) -> BeautifulSoup:
    """Fetch a webpage and return a BeautifulSoup object."""
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into a clean single-spaced string."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_date_from_url(url: str) -> pd.Timestamp | None:
    """
    Extract YYYYMMDD from Federal Reserve monetary-policy press-release URLs.

    Examples:
    - /newsevents/pressreleases/monetary20250618a.htm
    - /newsevents/press/monetary/20100127a.htm
    """
    patterns = [
        r"monetary(\d{8})a\.htm",
        r"/monetary/(\d{8})a\.htm",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return pd.to_datetime(match.group(1), format="%Y%m%d")

    return None


def extract_title(soup: BeautifulSoup) -> str:
    """Extract a reasonable title from a Federal Reserve document page."""
    for selector in ["h1", "h2", "h3", "title"]:
        tag = soup.select_one(selector)
        if tag:
            title = normalize_whitespace(tag.get_text(" ", strip=True))
            if title:
                return title

    return "Federal Reserve document"


def extract_article_text(soup: BeautifulSoup, title: str) -> str:
    """
    Extract article text while removing most page navigation boilerplate.

    Federal Reserve pages vary over time, so this uses conservative fallbacks.
    """
    soup_copy = BeautifulSoup(str(soup), "html.parser")

    for tag in soup_copy.select("script, style, nav, header, footer, form, noscript"):
        tag.decompose()

    containers = [
        soup_copy.select_one("main"),
        soup_copy.select_one("#content"),
        soup_copy.select_one(".content"),
        soup_copy.body,
    ]

    raw = ""
    for container in containers:
        if container is None:
            continue

        candidate = container.get_text("\n", strip=True)
        if (
            "For immediate release" in candidate
            or "For release at" in candidate
            or "FOMC statement" in candidate
            or "Federal Reserve issues FOMC statement" in candidate
        ):
            raw = candidate
            break

    if not raw:
        raw = soup_copy.get_text("\n", strip=True)

    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    # Start near the actual press-release content.
    start_idx = 0
    lowered_title = title.lower()

    for i, line in enumerate(lines):
        low = line.lower()
        if lowered_title and lowered_title in low:
            start_idx = i
            break
        if low.startswith("for immediate release") or low.startswith("for release at"):
            start_idx = max(0, i - 2)
            break

    lines = lines[start_idx:]

    # Stop before footer boilerplate.
    stop_idx = len(lines)
    stop_markers = [
        "Last Update:",
        "Board of Governors of the Federal Reserve System",
        "Stay Connected",
        "Tools and Information",
    ]

    for i, line in enumerate(lines):
        if any(marker in line for marker in stop_markers):
            stop_idx = i
            break

    lines = lines[:stop_idx]

    # Remove obvious non-content fragments.
    remove_exact = {
        "Share",
        "Press Release",
        "Please enable JavaScript if it is disabled in your browser or access the information through the links provided below.",
    }

    lines = [line for line in lines if line not in remove_exact]

    return "\n".join(lines).strip()


def is_fomc_statement_link(anchor_text: str, href: str) -> bool:
    """
    Identify FOMC statement HTML links from calendar and historical pages.

    This intentionally excludes minutes, implementation notes, PDFs,
    transcripts, speeches, testimony, and longer-run goals statements.
    """
    text = normalize_whitespace(anchor_text).lower()

    if href.endswith(".pdf"):
        return False

    url_has_statement_pattern = bool(
        re.search(r"monetary\d{8}a\.htm", href) or re.search(r"/monetary/\d{8}a\.htm", href)
    )

    if not url_has_statement_pattern:
        return False

    # Historical pages often use anchor text "Statement".
    # Recent calendar pages often use anchor text "HTML" under the Statement label.
    allowed_anchor_text = text in {"statement", "html"}

    return allowed_anchor_text


def collect_statement_urls() -> list[str]:
    """
    Collect FOMC statement URLs from:
    - current FOMC calendar page
    - annual historical pages from START_DATE.year through current year
    """
    current_year = date.today().year

    index_urls = [
        f"{BASE_URL}/monetarypolicy/fomccalendars.htm",
    ]

    for year in range(START_DATE.year, current_year + 1):
        index_urls.append(f"{BASE_URL}/monetarypolicy/fomchistorical{year}.htm")

    statement_urls: set[str] = set()

    for index_url in index_urls:
        try:
            soup = fetch_soup(index_url)
        except requests.HTTPError:
            # Some annual pages may not exist or may be replaced by the main calendar.
            continue

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            anchor_text = a_tag.get_text(" ", strip=True)

            if is_fomc_statement_link(anchor_text, href):
                full_url = urljoin(BASE_URL, href)
                statement_urls.add(full_url)

        time.sleep(SLEEP_SECONDS)

    return sorted(statement_urls)


def scrape_statement(url: str) -> dict[str, object] | None:
    """Scrape one FOMC statement page."""
    try:
        soup = fetch_soup(url)
    except requests.RequestException as exc:
        print(f"WARNING: failed to fetch {url}: {exc}")
        return None

    final_url = url
    title = extract_title(soup)
    document_date = extract_date_from_url(final_url)

    if document_date is None:
        print(f"WARNING: could not parse date from URL: {final_url}")
        return None

    if document_date < START_DATE or document_date > END_DATE:
        return None

    raw_text = extract_article_text(soup, title)
    clean_text = normalize_whitespace(raw_text)

    if len(clean_text) < 100:
        print(f"WARNING: unusually short text for {final_url}")
        return None

    return {
        "document_date": document_date.date(),
        "document_type": "fomc_statement",
        "title": title,
        "url": final_url,
        "raw_text": raw_text,
        "clean_text": clean_text,
    }


def validate_documents(df: pd.DataFrame) -> None:
    """Run basic quality checks on the scraped document dataset."""
    required_columns = [
        "document_date",
        "document_type",
        "title",
        "url",
        "raw_text",
        "clean_text",
    ]

    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("No Federal Reserve text documents were scraped.")

    if df["url"].duplicated().any():
        duplicate_urls = df.loc[df["url"].duplicated(), "url"].tolist()
        raise ValueError(f"Duplicate URLs found: {duplicate_urls[:5]}")

    if df[required_columns].isna().any().any():
        raise ValueError("Missing values found in required columns.")

    if (df["clean_text"].str.len() < 100).any():
        raise ValueError("Some documents have suspiciously short clean_text values.")

    if pd.to_datetime(df["document_date"]).max() > END_DATE:
        raise ValueError("Found documents dated after today's date.")


def documents_available_as_of(df: pd.DataFrame, feature_date: str | pd.Timestamp) -> pd.DataFrame:
    """
    Helper for downstream feature engineering.

    This enforces the no-lookahead rule:
    only documents published on or before feature_date are available.
    """
    feature_date = pd.to_datetime(feature_date)
    document_dates = pd.to_datetime(df["document_date"])

    return df.loc[document_dates <= feature_date].copy()


def main() -> None:
    """Collect FOMC statements and save them as parquet."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    urls = collect_statement_urls()
    print(f"Collected {len(urls)} candidate FOMC statement URLs.")

    records = []

    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{len(urls)}] Scraping {url}")
        record = scrape_statement(url)

        if record is not None:
            records.append(record)

        time.sleep(SLEEP_SECONDS)

    df = pd.DataFrame.from_records(records)

    if not df.empty:
        df = df.sort_values(["document_date", "url"]).drop_duplicates("url").reset_index(drop=True)

    validate_documents(df)

    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} Federal Reserve text documents to {OUTPUT_PATH}")
    print(f"Date range: {df['document_date'].min()} to {df['document_date'].max()}")
    print(df[["document_date", "document_type", "title", "url"]].tail())


if __name__ == "__main__":
    main()
