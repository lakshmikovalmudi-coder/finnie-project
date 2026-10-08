"""Scrape a curated list of public-domain U.S. government pages into knowledge-base articles.

Only sources that allow automated access are used (robots.txt is checked on every
run). Investor.gov and Bogleheads block bots and FINRA's terms forbid scraping, so
topics from those sites are written by hand instead.

Run:  python scripts/scrape_kb.py            # skip articles that already exist
      python scripts/scrape_kb.py --force    # re-fetch everything (e.g. new IRS limits)
Then: python -m src.rag.ingest
"""

import argparse
import logging
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag

ROOT_DIR = Path(__file__).resolve().parent.parent
KB_DIR = ROOT_DIR / "data" / "knowledge_base"

USER_AGENT = "FinnieEduBot/0.1 (student capstone; educational finance tutor)"
DELAY_S = 3.0  # be polite: one request every few seconds
TIMEOUT_S = 20
MIN_WORDS = 80  # below this the extraction probably grabbed the wrong element

# (filename stem, url). The stem's prefix follows data/knowledge_base/README.md.
# Excluded: irs.gov/retirement-plans/roth-comparison-chart is stale (RMD age 72, limits
# only to 2024) and contradicts the current RMD article.
SOURCES: list[tuple[str, str]] = [
    ("retirement-traditional-ira", "https://www.irs.gov/retirement-plans/traditional-iras"),
    ("retirement-roth-ira", "https://www.irs.gov/retirement-plans/roth-iras"),
    ("retirement-401k-plans", "https://www.irs.gov/retirement-plans/401k-plans"),
    (
        "retirement-ira-contribution-limits",
        "https://www.irs.gov/retirement-plans/plan-participant-employee/retirement-topics-ira-contribution-limits",
    ),
    (
        "retirement-401k-contribution-limits",
        "https://www.irs.gov/retirement-plans/plan-participant-employee/retirement-topics-401k-and-profit-sharing-plan-contribution-limits",
    ),
    (
        "retirement-required-minimum-distributions",
        "https://www.irs.gov/retirement-plans/plan-participant-employee/retirement-topics-required-minimum-distributions-rmds",
    ),
    (
        "retirement-rollovers",
        "https://www.irs.gov/retirement-plans/plan-participant-employee/rollovers-of-retirement-plan-and-ira-distributions",
    ),
    (
        "retirement-early-withdrawal-penalty",
        "https://www.irs.gov/retirement-plans/plan-participant-employee/retirement-topics-tax-on-early-distributions",
    ),
    (
        "retirement-vesting",
        "https://www.irs.gov/retirement-plans/plan-participant-employee/retirement-topics-vesting",
    ),
    (
        "retirement-catch-up-contributions",
        "https://www.irs.gov/retirement-plans/plan-participant-employee/retirement-topics-catch-up-contributions",
    ),
    (
        "retirement-hardship-distributions",
        "https://www.irs.gov/retirement-plans/plan-participant-employee/retirement-topics-hardship-distributions",
    ),
    (
        "retirement-403b-plans",
        "https://www.irs.gov/retirement-plans/irc-403b-tax-sheltered-annuity-plans",
    ),
    (
        "retirement-savers-credit",
        "https://www.irs.gov/retirement-plans/plan-participant-employee/retirement-savings-contributions-savers-credit",
    ),
    ("tax-capital-gains", "https://www.irs.gov/taxtopics/tc409"),
    ("tax-dividends", "https://www.irs.gov/taxtopics/tc404"),
    ("tax-interest-income", "https://www.irs.gov/taxtopics/tc403"),
    ("tax-cost-basis", "https://www.irs.gov/taxtopics/tc703"),
    ("tax-ira-overview", "https://www.irs.gov/taxtopics/tc451"),
    ("tax-roth-ira-contributions", "https://www.irs.gov/taxtopics/tc309"),
    ("tax-pensions-and-annuities", "https://www.irs.gov/taxtopics/tc410"),
    ("tax-standard-deduction", "https://www.irs.gov/taxtopics/tc551"),
    ("tax-income-brackets", "https://www.irs.gov/filing/federal-income-tax-rates-and-brackets"),
    (
        "basics-emergency-fund",
        "https://www.consumerfinance.gov/an-essential-guide-to-building-an-emergency-fund/",
    ),
    (
        "basics-credit-report-vs-score",
        "https://www.consumerfinance.gov/ask-cfpb/what-is-the-difference-between-a-credit-report-and-a-credit-score-en-2069/",
    ),
    (
        "basics-good-credit-score",
        "https://www.consumerfinance.gov/ask-cfpb/how-do-i-get-and-keep-a-good-credit-score-en-318/",
    ),
    (
        "basics-certificate-of-deposit",
        "https://www.consumerfinance.gov/ask-cfpb/what-is-a-certificate-of-deposit-cd-en-917/",
    ),
    (
        "basics-debt-to-income-ratio",
        "https://www.consumerfinance.gov/ask-cfpb/what-is-a-debt-to-income-ratio-en-1791/",
    ),
    (
        "tax-hsa-and-fsa",
        "https://www.consumerfinance.gov/ask-cfpb/what-is-a-flexible-spending-account-fsa-card-health-savings-account-card-hsa-en-417/",
    ),
    (
        "risk-common-scams",
        "https://www.consumerfinance.gov/ask-cfpb/what-are-some-common-types-of-fraud-and-scams-en-2092/",
    ),
    (
        "risk-scam-warning-signs",
        "https://www.consumerfinance.gov/ask-cfpb/what-are-some-classic-warning-signs-of-possible-fraud-and-scams-en-2094/",
    ),
]

# Main-content selectors per host, tried in order; falls back to <main>.
CONTENT_SELECTORS = {
    "www.irs.gov": ["article.pup-article .field--name-body", "article.pup-article"],
    "www.consumerfinance.gov": [".u-layout-grid__main"],
}
PUBLISHER = {
    "www.irs.gov": "Internal Revenue Service (IRS)",
    "www.consumerfinance.gov": "Consumer Financial Protection Bureau (CFPB)",
}
BOILERPLATE_HEADING = re.compile(
    r"^(related|stay informed|browse related|don.t see what|subscribe|share this)", re.I
)
BOILERPLATE_LINE = re.compile(
    r"^(page last (modified|reviewed)|sign up for the latest)"
    r"|^(share this|return to top|take our quiz|https?://\S+)$",
    re.I,
)
NOISE = ["script", "style", "nav", "aside", "form", "button", "noscript", "svg", "img"]

logger = logging.getLogger("scrape_kb")


class Fetcher:
    """HTTP client that honours robots.txt and spaces out requests."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.robots: dict[str, RobotFileParser] = {}
        self.last_request = 0.0

    def _wait(self) -> None:
        pause = self.last_request + DELAY_S - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        self.last_request = time.monotonic()

    def allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        if host not in self.robots:
            self._wait()
            resp = self.session.get(f"https://{host}/robots.txt", timeout=TIMEOUT_S)
            parser = RobotFileParser()
            if resp.status_code != 200:  # blocked or missing robots.txt: don't scrape
                parser.disallow_all = True
            else:
                parser.parse(resp.text.splitlines())
            self.robots[host] = parser
        return self.robots[host].can_fetch(USER_AGENT, url)

    def get(self, url: str) -> str:
        self._wait()
        resp = self.session.get(url, timeout=TIMEOUT_S)
        resp.raise_for_status()
        return resp.text


def _content_root(soup: BeautifulSoup, host: str) -> Tag | None:
    """First selector that matches wins; among its matches take the largest, since
    sites reuse the same class for promo boxes and menus."""
    for selector in CONTENT_SELECTORS.get(host, []):
        if nodes := soup.select(selector):
            return max(nodes, key=lambda n: len(n.get_text(" ", strip=True)))
    return soup.find("main")


def _title(soup: BeautifulSoup) -> str:
    if (h1 := soup.find("h1")) and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)
    return soup.title.get_text(strip=True).split("|")[0].strip() if soup.title else "Untitled"


def _own_text(el: Tag) -> str:
    """Text of `el` excluding nested blocks, which are emitted on their own."""
    if el.name == "div":  # e.g. CFPB lead paragraphs and tips are bare text in a <div>
        strings = el.find_all(string=True, recursive=False)
    else:
        strings = (t for t in el.find_all(string=True) if t.find_parent("li") is el)
    return " ".join(" ".join(strings).split())


def _tidy(text: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


def _row(tr: Tag) -> list[str]:
    cells = tr.find_all(["td", "th"], recursive=False)
    return [_tidy(" ".join(c.get_text(" ", strip=True).split())) for c in cells]


def to_markdown(root: Tag) -> str:
    """Flatten headings, paragraphs, lists, and tables into Markdown, minus site boilerplate."""
    for tag in root.find_all(NOISE):
        tag.decompose()
    lines: list[str] = []
    skipping = False  # inside a boilerplate section such as "Related"
    for el in root.find_all(["h2", "h3", "h4", "p", "li", "tr", "div"]):
        if el.name != "tr" and el.find_parent(["td", "th"]) is not None:
            continue  # table cells are emitted as part of their row
        if el.name == "p" and el.find_parent("li") is not None:
            continue  # the enclosing <li> already covers it
        if el.name == "tr":
            cells = _row(el)
            if skipping or not any(cells):
                continue
            lines.append("| " + " | ".join(cells) + " |")
            if el.find_previous_sibling("tr") is None and el.parent.name != "tbody":
                lines.append("|" + " --- |" * len(cells))  # header separator
            continue
        if el.name in {"li", "div"}:
            text = _own_text(el)
        else:
            text = " ".join(el.get_text(" ", strip=True).split())
        text = _tidy(text)
        if not text or BOILERPLATE_LINE.search(text):
            continue
        if el.name in {"h2", "h3", "h4"}:
            skipping = bool(BOILERPLATE_HEADING.search(text))
            if not skipping:
                lines.append(f"\n## {text}\n")
        elif skipping or (lines and lines[-1].strip() == text):
            continue
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(f"\n{text}\n")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def build_article(url: str, html: str) -> tuple[str, int]:
    """Return (markdown article, body word count)."""
    host = urlparse(url).netloc
    soup = BeautifulSoup(html, "html.parser")
    root = _content_root(soup, host)
    if root is None:
        raise ValueError("no content element found")
    body = to_markdown(root)
    header = (
        f"# {_title(soup)}\n\n"
        f"Source: {url}\n\n"
        f"_Retrieved {date.today().isoformat()} from the {PUBLISHER.get(host, host)}, "
        "a U.S. government work in the public domain. Figures such as contribution "
        "limits change yearly; check the source for current values._\n\n"
    )
    return header + body + "\n", len(body.split())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite existing articles")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    fetcher = Fetcher()
    written = skipped = failed = 0
    for stem, url in SOURCES:
        path = KB_DIR / f"{stem}.md"
        if path.exists() and not args.force:
            skipped += 1
            continue
        try:
            if not fetcher.allowed(url):
                logger.warning("robots.txt disallows %s; skipping", url)
                failed += 1
                continue
            article, words = build_article(url, fetcher.get(url))
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Failed %s: %s", url, exc)
            failed += 1
            continue
        if words < MIN_WORDS:
            logger.warning("Only %d words from %s; check the selector", words, url)
        path.write_text(article)
        logger.info("Wrote %s (%d words)", path.relative_to(ROOT_DIR), words)
        written += 1

    logger.info("Done: %d written, %d skipped (exist), %d failed", written, skipped, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
