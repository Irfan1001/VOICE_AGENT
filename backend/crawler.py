import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urldefrag, urlunparse

import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

visited = set()
data = []

BASE_URL = "https://example.com"
MAX_PAGES = 300
REQUEST_TIMEOUT = 12
TEXT_LIMIT = 20000
VERIFY_SSL = True
CRAWL_MODE = "browser"


def normalize_url(url: str) -> str:
	clean, _fragment = urldefrag(url)
	parsed = urlparse(clean)
	path = parsed.path.rstrip("/") or "/"
	query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
	return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", query, ""))


def normalize_domain(domain: str) -> str:
	domain = domain.lower().strip()
	if domain.startswith("www."):
		return domain[4:]
	return domain


def same_domain(domain_a: str, domain_b: str) -> bool:
	return normalize_domain(domain_a) == normalize_domain(domain_b)


def should_visit(url: str, base_domain: str) -> bool:
	parsed = urlparse(url)
	if parsed.scheme not in ("http", "https"):
		return False
	if not same_domain(parsed.netloc, base_domain):
		return False
	blocked_suffixes = (
		".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".pdf", ".doc", ".docx",
		".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".rar", ".mp4", ".mp3", ".avi",
	)
	return not parsed.path.lower().endswith(blocked_suffixes)


def fetch_url(url: str):
	return requests.get(
		url,
		timeout=REQUEST_TIMEOUT,
		verify=VERIFY_SSL,
		headers={"User-Agent": "IST-Voice-Agent-Crawler/1.0"},
	)


def save_page(url: str, text: str) -> None:
	clean_text = re.sub(r"\s+", " ", text).strip()
	if not clean_text:
		return
	data.append({"url": url, "text": clean_text[:TEXT_LIMIT]})


def extract_links_from_html(url: str, html: str, base_domain: str):
	soup = BeautifulSoup(html, "html.parser")
	links = set()
	for anchor in soup.find_all("a", href=True):
		link = normalize_url(urljoin(url, anchor["href"]))
		if should_visit(link, base_domain):
			links.add(link)
	return sorted(links)


def parse_sitemap_xml(xml_text: str):
	try:
		root = ET.fromstring(xml_text)
	except ET.ParseError:
		return []

	locs = []
	for loc in root.findall(".//{*}loc"):
		if loc.text:
			locs.append(loc.text.strip())
	return locs


def discover_sitemap_urls(base_url: str):
	base_domain = urlparse(base_url).netloc
	candidates = {urljoin(base_url, "/sitemap.xml")}

	robots_url = urljoin(base_url, "/robots.txt")
	try:
		robots_resp = fetch_url(robots_url)
		if robots_resp.ok:
			for line in robots_resp.text.splitlines():
				if line.lower().startswith("sitemap:"):
					parts = line.split(":", 1)
					if len(parts) == 2:
						candidates.add(parts[1].strip())
	except requests.RequestException:
		pass

	to_visit = list(candidates)
	seen_sitemaps = set()
	page_urls = set()

	while to_visit:
		sitemap_url = normalize_url(to_visit.pop())
		if sitemap_url in seen_sitemaps:
			continue
		seen_sitemaps.add(sitemap_url)

		try:
			resp = fetch_url(sitemap_url)
			resp.raise_for_status()
		except requests.RequestException:
			continue

		locs = parse_sitemap_xml(resp.text)
		if not locs:
			locs = re.findall(r"https?://[^\s\"'<>]+", resp.text)

		for loc in locs:
			loc = normalize_url(loc)
			parsed = urlparse(loc)
			if not same_domain(parsed.netloc, base_domain):
				continue
			if loc.lower().endswith(".xml") or "sitemap" in loc.lower():
				if loc not in seen_sitemaps:
					to_visit.append(loc)
			elif should_visit(loc, base_domain):
				page_urls.add(loc)

	return sorted(page_urls)


def crawl_requests(base_url: str, base_domain: str) -> None:
	queue = deque([base_url])
	for sitemap_url in discover_sitemap_urls(base_url):
		queue.append(sitemap_url)

	while queue and len(visited) < MAX_PAGES:
		url = normalize_url(queue.popleft())
		if url in visited or not should_visit(url, base_domain):
			continue
		visited.add(url)

		try:
			response = fetch_url(url)
			response.raise_for_status()
		except requests.RequestException:
			continue

		content_type = response.headers.get("Content-Type", "")
		if "text/html" not in content_type.lower():
			continue

		save_page(url, BeautifulSoup(response.text, "html.parser").get_text(separator=" ", strip=True))

		for link in extract_links_from_html(url, response.text, base_domain):
			if link not in visited:
				queue.append(link)


def crawl_browser(base_url: str, base_domain: str) -> None:
	try:
		from playwright.sync_api import Error as PlaywrightError
		from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
		from playwright.sync_api import sync_playwright
	except ImportError as exc:
		raise RuntimeError(
			"Playwright is not installed. Install it with 'pip install playwright' and run 'python -m playwright install chromium'."
		) from exc

	seed_urls = [base_url]
	seed_urls.extend(discover_sitemap_urls(base_url))
	queue = deque(normalize_url(url) for url in seed_urls)

	with sync_playwright() as playwright:
		browser = playwright.chromium.launch(headless=True)
		context = browser.new_context(ignore_https_errors=not VERIFY_SSL)
		page = context.new_page()

		while queue and len(visited) < MAX_PAGES:
			url = normalize_url(queue.popleft())
			if url in visited or not should_visit(url, base_domain):
				continue

			visited.add(url)

			try:
				page.goto(url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT * 3000)
				try:
					page.wait_for_load_state("networkidle", timeout=5000)
				except PlaywrightTimeoutError:
					pass
				page.wait_for_timeout(2000)
			except (PlaywrightTimeoutError, PlaywrightError):
				continue

			try:
				text = page.locator("body").inner_text(timeout=5000)
			except (PlaywrightTimeoutError, PlaywrightError):
				text = ""

			save_page(url, text)

			try:
				hrefs = page.eval_on_selector_all(
					"a[href]",
					"elements => elements.map(element => element.href)",
				)
			except PlaywrightError:
				hrefs = []

			for href in hrefs:
				link = normalize_url(href)
				if should_visit(link, base_domain) and link not in visited:
					queue.append(link)

		context.close()
		browser.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Crawl a site and save plain text pages.")
	parser.add_argument("--base-url", default=BASE_URL, help="Root URL to start crawling from")
	parser.add_argument("--max-pages", type=int, default=MAX_PAGES, help="Maximum number of pages")
	parser.add_argument("--text-limit", type=int, default=TEXT_LIMIT, help="Maximum stored characters per page")
	parser.add_argument("--mode", choices=["browser", "requests"], default=CRAWL_MODE, help="Crawler mode")
	parser.add_argument("--insecure", action="store_true", help="Disable SSL verification")
	args = parser.parse_args()

	VERIFY_SSL = not args.insecure
	if not VERIFY_SSL:
		requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

	MAX_PAGES = args.max_pages
	TEXT_LIMIT = args.text_limit
	base_url = normalize_url(args.base_url)
	base_domain = urlparse(base_url).netloc

	if args.mode == "browser":
		crawl_browser(base_url, base_domain)
	else:
		crawl_requests(base_url, base_domain)

	output_path = Path(__file__).resolve().parents[1] / "data" / "texts.json"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False)

	print("Crawled:", len(data), "pages")
