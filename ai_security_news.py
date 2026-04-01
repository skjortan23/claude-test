#!/usr/bin/env python3
"""Daily AI Security News Searcher — fetches, ranks, and posts a digest."""

import datetime
import html
import json
import os
import re
import sys
import time

import feedparser
import requests

# --- Configuration ---

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_FAST = "meta-llama/llama-4-scout-17b-16e-instruct"  # 30K TPM, for ranking
GROQ_MODEL_CREATIVE = "llama-3.3-70b-versatile"  # 12K TPM, for witty summaries

GNEWS_API_URL = "https://gnews.io/api/v4/search"

SEARCH_QUERIES = [
    "AI security",
    "artificial intelligence cybersecurity",
    "LLM vulnerability",
    "AI safety threat",
    "machine learning attack",
]

RSS_FEEDS = {
    "The Hacker News": "https://feeds.feedburner.com/TheHackersNews",
    "BleepingComputer": "https://www.bleepingcomputer.com/feed/",
    "Krebs on Security": "https://krebsonsecurity.com/feed/",
    "Dark Reading": "https://www.darkreading.com/rss.xml",
    "SecurityWeek": "https://feeds.feedburner.com/securityweek",
    "Schneier on Security": "https://www.schneier.com/feed/atom/",
    "MIT Tech Review AI": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "The Register Security": "https://www.theregister.com/security/headlines.atom",
    "Google Security Blog": "https://security.googleblog.com/feeds/posts/default",
    "OpenAI Blog": "https://openai.com/blog/rss.xml",
}

RANKING_PROMPT = """\
You are an AI security analyst. Evaluate the following news articles and rank them \
by relevance to AI security — the intersection of artificial intelligence and \
cybersecurity/safety.

Highly relevant topics include:
- Vulnerabilities in AI/ML systems (prompt injection, model poisoning, adversarial attacks)
- AI-powered cyber threats (AI-generated malware, deepfakes for social engineering)
- AI safety and alignment research with security implications
- Regulatory and policy developments for AI security
- Security of AI infrastructure (model supply chain, training data security)
- Defensive uses of AI in cybersecurity

Less relevant (score low):
- General cybersecurity news with no AI angle
- General AI news with no security angle
- Product announcements without security relevance

Here are today's articles:

{articles}

Return a JSON array ranking the top 10 most relevant articles. Format:
[
  {{
    "rank": 1,
    "article_number": 5,
    "relevance_score": 9,
    "explanation": "One sentence explaining why this is relevant to AI security."
  }}
]

IMPORTANT: Many articles may cover the same story from different sources. When multiple \
articles cover the same topic/event, only include the SINGLE best one (most detailed or \
highest quality source). The final list should have 10 DISTINCT stories, not 10 articles \
about 3 stories.

Only include articles scoring 4 or above. If fewer than 10 qualify, return fewer.
Return ONLY the JSON array, no other text."""

WITTY_SUMMARY_PROMPT = """\
You are a tech journalist with the sardonic wit of The Register and the irreverent \
snark of old-school Slashdot editors. Write a short, punchy summary (2-3 paragraphs) \
for each of the following AI security news articles. Your tone should be:

- Tongue-in-cheek and slightly world-weary, like you've seen it all before
- Sharp, opinionated, and entertaining — don't be afraid to editorialize
- Technically accurate but accessible — no jargon soup
- Think "seasoned hack who's had too much coffee and not enough patience for PR fluff"

For each article, write a witty summary that captures the key facts while making \
the reader smirk. Include a snarky subheadline for each one.

Articles:

{articles}

Return a JSON array with your summaries. Format:
[
  {{
    "article_number": 1,
    "subheadline": "A short, snarky subheadline",
    "witty_summary": "Your 2-3 paragraph witty summary here."
  }}
]

Return ONLY the JSON array, no other text."""


# --- LLM ---


def _call_groq(prompt: str, model: str = GROQ_MODEL_FAST, max_tokens: int = 4096) -> str:
    """Call Groq API with retry on rate limit using retry-after header."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")

    for attempt in range(3):
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            timeout=60,
        )
        if resp.status_code == 429:
            wait = int(resp.headers.get("retry-after", 15))
            print(f"Groq rate limited ({model}), waiting {wait}s (attempt {attempt + 1}/3)...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    resp.raise_for_status()
    return ""


def _parse_json_response(text: str) -> list:
    """Parse JSON from LLM response, handling markdown fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```json?\s*(.*?)```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError("Could not parse JSON from LLM response.")


# --- Fetching ---


def fetch_news_api_articles() -> list[dict]:
    """Fetch articles from GNews API."""
    api_key = os.environ.get("GNEWS_API_KEY", "")
    if not api_key:
        print("Warning: GNEWS_API_KEY not set, skipping News API.")
        return []

    articles = []
    seen_urls = set()

    for i, query in enumerate(SEARCH_QUERIES):
        if i > 0:
            time.sleep(1)  # avoid rate limits
        try:
            resp = requests.get(
                GNEWS_API_URL,
                params={"q": query, "lang": "en", "max": 10, "token": api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("articles", []):
                url = item.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                articles.append({
                    "title": item.get("title", ""),
                    "summary": (item.get("description") or "")[:500],
                    "url": url,
                    "source": item.get("source", {}).get("name", "Unknown"),
                    "published": item.get("publishedAt", ""),
                    "origin": "newsapi",
                })
        except Exception as e:
            print(f"Warning: GNews query '{query}' failed: {e}")

    print(f"Fetched {len(articles)} articles from GNews API.")
    return articles


def fetch_rss_articles() -> list[dict]:
    """Fetch articles from RSS feeds, filtered to last 24 hours."""
    cutoff = time.time() - 86400  # 24 hours ago
    articles = []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                # Check publication date
                published_parsed = getattr(entry, "published_parsed", None) or getattr(
                    entry, "updated_parsed", None
                )
                if published_parsed:
                    entry_time = time.mktime(published_parsed)
                    if entry_time < cutoff:
                        continue

                # Extract and clean summary
                raw_summary = getattr(entry, "summary", "") or getattr(
                    entry, "description", ""
                )
                clean_summary = html.unescape(re.sub(r"<[^>]+>", "", raw_summary))[:500]

                articles.append({
                    "title": getattr(entry, "title", "No title"),
                    "summary": clean_summary,
                    "url": getattr(entry, "link", ""),
                    "source": source_name,
                    "published": getattr(entry, "published", ""),
                    "origin": "rss",
                })
        except Exception as e:
            print(f"Warning: RSS feed '{source_name}' failed: {e}")

    print(f"Fetched {len(articles)} articles from RSS feeds.")
    return articles


# --- Deduplication ---


def _normalize_url(url: str) -> str:
    """Normalize URL for deduplication."""
    url = url.rstrip("/")
    url = url.split("?")[0]
    url = url.split("#")[0]
    return url.lower()


def deduplicate(articles: list[dict]) -> list[dict]:
    """Remove duplicate articles by normalized URL."""
    seen = set()
    unique = []
    for article in articles:
        normalized = _normalize_url(article["url"])
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(article)
    return unique


def _title_keywords(title: str) -> set[str]:
    """Extract significant keywords from a title for similarity matching."""
    stop_words = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
        "is", "its", "it", "by", "with", "from", "as", "how", "why", "what",
        "new", "says", "could", "may", "will", "just", "has", "had", "are",
        "was", "been", "be", "do", "does", "did", "that", "this", "but",
    }
    words = set(re.sub(r"[^a-z0-9\s]", "", title.lower()).split())
    return words - stop_words


def deduplicate_by_topic(articles: list[dict]) -> list[dict]:
    """Remove articles covering the same story based on title similarity."""
    unique = []
    seen_keyword_sets: list[set[str]] = []

    for article in articles:
        keywords = _title_keywords(article["title"])
        if len(keywords) < 2:
            unique.append(article)
            seen_keyword_sets.append(keywords)
            continue

        is_duplicate = False
        for seen_kw in seen_keyword_sets:
            if len(seen_kw) < 2:
                continue
            overlap = len(keywords & seen_kw)
            similarity = overlap / min(len(keywords), len(seen_kw))
            if similarity >= 0.6:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(article)
            seen_keyword_sets.append(keywords)

    removed = len(articles) - len(unique)
    if removed:
        print(f"Topic dedup removed {removed} duplicate stories.")
    return unique


# --- Ranking ---


def rank_articles(articles: list[dict]) -> list[dict]:
    """Rank articles by AI security relevance using Groq."""
    if not articles:
        return []

    # Cap at 30 articles to stay within token limits
    articles_to_rank = articles[:30]

    # Build numbered list
    numbered_list = "\n".join(
        f"{i + 1}. [{a['title']}] — {a['summary']} (Source: {a['source']})"
        for i, a in enumerate(articles_to_rank)
    )

    prompt = RANKING_PROMPT.format(articles=numbered_list)

    try:
        response_text = _call_groq(prompt, model=GROQ_MODEL_FAST)
        rankings = _parse_json_response(response_text)

        # Merge rankings back into articles
        ranked = []
        for entry in sorted(rankings, key=lambda x: x.get("rank", 99)):
            idx = entry.get("article_number", 0) - 1
            if 0 <= idx < len(articles_to_rank):
                article = articles_to_rank[idx].copy()
                article["relevance_score"] = entry.get("relevance_score", 0)
                article["explanation"] = entry.get("explanation", "")
                article["rank"] = entry.get("rank", 0)
                ranked.append(article)

        print(f"Ranked {len(ranked)} articles via Groq ({GROQ_MODEL_FAST}).")
        return ranked

    except Exception as e:
        print(f"Warning: Ranking failed: {e}")
        print("Falling back to unranked results (most recent first).")
        for i, article in enumerate(articles_to_rank[:10]):
            article["relevance_score"] = 0
            article["explanation"] = "Ranking unavailable."
            article["rank"] = i + 1
        return articles_to_rank[:10]


# --- Witty Summaries ---


def generate_witty_summaries(top_articles: list[dict]) -> list[dict]:
    """Generate tongue-in-cheek summaries for the top 3 articles."""
    if not top_articles:
        return top_articles

    articles_text = "\n\n".join(
        f"{i + 1}. Title: {a['title']}\n   Source: {a['source']}\n   Summary: {a['summary']}"
        for i, a in enumerate(top_articles[:3])
    )

    prompt = WITTY_SUMMARY_PROMPT.format(articles=articles_text)

    try:
        response_text = _call_groq(prompt, model=GROQ_MODEL_CREATIVE)
        summaries = _parse_json_response(response_text)

        for entry in summaries:
            idx = entry.get("article_number", 0) - 1
            if 0 <= idx < len(top_articles):
                top_articles[idx]["subheadline"] = entry.get("subheadline", "")
                top_articles[idx]["witty_summary"] = entry.get("witty_summary", "")

        print(f"Generated witty summaries for {len(summaries)} articles.")

    except Exception as e:
        print(f"Warning: Witty summary generation failed: {e}")

    return top_articles


# --- Output ---


def create_github_issue(ranked_articles: list[dict]) -> None:
    """Create a GitHub issue with the daily digest."""
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not token or not repo:
        print("Error: GITHUB_TOKEN or GITHUB_REPOSITORY not set.")
        sys.exit(1)

    today = datetime.date.today().isoformat()
    title = f"AI Security News Digest — {today}"

    # Check if today's issue already exists
    try:
        check_resp = requests.get(
            f"https://api.github.com/repos/{repo}/issues",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            params={"labels": "daily-digest", "state": "open", "per_page": 5},
            timeout=10,
        )
        if check_resp.ok:
            for issue in check_resp.json():
                if today in issue.get("title", ""):
                    print(f"Issue for {today} already exists: {issue['html_url']}")
                    return
    except Exception:
        pass  # Proceed with creation even if check fails

    # Build issue body
    body_lines = [
        f"# AI Security News Digest — {today}\n",
        "> Automatically generated daily digest of AI security news, ranked by relevance using Llama on Groq.\n",
    ]

    # Featured top 3 with witty summaries
    top_3 = [a for a in ranked_articles if a.get("witty_summary")]
    if top_3:
        body_lines.append("## Featured Stories\n")
        for article in top_3[:3]:
            score = article.get("relevance_score", "N/A")
            subheadline = article.get("subheadline", "")
            body_lines.append(f"### {article['rank']}. [{article['title']}]({article['url']})")
            if subheadline:
                body_lines.append(f"*{subheadline}*\n")
            body_lines.append(f"**Source:** {article['source']} | **Relevance:** {score}/10\n")
            body_lines.append(article["witty_summary"])
            body_lines.append("\n---\n")

    # Remaining articles
    remaining = [a for a in ranked_articles if not a.get("witty_summary")]
    if remaining:
        body_lines.append("## Also Noteworthy\n")
        for article in remaining:
            score = article.get("relevance_score", "N/A")
            explanation = article.get("explanation", "")
            body_lines.append(f"### {article['rank']}. [{article['title']}]({article['url']})")
            body_lines.append(f"**Source:** {article['source']} | **Relevance:** {score}/10")
            if explanation:
                body_lines.append(f"> {explanation}")
            if article.get("summary"):
                body_lines.append(f"\n{article['summary'][:200]}...")
            body_lines.append("\n---\n")

    body_lines.append("## Methodology")
    body_lines.append(f"- **News sources:** GNews API + {len(RSS_FEEDS)} RSS feeds")
    body_lines.append(f"- **Ranking model:** {GROQ_MODEL_FAST} (via Groq)")
    body_lines.append(f"- **Summary model:** {GROQ_MODEL_CREATIVE} (via Groq)")
    body_lines.append(f"- **Generated:** {datetime.datetime.utcnow().isoformat()}Z")

    body = "\n".join(body_lines)

    resp = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        json={
            "title": title,
            "body": body,
            "labels": ["ai-security", "daily-digest"],
        },
        timeout=15,
    )
    resp.raise_for_status()
    print(f"Created issue: {resp.json()['html_url']}")


# --- Main ---


def main():
    news_articles = fetch_news_api_articles()
    rss_articles = fetch_rss_articles()

    all_articles = deduplicate(news_articles + rss_articles)
    print(f"Unique articles by URL: {len(all_articles)}")
    all_articles = deduplicate_by_topic(all_articles)
    print(f"Unique articles by topic: {len(all_articles)}")

    if not all_articles:
        print("No articles found. Exiting.")
        return

    ranked = rank_articles(all_articles)
    time.sleep(10)  # cooldown to avoid Groq TPM rate limit
    ranked = generate_witty_summaries(ranked)
    create_github_issue(ranked)


if __name__ == "__main__":
    main()
