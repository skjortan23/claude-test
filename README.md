# Daily AI Security News Searcher

Automated daily digest of AI security news, ranked by relevance using Llama 3.3 on Groq (free).

## How It Works

1. **Fetches** articles from GNews API and 11 RSS feeds (security + AI sources)
2. **Ranks** articles by AI security relevance using Llama 3.3 70B on Groq (free tier)
3. **Posts** a daily GitHub Issue with the top 10 ranked stories

Runs daily at 2:00 PM UTC via GitHub Actions.

## Setup

### Required Secrets

Add these in **Settings → Secrets and variables → Actions**:

| Secret | Description |
|--------|-------------|
| `GNEWS_API_KEY` | Free API key from [gnews.io](https://gnews.io/) |
| `GROQ_API_KEY` | Free API key from [console.groq.com](https://console.groq.com/) |

`GITHUB_TOKEN` is provided automatically by GitHub Actions.

### Manual Trigger

Go to **Actions → Daily AI Security News → Run workflow** to trigger manually.

## Adding/Removing RSS Feeds

Edit the `RSS_FEEDS` dictionary in `ai_security_news.py`. Each entry is:

```python
"Source Name": "https://example.com/feed.xml",
```

## Estimated Costs

- **GNews API:** 5 requests/day out of 100 free
- **Groq API:** Free tier (30 req/min, 14,400 req/day)
- **GitHub Actions:** ~1 min/run, well within free tier
