---
name: blogbot
description: Generate social media posts and blog content from local blog posts. Use when asked to create LinkedIn posts, BlueSky posts, Substack posts, summaries, tags, or DALL-E banner images for blog posts. The banner is automatically saved as logo.webp in the blog post directory. Works with blog post slugs from content/blog/. Triggers: "generate a linkedin post", "create a bluesky post", "substack post for", "summarize this blog", "get tags for blog", "create a banner for".
---

# Blogbot

Generate social media posts and blog content from your local blog posts.

## Available Scripts

### Social Media Posts

**LinkedIn Post** - Generates a structured LinkedIn post with hook, authority elements, main content, CTA, and hashtags:
```bash
uv run .agents/skills/blogbot/scripts/linkedin_post.py <blog_slug>
```

**BlueSky Post** - Generates a concise BlueSky post (< 283 chars) optimized for engagement:
```bash
uv run .agents/skills/blogbot/scripts/bluesky_post.py <blog_slug>
```

**Substack Post** - Generates a Substack post with title variants for A/B testing:
```bash
uv run .agents/skills/blogbot/scripts/substack_post.py <blog_slug>
```

### Blog Content

**Summary** - Generates a 100-word summary in first person:
```bash
uv run .agents/skills/blogbot/scripts/summary.py <blog_slug>
```

**Tags** - Generates 10 tags (max 2 words each, 7+ one-word tags):
```bash
uv run .agents/skills/blogbot/scripts/tags.py <blog_slug>
```

**Banner** - Generates a DALL-E banner image and saves it as `logo.webp` in the blog post directory:
```bash
uv run .agents/skills/blogbot/scripts/banner.py <blog_slug>
```

## Usage Examples

```bash
# Generate a LinkedIn post for a blog post
uv run .agents/skills/blogbot/scripts/linkedin_post.py a-practical-guide-to-securing-secrets-in-data-science-projects

# Get tags for a blog post
uv run .agents/skills/blogbot/scripts/tags.py a-practical-guide-to-securing-secrets-in-data-science-projects

# Create a banner image
uv run .agents/skills/blogbot/scripts/banner.py a-practical-guide-to-securing-secrets-in-data-science-projects
```

Run any script without arguments to see a list of available blog posts.

## Script Structure

- `scripts/models.py` - Pydantic models for structured output
- `scripts/prompts.py` - LLM prompt templates
- `scripts/scraper.py` - Local blog post reading utilities
- `scripts/linkedin_post.py` - LinkedIn post generator
- `scripts/bluesky_post.py` - BlueSky post generator
- `scripts/substack_post.py` - Substack post generator
- `scripts/summary.py` - Summary generator
- `scripts/tags.py` - Tag generator
- `scripts/banner.py` - DALL-E banner generator
