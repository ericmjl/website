---
name: blogbot
description: Generate social media posts and blog content from local blog posts, and schedule posts to Buffer via the buffer MCP server. Use when asked to create LinkedIn posts, BlueSky posts, Substack posts, summaries, tags, or DALL-E banner images for blog posts, or to schedule/share a blog post to social media through Buffer. The banner is automatically saved as logo.webp in the blog post directory. Works with blog post slugs from content/blog/. Triggers: "generate a linkedin post", "create a bluesky post", "substack post for", "summarize this blog", "get tags for blog", "create a banner for", "schedule a post", "share my blog post", "add to buffer".
---

# Blogbot

Generate social media posts and blog content from your local blog posts, and
schedule them to Buffer.

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

**Substack Post** - Generates a Substack post (single title, no A/B variants):
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

## CRITICAL: Run Scripts Sequentially, Never in Parallel

All blogbot generation scripts (linkedin_post.py, bluesky_post.py,
substack_post.py, summary.py, tags.py) share a single llamabot sqlite
database that enforces UNIQUE constraints on `prompts.hash`. Running two
or more concurrently causes `sqlite3.IntegrityError` (UNIQUE constraint
failed: prompts.hash) because both processes try to insert the same
prompt at once.

This is an **exception to the general AGENTS.md rule to prefer parallel
subagents.** When generating posts for one or more blog posts, run each
script one after the other, not in a parallel batch. If one fails due to
a race, simply re-run it sequentially after the others complete (the
retry succeeds because there is no longer contention).

## One-Click Scheduling to Buffer

The **buffer** MCP server (configured in `opencode.json`) connects opencode to
your Buffer account. It exposes tools to list channels, create posts, and
schedule them. The `/schedule-post` command ties blogbot generation together
with Buffer scheduling in one shot.

**One-click button:**

```
/schedule-post <blog_slug>
```

This command:

1. Reads the blog post and builds its public URL
   (`https://ericmjl.github.io/blog/YYYY/M/d/slug/`, no leading zeros on
   month/day, derived from `pub_date`).
2. Asks which platform(s) to post to (LinkedIn, BlueSky).
3. Generates the post copy with the matching blogbot script and fills in the
   `[URL]` placeholder with the real URL.
4. Shows you the copy for approval.
5. Asks whether to queue, schedule at a specific time, or share now.
6. Pushes the post to Buffer through the buffer MCP server.

When scheduling manually (without the command), use the buffer MCP `create_post`
tool. It wraps Buffer's `createPost` GraphQL mutation, whose input takes `text`,
`channelId`, `schedulingType: automatic`, and a `mode` of `addToQueue`,
`customScheduled` (+ `dueAt` ISO timestamp), or `shareNow`. Handle both the
`PostActionSuccess` and `MutationError` response shapes.

## Script Structure

Each script is a standalone PEP-723 inline-metadata script (blog-scraping
helpers are duplicated per script so each can run independently with `uv run`).

- `scripts/linkedin_post.py` - LinkedIn post generator
- `scripts/bluesky_post.py` - BlueSky post generator
- `scripts/substack_post.py` - Substack post generator
- `scripts/summary.py` - Summary generator
- `scripts/tags.py` - Tag generator
- `scripts/banner.py` - DALL-E banner generator
