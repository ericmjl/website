---
description: Generate and schedule a social media post for a blog post to Buffer in one click
agent: build
---

# Schedule Post: One-Click Social Media Scheduling via Buffer

You are scheduling a social media post for one of Eric's blog posts. You will
generate the post copy with the blogbot scripts and push it to Buffer through
the **buffer** MCP server. Do this end-to-end so the user only has to approve.

## Step 1: Identify the blog post

If `$ARGUMENTS` is a non-empty blog slug, use it. Otherwise, list recent blog
posts by globbing the blog directory and reading each title:

```bash
ls -1 content/blog/
```

Ask the user which one to promote. Then read its metadata to get `title` and
`pub_date`:

```bash
cat content/blog/<slug>/contents.lr
```

## Step 2: Construct the blog post URL

The site is at `https://ericmjl.github.io/`. Blog URLs use the **date-based**
format with NO leading zeros on month or day (see AGENTS.md):

```
https://ericmjl.github.io/blog/<YYYY>/<M>/<d>/<slug>/
```

Derive `<YYYY>/<M>/<d>` from the `pub_date` field in `contents.lr`. For example,
`pub_date: 2024-01-28` becomes `https://ericmjl.github.io/blog/2024/1/28/<slug>/`.

## Step 3: Choose platform(s)

Use the `question` tool to ask the user which platform(s) to post to. The
connected text channels are **LinkedIn** and **BlueSky** (YouTube is a video
channel, skip it for text posts). Allow selecting one or both.

## Step 4: Generate the post copy

Run the matching blogbot script for EACH chosen platform and capture the
generated post text from its output (it is printed inside a rich `Panel`):

- LinkedIn: `uv run .agents/skills/blogbot/scripts/linkedin_post.py <slug>`
- BlueSky: `uv run .agents/skills/blogbot/scripts/bluesky_post.py <slug>`

Then **replace the `[URL]` placeholder** in each generated post with the full
blog post URL from Step 2.

## Step 5: Get user approval

Show the user the final post copy for each platform (with the real URL filled
in). Use the `question` tool to confirm it is good to schedule, or let them
request edits. If they want edits, apply them before continuing.

## Step 6: Choose scheduling mode

Use the `question` tool to ask how to schedule (offer these options):

- **Add to Buffer queue** (Buffer picks the next free slot) — recommended
- **Schedule for a specific time** (ask for date/time, you convert to UTC ISO 8601)
- **Share now**

## Step 7: Push to Buffer via the buffer MCP server

Use the **buffer** MCP server tools to do the following. (Tell the model to
"use the buffer tools" so it routes to that server.)

1. **List channels** with the buffer MCP to find the `channelId` for each chosen
   platform (match on the `service` field: `linkedin` or `bluesky`).
2. **Create the post** for each channel with the buffer MCP. The relevant fields:
   - `text` — the approved post copy (URL already filled in)
   - `channelId` — from the channel list
   - `schedulingType` — `automatic`
   - `mode` — `addToQueue` (queue), `customScheduled` + `dueAt` ISO timestamp
     (specific time), or `shareNow` (now)

   Always handle BOTH the `PostActionSuccess` and `MutationError` response
   shapes so a failure is surfaced clearly instead of ignored.

## Step 8: Confirm

Report back for each platform: the post status, the Buffer post `id`, and the
`dueAt` time it will publish (if scheduled). If anything failed, show the
`MutationError` message and offer to retry.

## Rules

- NEVER publish without showing the user the final copy and getting approval in
  Step 5.
- The `[URL]` placeholder MUST be replaced with the real URL before scheduling.
- If the buffer MCP server is unavailable or returns auth errors, tell the user
  to check that `BUFFER_API_KEY` is set in `.env` and to restart opencode.

$ARGUMENTS
