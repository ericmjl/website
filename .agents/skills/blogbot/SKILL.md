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

> **Scrub the output before writing it into `contents.lr`.** `summary.py`'s
> prompt historically mandated a `[URL]` placeholder and a "read on"
> enticement (both now removed from the script). If you ever regenerate a
> summary and see either, strip them:
> - **No `[URL]` placeholder** — the `contents.lr` `summary:` field is the
>   post's own meta-description (shown on the blog index / for SEO), so it
>   is self-contained with no link to insert.
> - **No "Read on!" / "Read on to find out!" / "read more" tail** — the HTML
>   template (`templates/macros/blog.html`, lines ~238/244) already renders
>   a `Read on...` / `(read more)` link immediately after the summary. Any
>   such tail duplicates it. End the summary at the question instead.
>
> This mirrors the em-dash scrubbing rule (see AGENTS.md: no em dashes).

**Tags** - Generates 10 tags (max 2 words each, 7+ one-word tags):
```bash
uv run .agents/skills/blogbot/scripts/tags.py <blog_slug>
```

> **Verify tags for typos before writing them into `contents.lr`.** LLMs
> produce typos in short structured outputs like tags — duplicated letters
> (e.g. "aiaagents" instead of "ai agents"), missing spaces, or malformed
> compound words. Read every generated tag, confirm it is a real word or
> phrase, and fix or drop any typo before persisting. This is the tag-form
> sibling of the summary scrubbing rule above.

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

## Audit Before Queueing (decide what to schedule next)

Before generating or queueing NEW social posts, FIRST audit Buffer's current
state to ground the decision in what is already scheduled/sent. This is a
PRE-scheduling AUDIT and is distinct from the POST-scheduling VERIFICATION
below (which checks for collisions AFTER a batch is queued).

- Pre-audit answers: "What gaps exist? What should I queue next?"
- Post-verify answers: "Did I schedule correctly? Any same-day collisions?"

Workflow (observed 2026-07-19: "query buffer for what latest social media
posts we have made. I want to know what we need to queue up"):

1. `list_channels` for the organization to get channel IDs (LinkedIn,
   BlueSky, etc.).
2. `list_posts` on each channel with a date filter covering recent sent +
   upcoming scheduled posts (e.g. last 30 days through next 30 days).
3. Map the results to the blog cadence: which weeks already have a post,
   which are free, and which blog slugs have NO social promo yet.
4. PROPOSE what to queue next based on the gaps. Do not blindly generate
   posts for the most recent blog slug without confirming it is not already
   scheduled.

This is the data-driven entry point to the scheduling workflow. The
one-post-per-week cadence and cross-channel sync rules (below) then govern
HOW the proposed posts are dated, not WHETHER to propose them.

Connects to: pub_date is coupled to the post URL via Lektor slug_format,
and already-scheduled Buffer posts embed that URL, so the audit also
surfaces any post whose pub_date has since moved and whose Buffer links
would now be stale.

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

## Cross-Channel Scheduling Sync (one post per week)

- Each BLOG POST shares ONE release date across ALL channels (LinkedIn, BlueSky, etc.). The same post goes out on the same day everywhere; never give a single post different days per channel.
- Stagger posts ONE PER WEEK. N posts span N consecutive weeks (e.g. 3 posts -> July 2, July 9, July 16). Two posts landing on the same calendar day is wrong even across different channels, because the blog cadence is weekly.
- BlueSky is the source of truth. When channel schedules disagree, reconcile every channel to the BlueSky per-post dates. (Observed 2026-06-17: LinkedIn had two posts batched on July 2 while BlueSky staggered them; the fix was to realign LinkedIn's dates to BlueSky's.)
- VERIFY before finishing a scheduling job: call list_posts on each channel, extract dueAt per blog slug, and confirm (a) every slug has the SAME date on every channel, and (b) no two distinct slugs share a date. Treat this post-schedule sync check as mandatory, just like the URL-verification pre-check. This is exactly the kind of cross-channel drift that is invisible until the user reads the Buffer calendar.
- YouTube videos (uploaded DIRECTLY to YouTube, not via Buffer's Shorts-only integration) MUST be released on the same date the corresponding blog post is scheduled on Buffer. Buffer is the canonical publication date for the whole content bundle. If a video is set to go live earlier or later than the Buffer post, reschedule the VIDEO to match Buffer, never the other way around. Before uploading/scheduling a YouTube video for a blog post, call list_posts on Buffer, read the post's dueAt, and align the video's publish time to it. (Observed 2026-06-25: user stated "my youtube videos need to be released in sync with the buffer posts" after a video was set to publish ahead of its Buffer-scheduled blog post.)

## YouTube via Buffer (Shorts-only, portrait required)

Buffer's YouTube channel integration only accepts vertical (portrait) videos
and treats every upload as a YouTube Short. There is no `type` field in the
`metadata.youtube` schema of the buffer `create_post` tool to override this:
Buffer infers Short-vs-regular from dimensions and rejects anything that is
not 9:16.

Constraint:
- Submitting a landscape video (e.g. 1920x1080) fails with: "Video must be
  vertical (portrait orientation) for YouTube Shorts."
- Required aspect ratio: 9:16 portrait (e.g. 1080x1920).

Workflow when scheduling a Remotion-rendered video to YouTube via Buffer:
1. Confirm the composition is portrait (1080x1920). If only a landscape
   composition exists, register a portrait variant in `Root.tsx` and render it
   before scheduling.
2. Use a direct-download URL for the video asset (e.g. tmpfiles.org
   `https://tmpfiles.org/dl/<id>/<filename>`, not the `/w-` preview page).
   Buffer fetches and re-hosts the video at post-creation time, so a
   short-lived URL is acceptable as long as it is reachable when `create_post`
   runs.
3. Always pass `metadata.youtube.title` and `metadata.youtube.categoryId`
   (both required by the schema).

Reference: the `blog-video` skill already targets 9:16 vertical; this note
explains WHY portrait is mandatory for Buffer distribution.

## Rescheduling Existing Buffer Posts

- When asked to move an already-scheduled post to a new date (e.g. two posts landed on the same day and need spreading out), use buffer edit_post — NOT create_post/delete+recreate. Critical: edit_post RE-VALIDATES THE WHOLE POST and does NOT merge with stored data, so you must carry every field forward from get_post and change only what the user asked to change.

Workflow:
1. list_posts (or get_post) to read the current post: text, assets, metadata, schedulingType, shareMode.
2. Carry ALL of those forward unchanged. Map stored assets to edit_post shape (use source->url, thumbnail->thumbnailUrl).
3. Change ONLY: mode='customScheduled' + dueAt=<new ISO 8601 datetime with tz offset>. Keep schedulingType from get_post.
4. If get_post returns schedulingType: null (common), pass schedulingType='automatic' — it matches auto-publish channels and edit_post requires the field.
5. Pass text and metadata verbatim. Dropping a required metadata field (e.g. instagram.type) will reject the edit.

Gotcha: do NOT trust a reference schedule blindly. If the user says 'follow the Bluesky schedule', verify that schedule does not ALSO have same-day collisions before mirroring it. Compute the ideal cadence independently.

## Scheduling Cadence

- Default cadence: ONE post per week per channel, spread evenly across the publishing window. When scheduling a batch of blog posts to LinkedIn/Bluesky, do NOT stack two posts on the same date even if the queue allows it. Stagger them weekly (e.g. week 1, week 2, week 3...). Before finishing a scheduling batch, verify each channel has no same-day duplicates. Keep LinkedIn and Bluesky in sync (same post on the same week) unless told otherwise.

- **Use `addToQueue`, NOT `customScheduled`, for blog post scheduling.** The channel's queue already has the correct time slots configured (e.g. 7 AM). `addToQueue` places the post in the next available weekly slot automatically. Do NOT hardcode a `dueAt` time. Do NOT use `customScheduled` unless the user EXPLICITLY asks for a specific time ("post at 3 PM", "schedule for Tuesday morning"). Observed 2026-07-19: the agent hardcoded `dueAt: 11:00 AM EDT` via `customScheduled` instead of using `addToQueue`, which placed posts at 11 AM instead of the queue's 7 AM slot. The user had to manually fix the times. The root cause: the agent saw inconsistent times in the existing schedule (one `addToQueue` post at 7 AM, one `customScheduled` post at 11 AM) and "picked a consistent time" instead of trusting the queue. Trust the queue. Pass `mode: "addToQueue"` + `schedulingType: "automatic"` with no `dueAt`.

## Post-Merge GitHub Pages Rebuild Delay

After a blog post PR is merged, the public URL
(`https://ericmjl.github.io/blog/YYYY/M/d/slug/`) returns **404 for 1-5
minutes** while GitHub Pages rebuilds the site. The URL-verification
pre-check (confirm HTTP 200 against the LIVE deployed site before
scheduling) WILL fail during this window.

This is a **recurring, expected** state — not an error. Do not treat the
404 as a broken URL or a reason to abort scheduling.

**Workflow when the URL is 404 immediately after merge:**

1. Generate the social copy FIRST (linkedin_post.py, bluesky_post.py).
   These scripts read the local `contents.lr` and do NOT need the live
   URL. This uses the wait productively.
2. Determine the target scheduling date (next free weekly slot — see
   Scheduling Cadence) by calling `list_posts` on the target channels.
3. Re-check the URL with a curl/HTTP probe every ~60 seconds. GitHub
   Pages typically finishes within 2-3 minutes of the merge.
4. Once the URL returns HTTP 200, proceed to `create_post` on each
   channel with the verified URL.

**Do NOT:**
- Queue a post whose URL has not been verified as HTTP 200 against the
  live site (the hard rule still holds — wait it out).
- Abort the whole scheduling task because the URL is 404 for the first
  minute — the site just hasn't rebuilt yet.
- Burn turns re-checking in a tight loop; generate copy and find the
  target date in parallel, then re-check at ~60s intervals.

## Em-Dash Rule Extends to All Generated Content

AGENTS.md states: "I do not use em dashes (—); use commas, periods, or
separate sentences instead." This is Eric's voice rule and applies to
**ALL generated content**, not just blog post bodies and summaries:

- LinkedIn posts (linkedin_post.py output)
- BlueSky posts (bluesky_post.py output)
- Substack posts (substack_post.py output)
- Summaries (summary.py output — already documented above)

LLMs frequently emit em dashes (U+2014) in social copy even when the
prompt says not to. After generating ANY social copy, scan the output
for em dashes (—, \u2014) and replace each with a comma, period, colon,
or separate sentence before scheduling or showing it for approval. Treat
em-dash scrubbing as a mandatory post-generation step for every blogbot
text output, alongside URL-verification and tag-typo-checking.

## Publishing to Substack via Browser

Substack is NOT a Buffer channel, so publishing a Substack post is a
browser-automation task (use the agent-browser skill), not a buffer create_post
call. The blogbot `substack_post.py` script only GENERATES the copy; it does
not publish. The conversation agent (glm-5.2) is the PREFERRED path for
writing Substack post bodies; the script is a fallback tool.

SUBSTACK POST TYPES — know the difference:
- "notes" = short-form (tweet-like, shown in the Substack feed). The "New post"
  button in the Substack nav opens a NOTE composer, NOT the long-form editor.
  This is a trap: clicking "New post" drops you into a note dialog when you
  want a full article.
- "posts" = long-form articles (what substack_post.py generates).

DIRECT URL FOR THE LONG-FORM EDITOR:
Navigate directly to `https://{publication}.substack.com/publish/post` instead
of clicking the "New post" button. This bypasses the note-composer trap. You
need the publication subdomain (e.g. `ericma.substack.com`); ask the user if
you do not know it.

LOGIN-STATE DETECTION:
- "Start your Substack" button visible = not logged in as a publication owner.
  Ask the user to sign in.
- "Dashboard" / "Profile" visible = logged in and ready to publish.

WORKFLOW:
1. Generate the Substack post copy:
   `uv run .agents/skills/blogbot/scripts/substack_post.py <blog_slug>`
2. Confirm the user is logged into Substack in the debug-Chrome window (check
   for Dashboard/Profile).
3. Navigate directly to `https://{publication}.substack.com/publish/post` (do
   NOT click "New post").
4. Fill the post in this EXACT mandatory order. Skipping the banner or the
   greeting is a recurring error the user flags every time ("like I always
   do"); follow the order literally:
   a. Title (from substack_post.py output).
   b. Subtitle (from substack_post.py output).
   c. Banner image (logo.webp) at the TOP of the body, BEFORE any text.
      ALWAYS included, never optional. Insert via the insertHTML `<img>`
      technique in the section below.
   d. Greeting line on its own: "Hello fellow datanistas,"
   e. The composed Substack post body, where the phrase "this post" is
      hyperlinked to the blog post public URL
      (https://ericmjl.github.io/blog/YYYY/M/d/slug/, no leading zeros).
   f. Sign-off: "Happy coding,\nEric" (or a short theme variant).
5. VERIFY before saving/publishing (mandatory, same status as the
   cross-channel sync check): take an agent-browser snapshot of the editor
   and confirm ALL of: (1) banner image RENDERED at the top (not a broken
   or missing node), (2) greeting line present, (3) "this post" is a live
   hyperlink. The insertHTML path can silently drop the `<img>` because the
   ProseMirror schema may reject it, so a visual check is required. Do NOT
   declare done from the command merely having run.
6. Save as draft or publish per the user's instruction.

EDITOR CONTENT-INSERTION TECHNIQUE (CONFIRMED WORKING 06-29):
The Substack body is a ProseMirror (schema-driven) editor. Do NOT type the
post character-by-character via `agent-browser type` for long-form posts,
it is prohibitively slow. Do NOT mutate the DOM directly, ProseMirror enforces
its own document model and ignores injected nodes (they vanish on the next
render). Instead, after focusing the editor, inject HTML via the command path
that ProseMirror handles like a paste:

    agent-browser --cdp 9222 execute \
      "document.querySelector('SELECTOR').focus(); \
       document.execCommand('insertHTML', false, '<p>...HTML...</p>');"

ProseMirror/TipTap/Slate all honor the insertHTML/paste command path.
Substack auto-saves the draft on insert, verify the "Saved" indicator afterward.

GOTCHA, TWO contenteditable elements: the publish page has a contenteditable
post-body editor AND a separate podcast editor, both matching
`[contenteditable]`. A bare `document.querySelector('[contenteditable]')` may
target the wrong one (the error output will echo text from the podcast editor,
revealing the mismatch). Scope the selector to the post form, and verify the
inserted text landed in the body via the snapshot before relying on it.

Banner image: the toolbar image tool opens a file dialog (no insert-by-URL),
so for a URL-based banner (logo.webp) insert an `<img>` tag through the same
insertHTML path at the top of the body instead.

## Script Structure

Each script is a standalone PEP-723 inline-metadata script (blog-scraping
helpers are duplicated per script so each can run independently with `uv run`).

- `scripts/linkedin_post.py` - LinkedIn post generator
- `scripts/bluesky_post.py` - BlueSky post generator
- `scripts/substack_post.py` - Substack post generator
- `scripts/summary.py` - Summary generator
- `scripts/tags.py` - Tag generator
- `scripts/banner.py` - DALL-E banner generator
