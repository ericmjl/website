# Buffer <-> Substack schedule sync

# Buffer-Substack Sync

Buffer is the canonical publication calendar (see the blogbot skill). Every
blog-promo post scheduled in Buffer must have a corresponding Substack
newsletter edition scheduled for the SAME date and time. This skill audits
both systems, reports the gaps, and fills them with scheduled Substack drafts.

## What It Does

1. Audits Buffer's scheduled (and recently sent) posts via the buffer MCP server.
2. Groups them into sync units (one unit per blog post; Buffer mirrors each
   post across LinkedIn + Bluesky, which is ONE unit, not two).
3. Classifies each unit: blog promo (contains an `ericmjl.github.io/blog/`
   URL) vs announcement (no blog URL, e.g. CFPs, retreat promos). Announcements
   are reported but NOT auto-synced.
4. Audits Substack drafts + published posts via `scripts/substack_sync.py`.
5. Reports gaps: Buffer blog promos with no matching Substack edition.
6. Generates each missing edition with blogbot's `generate_social.py`
   (NEVER hand-composition — see Step 5), creates it as a draft, and
   schedules it at the Buffer post's exact `dueAt`.
7. Verifies and hands back a table + edit URLs for review.

## Requirements

- Buffer MCP server connected (tools: `get_account`, `list_posts`).
- Authentication: a Substack session-cookie file at
  `~/.config/buffer-substack-sync/cookies.json` (see Authentication below).
  Email+password login is captcha-gated by Substack and will NOT work.
- `uv` (the script runs with `uv run`).
- For composing: the source blog post (`content/blog/<slug>/contents.lr` in
  the website repo) and the blogbot skill's voice/format conventions.

## Authentication (session cookies)

Substack's password login endpoint is behind a captcha
(`"Please complete the captcha to continue"`), so programmatic login does
not work. Auth is a session cookie: a valid `substack.sid` from a logged-in
browser. The script reads the cookie file automatically; `whoami` is the
smoke test.

Extraction recipe (Chrome on macOS; values never echoed):

```bash
# 1. Copy the cookie DB (Chrome flushes it periodically; a live session
#    expires months out — check expires_utc is in the future):
cp "$HOME/Library/Application Support/Google/Chrome/Default/Cookies" /tmp/cc
sqlite3 "file:/tmp/cc?immutable=1" \
  "SELECT name, datetime(expires_utc/1000000-11644473600,'unixepoch') FROM cookies \
   WHERE host_key LIKE '%substack%' AND name LIKE 'substack.%';"

# 2. Decrypt with the Chrome Safe Storage keychain item (GUI approval prompt
#    appears once). Scheme (current Chrome v10 format):
#    key = PBKDF2-HMAC-SHA1(keychain_secret, salt='saltysalt', 1003 iters, 16 bytes)
#    plaintext = AES-128-CBC(key, IV=16 spaces) of encrypted_value[3:]
#    cookie value = plaintext[:-pad], with a 32-byte sha256(host_key) PREFIX
#    in front of the value (strip it). v20-prefixed blobs are app-bound
#    encryption and are not derivable; skip them.
# 3. Write a FLAT dict {"cookie-name": "value"} to
#    ~/.config/buffer-substack-sync/cookies.json, chmod 0600.
#    requests' cookiejar rejects browser-style list-of-dicts.
```

Keep every cookie value out of chat output and logs; only lengths may be
printed. If `whoami` returns `APIError(code=401): Please sign in`, the
session expired: log into dspn.substack.com in Chrome, then re-extract.

## Usage

Run the script from the skill directory:

```bash
cd ~/.agents/skills/buffer-substack-sync/scripts

# Auth smoke test (no secrets printed)
uv run substack_sync.py whoami

# Inventory Substack
uv run substack_sync.py drafts --limit 100
uv run substack_sync.py published --limit 30

# Create + schedule (dry-run plan by default; --yes executes)
uv run substack_sync.py create \
  --title "Post Title" \
  --subtitle "One-line subtitle" \
  --body-file /tmp/buffer-substack-sync/<slug>.md \
  --schedule-at 2026-10-01T11:00:00Z \
  --yes

# Move or cancel an existing schedule
uv run substack_sync.py schedule --draft-id <id> --at 2026-10-02T11:00:00Z --yes
uv run substack_sync.py unschedule --draft-id <id> --yes
```

## How It Works

### Step 1: Pull the Buffer schedule

Call `get_account` for the organization id, then `list_posts` twice:
`status: ["scheduled"]` with a `dueAt` window from today to +90 days, and
`status: ["sent"]` for the last 30 days (for backfill detection). Sort by
`dueAt`.

### Step 2: Build sync units

Group Buffer posts by `dueAt` DATE. Posts sharing a date across LinkedIn and
Bluesky are ONE unit (the same blog post mirrored). For each unit, extract the
blog slug with:

```
ericmjl\.github\.io/blog/(\d{4})/(\d{1,2})/(\d{1,2})/([a-z0-9-]+)
```

- Slug found → blog promo unit, sync key = slug, target datetime = `dueAt`.
- No slug → announcement unit. Report it, skip it (ask Eric only if it looks
  like a blog promo whose URL is missing).

### Step 3: Audit Substack

`substack_sync.py drafts` calls the API's `filter=scheduled` and
`filter=draft` endpoints and returns both lists, with blog slugs extracted
from each scheduled draft's body (fetched per draft; list payloads omit
bodies). A unit is COVERED when:

- A scheduled draft's `blog_slugs` contains the unit's slug, OR
- A published post's title matches the blog post's title (`published` list
  output has no body, so published matching is by title).

History (validated against the real calendar, 2026-07 through 2026-09):
Substack editions go out the SAME DAY as the Buffer promo at ~11:00-11:03Z
(7 AM EDT), and Eric hand-schedules the Substack draft at exactly the
Buffer `dueAt`. The sync formalizes that habit.

### Step 4: Report gaps

Table: Buffer date | slug | Substack state (none / draft unscheduled /
scheduled / published). Units with no Substack edition are the gaps.

### Step 5: Generate each missing edition (blogbot, never hand-written)

**The copy MUST come from blogbot's generator, not hand-composition.**
Correction 2026-09-26 (Eric, after finding a hand-composed edition):
"doesn't look like it was written using the blogbot, can you make it such?"
— writing copy that merely FOLLOWS the blogbot format does not count as
using the blogbot. Run (from the website repo):

```bash
uv run .agents/skills/blogbot/scripts/generate_social.py <slug>
```

It emits all three platforms plus the structured Substack fields (title,
subtitle, greeting, body, sign-off, retreat P.S. bridge). Then apply the
MANDATORY mechanical scrubs — words otherwise untouched:

- Remove the duplicated raw blog URL the model sometimes emits next to the
  auto-hyperlinked "this post" (`format_post` already hyperlinks it).
- Convert quoted code fragments to backticks.
- Run the unslop banned-phrase scan; fix hard hits by stating the point
  directly (e.g. "the real work was" → direct statement).
- Verify facts and numbers against `contents.lr` before scheduling.

To REPLACE the body of an EXISTING edition (e.g. a hand-written draft that
needs converting): regenerate with the command above, then swap in place
via `update_draft_from_markdown` — see Gotchas for
`allow_unsupported_change=True` and the `draft_subtitle` no-op.

The template below is the TARGET SHAPE the generated copy must land in —
use it to VERIFY the generator output, not as a dictation source
(skipping the banner or greeting is a recurring error):

```markdown
![banner](https://ericmjl.github.io/blog/YYYY/M/D/<slug>/logo.webp)

Hello fellow datanistas,

<teaser paragraphs in Eric's voice: 1-2 short paragraphs on what the post
teaches and why it matters. Hyperlink the phrase "this post" to the blog
URL. NO em dashes anywhere.>

Read [this post](https://ericmjl.github.io/blog/YYYY/M/D/<slug>/) for the
full write-up.

Happy coding,
Eric

---

**P.S.** <bridge line connecting THIS post's topic to it> Daniel Chen and I
are organizing a retreat on how to learn anything with AI, February 2027.
Applications open to the waitlist first:
https://learn-anything.nonlinearlabs.ai/
```

- Banner date path = the post's `pub_date`, no leading zeros (`2026/9/11`,
  not `2026/09/11`).
- Title = the blog post's `title` field; Subtitle = its `summary` field
  (or a one-line subtitle if the summary is too long).
- Source the content from `content/blog/<slug>/contents.lr`, not from the
  Buffer post text alone.
- Verify the banner URL returns HTTP 200 first (the blog post must be live;
  GitHub Pages needs 1-5 min after a fresh merge). If the banner 404s, fix
  the date path; if the post itself 404s, stop and report.

### Step 6: Create + schedule

Dry-run first (no `--yes`), check the plan (banner flag true, slug detected,
schedule time right), then run with `--yes`. Scheduling rule:

- **Substack edition goes out at the SAME instant as the Buffer post's
  `dueAt`.** Buffer is the source of truth; never move a Buffer post to
  match Substack. `dueAt` is UTC (the queue slot 7 AM EDT = `11:00:00Z`;
  the 11 is UTC, not local).
- If the Buffer slot is TODAY and still in the future, schedule for it (the
  scheduled draft is editable/deletable until it fires — hand Eric the edit
  URL immediately). If the slot already passed, ask Eric: schedule later
  today or skip.
- Idempotency: NEVER create a second draft for a slug that already has one
  (draft, scheduled, or published). Match first, create only true gaps.
- The scheduled post publishes to the publication AND emails subscribers at
  trigger time (default newsletter behavior). This is intended.

### Step 7: Verify (mandatory)

Re-run `substack_sync.py drafts` and confirm: every blog-promo unit now has
a scheduled draft at the matching datetime. Close with:

```
| Buffer date | slug | Substack draft id | scheduled at | edit URL |
```

and remind Eric to eyeball each edit URL (banner rendered, links live)
before the trigger time.

## Backfill Mode (report only)

Sent Buffer blog promos from the last 30 days with no matching published
Substack post are MISSED editions. Report them; do not auto-publish (an
immediate publish emails subscribers — Eric decides whether to backfill).

## Gotchas

- Substack's API is unofficial (driven via the `python-substack` session);
  endpoints can move. If a command errors oddly, re-check the package, not
  just the cookie.
- API quirks learned 2026-09-24: (1) list responses use the key `posts`;
  (2) `limit` must be <= 20 (50 returns `400 Invalid value`); (3) schedule
  state lives in `scheduled_at` (`post_date` stays null until it fires);
  (4) the UNFILTERED `/drafts` endpoint returns the entire post archive
  oldest-first including published items — always use the filters; (5)
  `draft_body` is omitted from list payloads (fetch per draft for slug
  matching); (6) password login is captcha-gated — cookies only.
- The banner is a markdown image pointing at the LIVE logo.webp; Substack
  renders it via its own proxy. Verify it renders in the draft editor after
  creation — an API-created external image can silently fail to render.
- `schedule_draft` posts `trigger_at` in UTC ISO format; always pass a
  timezone-aware ISO timestamp (the script converts to UTC).
- The `create` script refuses em dashes (Eric's voice rule) — scrub before
  calling, don't bypass.
- Library quirk learned 2026-09-26: `update_draft_from_markdown` can replace a
  scheduled draft's body in place (keeps the draft id, edit URL, and its
  `postSchedules` trigger). Two gotchas: (a) it refuses when the existing body
  has nodes the markdown round-trip can't represent (the banner
  captionedImage) — passing `allow_unsupported_change=True` is correct for a
  deliberate full-body replacement; (b) its `subtitle=` kwarg SILENTLY NO-OPS
  because the PUT endpoint expects the `draft_subtitle` key — after updating,
  call `api.put_draft(id, draft_subtitle=...)` and re-`get_draft` to verify.
  Verify the body landed by re-fetching and rendering block types
  (captionedImage first, paragraphs, horizontal_rule, P.S.).
- API quirk learned 2026-09-26: `schedule --draft-id X --at T` ADDS a trigger,
  it does not replace an existing one — a rescheduled draft ends up with TWO
  `postSchedules` and can fire twice. To MOVE a scheduled edition:
  `unschedule --draft-id X --yes` (clears all triggers) then
  `schedule --draft-id X --at T --yes`, then verify `postSchedules` count == 1.
  Same-day Buffer lesson: deleting a queued post makes the remaining queue
  posts (shareMode addToQueue) auto-advance into the vacated slots, while
  customScheduled LinkedIn posts stay put and must be moved explicitly via
  `buffer_edit_post` — include `text` (edits are whole-post validated and
  reject without it), and passing `dueAt` with `mode: addToQueue` is rejected
  ("calculated automatically").
- Buffer OAuth sessions EXPIRE MID-SESSION (2026-09-26: writes worked at
  12:09Z, dead by 13:05Z; even reads bounce with "Re-authentication
  required"). Recovery: `mcp({action: "auth-start", server: "buffer"})` ->
  Eric approves the Buffer consent screen in his browser ->
  `mcp({action: "connect", server: "buffer"})` to reload tools -> retry.
  The consent page can render blank (network/browser flake); `open <auth-url>`
  in the default browser after confirming auth.buffer.com serves (curl 303)
  clears it. Substack scheduling is independent of Buffer auth, so finish the
  Substack half while waiting for approval.
