---
name: website-blog-publishing
description: |
  Publish or edit a blog post on the Lektor-based website (~/github/website) via a git worktree. Use when the user asks to publish, post, add, EDIT, RE-EDIT, or REVISE a blog post (including bare `edit: "..."` messages targeting an open PR), create a PR for a blog post, add a NEW TALK to the talks page from a pasted YouTube URL (see Adding a new talk section), or asks how blog posts are structured, on ericmjl.github.io. Covers the worktree convention (check `git worktree list` and reuse existing worktrees before creating new ones), the Lektor contents.lr format, commit message format, and the re-edit workflow for already-open PRs, and syncing the Substack
  newsletter schedule with Buffer (references/buffer-substack-sync.md).
  Publishing INCLUDES generating the summary, tags, and banner via the blogbot scripts (summary.py, tags.py, banner.py) BEFORE the commit — run them proactively, never wait to be asked (user correction 2026-08-30: "you should have used them...").
created_by: autolearn
created_at: "2026-06-14"
---

# Website Blog Publishing

Publish a blog post to Eric's Lektor-based website (`~/github/website`, deployed
at ericmjl.github.io) via a git worktree + PR. This is the "get the post onto the
site" step that sits between drafting (write-like-eric skill) and social
distribution (blogbot skill).

## Worktree Convention

The website repo uses **sibling-path worktrees**, one per blog post:

- Worktree path: `~/github/website-<slug>` (sibling of `~/github/website`)
- Branch: `blog/<slug>`
- Slug: kebab-case, no date prefix (the date lives in `pub_date`, not the dir name)

Confirm by checking existing worktrees before creating a new one:

```bash
cd ~/github/website && git worktree list
```

Existing example: `~/github/website-autolearn-blog` on branch `blog/opencode-autolearn`.

### Always base from origin/main

```bash
cd ~/github/website && git fetch origin main
git worktree add -b blog/<slug> ~/github/website-<slug> origin/main
```

**Why origin/main, not local main:** the main worktree often has uncommitted
changes (modified skills, .gitignore, untracked dirs). `git worktree add` from a
commit/branch gives a **clean working directory that does NOT carry over
uncommitted changes** from the main checkout. Basing on `origin/main` (after
fetch) also ensures the PR base is current.

## Lektor contents.lr Format

Each post is a directory `content/blog/<slug>/` containing a single `contents.lr`
file. Lektor uses an ini-style format where **every field is separated by a line
containing only `---`**:

```
title: My Post Title
---
author: Eric J. Ma
---
body:

<markdown content here>
---
pub_date: 2026-08-30
---
twitter_handle: ericmjl
---
summary:

A one- or two-sentence hook ending with a question. Read on to find out!
---
tags:

ai
agents
bayesian
```

Key points:
- **Copy the MOST RECENT post's `contents.lr`** (highest `pub_date`) to get the
  exact field order and spacing. Older posts can carry an OUTDATED field order:
  a 2026-08-30 session pulled a 2023 post as the template and its field order
  no longer matched the current canonical format, confirmed against the most
  recent post of that month. Go straight to the newest post; one template read.
- **The file ENDS with the last field's content.** No trailing `---` after the
  final `tags:` entries, and no stray generation artifacts: a trailing `===`
  line left at EOF by the build script had to be fixed immediately (2026-08-30).
  After writing, verify the last non-blank line is the final tag.
- The `body:` field holds the **markdown body WITHOUT an H1** — the title is
  rendered from the `title:` field, so do not repeat it as `# Heading` in body.
- `pub_date:` is `YYYY-MM-DD` (use today's date unless told otherwise).
- `tags:` lists one tag per line.
- `summary:` is a hook, often ending with a question ("Read on to find out!").
- `twitter_handle:` is `ericmjl`.

- ### Do NOT run markdownlint on .lr files

contents.lr uses Lektor's ini-style format (YAML-like fields separated by --- lines), NOT standard markdown. Running markdownlint-cli2 on a .lr file produces a wall of FALSE-POSITIVE errors that look like real issues but are not: MD013 (line length), MD003 (heading style), MD041 (first-line heading), MD022 (blanks around headings), MD026 (trailing punctuation), MD033 (inline HTML), MD060 (table style). These all stem from the front matter and field separators, not the body content.

This has bitten sessions twice (2026-02-15, 2026-06-15), each time costing turns analyzing 'errors' that were never real. Do ONE of these instead:
1. Skip markdownlint on .lr files entirely — lint only standalone .md files.
2. If you must lint, extract just the body: portion (everything after 'body:' up to the next standalone '---') into a temp .md file and lint that.

All existing posts in the repo would produce the same false positives, so these errors are never actionable signal for a contents.lr file.
- **Horizontal rules in the body corrupt field parsing.** Because Lektor uses a standalone `---` line as the field separator, a markdown horizontal rule written as `---` INSIDE the body is misread as a field boundary, silently truncating the body and creating a phantom field from the trailing content. When authoring a post that needs a horizontal rule, use `***` or `___` instead of `---`. When verifying an authored contents.lr, count the standalone `---` lines and confirm they land only at the field-separator positions (for the standard field set title/author/body/pub_date/twitter_handle/summary/tags that is exactly 6 separators (one between each adjacent pair of fields, with NO trailing `---` after the last field, i.e. 7 fields yield 6 separators); a higher count means a body `---` horizontal rule is being misparsed.
## Style Constraints (from website AGENTS.md)

- **No em dashes** in the body. Use commas or restructure the sentence.
- Write in Eric's natural voice (see write-like-eric skill). Avoid negation-first
  framing ("it's not X, it's Y"); state points directly.
- **No bare fork labels** (2026-09-26, seems post): when a post involves a fork,
  never use bare "upstream" or the bare project name as a label — every project
  mention explicitly names which repo it means ("the kavehmz original" vs
  "seems-laya, my fork"). This ambiguity drew corrections twice in one session
  ("Upstream's architecture — confusing, be explicit"; "'Seems ships as an
  import hook' — is that kavehmz's seems or seems-laya?").
- **No API-speak jargon**: phrases like "an envelope of answers with
  probabilities" or "speak a near-identical contract" are too jargon-y;
  describe what the code does in plain words ("give it a state and a list of
  questions, and it returns an answer with a probability for each one").
- **Comparative claims verified against actual code**: "near-identical" either
  becomes "identical" or states the concrete differences. Read both APIs before
  writing the comparison; never hand-wave a similarity claim.

## Commit Message Format

Website blog commits follow: `blog: <subject>`. Two styles, verified against merged history 2026-09-25 (Eric: "follow conventions with naming of PR, commit messages, etc..."):

- **Initial add commit**: `blog: add "<Post Title>"` where `<Post Title>` is the post's exact title in Title Case, wrapped in double quotes.
  - `blog: add "A data scientist walks into a GxP conference"`
  - `blog: add "Swapping the judge in a judgment language"`
- **Follow-up commits**: `blog: <lowercase descriptive subject>`
  - `blog: schedule gxp conference post for september 21`
  - `blog: tighten numbered list in gxp conference post`

Before committing or opening the PR, verify the live convention rather than trusting this section: `git log --oneline -8 -- content/blog/` for commit style and `gh pr list --state merged --limit 5` for PR-title style. Recorded formats can lag the repo.

## Full Workflow

1. Confirm worktree convention: `cd ~/github/website && git worktree list`
2. Fetch + create worktree from origin/main:
   `git fetch origin main && git worktree add -b blog/<slug> ~/github/website-<slug> origin/main`
3. Read the MOST RECENT post's `contents.lr` to confirm exact field order.
4. Create `content/blog/<slug>/contents.lr` with the post formatted per above.
5. Verify no em dashes in the body.
6. Commit: `git commit -m 'blog: <subject>'`
7. Push + open PR against main.
8. Monitor CI to green (per user preference).
9. Remove the worktree once the branch is pushed and the PR is open:

   ```bash
   cd ~/github/website && git worktree remove ~/github/website-<slug>
   ```

   The worktree is no longer needed once the PR is open, further work happens
   through the PR, not the worktree. User preference: clean up worktrees
   promptly rather than leaving them around.

- Step 8 operational detail (2026-08-14, PR #288): the website CI build (Lektor site build + Netlify deploy preview) takes roughly 5-10 minutes. Poll 'gh pr checks <num>' with spaced sleeps (60-90s between polls) rather than tight loops; a pending status after one or two polls is normal, not a failure. If checks are still pending after ~3 spaced polls, report status to the user with the PR link rather than blocking indefinitely. When green, the Netlify deploy preview (deploy-preview-<num>--ericmjl-github-io.netlify.app) is live and ready for the Post-PR URL Verification pass.
## Post-merge socials bundle (AUTO-TRIGGER, 2026-09-24)

The moment a blog post PR is merged and its public URL returns 200, run the
FULL socials bundle for that slug WITHOUT waiting to be asked. Eric,
2026-09-24: "as soon as we merge and the blog post is live, we trigger the
blogbot skill and just automatically draft the social media posts. That
means Substack, LinkedIn, BlueSky, all of it together."

Scope: the newly merged post, not historical backlog. Skip posts Eric
excludes. A GitHub Actions issue (blog-socials-pending workflow) is the
safety net for hand-merges.

1. Verify `https://ericmjl.github.io/blog/<YYYY>/<M>/<d>/<slug>/` AND its
   `logo.webp` return 200 (GitHub Pages rebuild delay: re-check at ~60s).
2. Generate copy via blogbot: `uv run .agents/skills/blogbot/scripts/generate_social.py <slug>`
   (single JSON: linkedin/bluesky/substack; can take 10-15 min when Z.ai is
   slow, so background it and compose manually from the tuned template if
   it stalls). Scrub em dashes.
3. Queue to Buffer: `buffer_create_post` with `mode: addToQueue` +
   `schedulingType: automatic` on BOTH channels. addToQueue DESYNCS when one
   channel has an extra announcement queued; reconcile to ONE date via
   edit_post WITH THE FULL TEXT carried forward (omitting text rejects with
   "Post must have either text or media").
4. Substack edition: compose per blogbot's "Publishing to Substack" format,
   create the draft via the BROWSER GOLDEN PATH (blogbot → "GOLDEN PATH for
   scheduled editions": real Chrome binary + agent-browser driving
   `dspn.substack.com/publish/post` in a cookie-copied data dir, fill title/
   subtitle refs, `eval` insertHTML for the body, verify imgs/links/Saved +
   screenshot), then schedule it at the reconciled Buffer dueAt:
   `substack_sync.py schedule --draft-id <id> --at <dueAt> --yes`.
   Same instant as Buffer, always.
5. Verify: `list_posts` (same date both channels, no collisions) +
   `substack_sync.py drafts` (scheduled draft at the matching instant), and
   hand Eric the Substack edit URL to eyeball before the trigger time.

## Substack <-> Buffer schedule sync

The newsletter-side half of publishing lives in
[references/buffer-substack-sync.md](references/buffer-substack-sync.md):
audit Buffer scheduled/sent posts, ensure every blog-promo post has a
scheduled Substack edition on the same date, report gaps, create and
schedule missing Substack drafts. Load it for "sync Substack with Buffer"
asks. (Former standalone buffer-substack-sync skill, folded 2026-09-27;
it had 5 loads, all adjacent to this skill or blogbot.)

## Related Skills

- **blogbot**: the socials machinery this bundle runs on (Buffer scheduling,
  Substack Schedule Sync, `scripts/substack_sync.py`).
- **write-like-eric**: draft the post body in Eric's voice.
- **blogbot**: generate the post's summary, tags, and banner via its scripts (summary.py / tags.py / banner.py, run BEFORE the commit — publishing is not done until these exist), then generate social posts from the published post + schedule to Buffer.
  blogbot's "URL Verification Before Scheduling" section depends on the
  `contents.lr` `pub_date` matching the DEPLOYED site — do not change pub_date
  after publishing without re-verifying URLs.

## Running blogbot scripts in a worktree

- blogbot's scripts (tags.py, summary.py, banner.py, and the *_post.py generators) locate the repo with find_repo_root(), which walks UP from the script's OWN location (Path(__file__).parent) looking for a .git entry. It does NOT use the current working directory.

Consequence: when you publish via a worktree, run the WORKTREE's own copy of the scripts so find_repo_root resolves to the worktree root and finds content/blog/<slug>/contents.lr there:

  uv run ~/github/website-<slug>/.agents/skills/blogbot/scripts/tags.py <slug>
  uv run ~/github/website-<slug>/.agents/skills/blogbot/scripts/summary.py <slug>
  uv run ~/github/website-<slug>/.agents/skills/blogbot/scripts/banner.py <slug>

If you instead run the MAIN repo's copy (~/github/website/.agents/...), find_repo_root resolves to ~/github/website and the scripts look in ~/github/website/content/blog/<slug>/, where the post does not exist (it lives only in the worktree). They will fail or operate on the wrong tree.

This works because a worktree's .git is a FILE (a gitdir pointer), and Path.exists() returns True for files as well as directories, so find_repo_root correctly stops at the worktree root. Verified: the .git file is ~56 bytes.
- BEFORE running blogbot scripts (tags.py, summary.py, banner.py) in a worktree, source the main clone's .env so API keys are available — gitignored files like .env do NOT carry over to worktrees:

  source ~/github/website/.env

The website .env uses the 'export' prefix on every line, so 'source' loads all keys (OPENAI_API_KEY and any others) directly into the shell. Even if OPENAI_API_KEY is already in the shell env, source .env anyway as a belt-and-suspenders measure so keys only present in .env (e.g. for banner generation or llamabot config) are available. User flagged this as a recurring gotcha: 'forgot about that, otherwise .env isn't available on the other side.'

Combined one-liner:

  source ~/github/website/.env && uv run ~/github/website-<slug>/.agents/skills/blogbot/scripts/tags.py <slug>

## Confidentiality Check

- Before drafting or publishing ANY blog post, verify the content does not disclose confidential business relationships. The user has private consulting engagements that must NEVER appear in public-facing content. Rules: (1) Treat all client, company, and consulting affiliations as confidential unless the user has explicitly stated they are public. (2) Before naming any specific company, client, or business relationship in a post, explicitly confirm with the user that it is public knowledge. (3) If the user mentions a constraint like 'don't write about X', scrub ALL references to X — direct mentions, indirect hints, and contextual clues that could identify it. (4) This applies to the blog body, summary, tags, and commit messages alike.

- Private INDIVIDUALS are in scope too, not just companies: prospective participants, call attendees, and other private people must be ANONYMIZED to generic descriptors ('One guy I talked with', 'someone on a recent call'); when verifying, grep for the real name AND surname-only fragments, not just the full name. Calibrate indirect hints by identifiability: generic career facts ('I used to be a lab scientist', 'sabbatical') are ALLOWED once no company is named; first-person employer identifiers ('my employer', 'the company I work for') are NOT. Give the reviewer an explicit WHITELIST of user-confirmed-public entities (co-organizers, conferences, institutions, public anecdotes, public event sites/dates) so a CLEAN verdict is reachable and whitelisted items are not re-flagged. Refined 2026-08-30 during the learn-anything post pre-publish verify pass; entity specifics (which employer, which people) live in the local memory registry, never in this skill file, because it may sync to public skill repos.
- INDIVIDUALS and REMOVAL-INTEGRITY are in scope too (2026-08-30): private INDIVIDUALS — prospective participants, call attendees, prospects — must never appear by name in public content; and when an entity was deliberately scrubbed or anonymized in a prior edit, the confidentiality reviewer must VERIFY the scrub is INTACT, not just re-scan for company names. Verification procedure: (1) grep the exact name plus surname-only and variant forms — expect ZERO matches; (2) grep indirect-hint phrases ('my employer', 'the company I work for', employer-sense 'day job') that could identify a scrubbed affiliation without naming it; (3) confirm the sanitized replacement wording is present in the draft ('One guy I talked with', 'someone on a recent call' style); (4) explicitly classify ALLOWED adjacent terms so the reviewer does not over-flag — 'sabbatical' with no company named is OK, generic bio statements ('I used to be a lab scientist') are OK, reader-directed generics ('even if you have a day job') are OK; (5) deliver the verdict as a line-cited near-miss table classifying EVERY named entity against an explicit public-knowledge whitelist (public figures with public anecdotes, co-organizers, public event dates/sites/links). Do not write the confidential entity names themselves into PUBLIC skill or repo files — describe them generically (the private memory store may name them).
- 404 disambiguation during link verification (2026-09-24, laya-pharma review): when verifying links in an UNMERGED draft, a repo blob URL that 404s is not automatically a private-repo leak. Diagnose: fetch the blob URL, then the repo root (HTTP 200 = repo is public), confirm the file exists on the draft's working branch (`ls`), and check `git remote -v` origin. Classify: (1) private repo = broken link once public, flag it; (2) unmerged branch = link targets main and resolves after merge; report as an informational publish-ordering note, NEVER as a confidentiality finding; (3) wrong URL = fix before publish.
## Mandatory post-draft quality chain (coherence THEN voice)

- After ANY drafting or substantive-editing pass on a blog post (new draft or re-edit), run the chain AUTOMATICALLY, no opt-in and no asking: (1) DRAFT/EDIT; (2) coherent-writing coherence pass immediately after; (3) write-like-eric voice pass after coherence; (4) only then present/commit. Coherence BEFORE voice: voice polish on incoherent structure gets redone. Full spec lives in the vault-blog-drafting skill (mandatory-post-draft-quality-chain); this mirrors it for website-side editing sessions. Stated 2026-08-30 (sea-of-change edit): 'as soon as we're drafting, we should immediately move into the coherence check. And after we're done with the coherence check, we need to do the write-like-Eric check... those need to happen automatically.'

## Pre-publish Review

- ## Pre-publish Review (parallel read-only subagents)

Before committing the post, dispatch parallel READ-ONLY review subagents, each with ONE distinct mandate. Split concerns so each reporter can focus. Consolidate all findings, then apply fixes yourself before committing.

1. **Privacy/confidentiality review**: scan the ENTIRE contents.lr for any mention, hint, or allusion to confidential entities (client names, consulting relationships, private projects). Report ANY match including indirect references. (See Confidentiality Check section.)

2. **Style compliance review**: check against write-like-eric rules — no em dashes as sentence separators, no "not X, it's Y" patterns, no negation-first constructions, no formulaic scaffolding ("in conclusion", "overall"). Report every instance with line numbers.

3. **Technical accuracy review**: cross-check all factual claims (dogfooding stats, architecture descriptions, feature lists) against the source of truth (README.md, source code, actual data files). Report any discrepancy with the claimed value vs. the source value.

Each subagent reads the file and reports findings only — it does NOT edit. Insert this review between Full Workflow step 5 (verify no em dashes) and step 6 (commit). This is an instance of the global parallel-subagent-review pattern: multiple READ-ONLY reporters with distinct focus areas, consolidated by the lead agent.

- DICTATED-SENTENCE CARVE-OUT (2026-08-30): in the style compliance review, sentences Eric DICTATED verbatim into the draft are EXEMPT from style-compliance flags — negation-flavored constructions ('not to a demographic', 'isn't an AI retreat, it's a...'), first-person framing ('I want to use this blog post to explain why'), and idiosyncratic phrasings are his voice, not violations. The lead agent preserves them verbatim even when the style reporter flags them; only assistant-generated scaffolding gets fixed. Distinguish dictation (keep) from agent prose (fix). PUBLIC-SURFACE SOURCE OF TRUTH (2026-08-30, technical accuracy review): when a post references facts that also appear on a public surface (retreat/product landing page copy), fact-check against the LIVE PUBLIC COPY, not internal planning docs — internal docs go stale and readers compare against the public page (a draft's 'American South' contradicted the site's 'United States (venue to be announced)'). POST-FIX SYNC GATE: after applying consolidated review fixes to BOTH the vault draft and contents.lr, re-diff the two bodies; the ONLY acceptable diff is the known H1/title-field divergence (vault draft carries the H1; contents.lr keeps it in title:). Any other diff = a fix landed on one surface only.
## Updating an existing PR (re-edit after worktree removal)

- After the worktree is removed and the PR is open, the user will often review the draft and send further edits ('edit: ...'). The Full Workflow above ends at removal, so this sub-workflow is the gap it does not cover.

## Recreate the worktree on the existing branch (no -b flag)

The branch already exists, so do NOT pass -b (that errors with 'already exists'):

```bash
cd ~/github/website && git worktree add ~/github/website-<slug> blog/<slug>
```

## Rebuild cycle

1. Apply the edits to the VAULT DRAFT (Blog Draft/<slug>.md), not contents.lr directly. The vault draft is the source of truth.
2. Re-run the build script to regenerate contents.lr from the edited draft. Note: build scripts often hardcode the worktree output path, so recreate the worktree at the SAME path it was first created at for the script to work unchanged.
3. Lint only the .md draft (never the .lr file — see the markdownlint false-positives section).
4. Commit: git commit -m 'blog: <subject>' (same format as initial publish).
5. Push — this adds a NEW commit to the PR, no force-push.
6. Remove the worktree again: git worktree remove ~/github/website-<slug>

## Batch rapid-fire edits

When the user sends edits one at a time in quick succession (e.g. four 'edit: ...' messages in a row), do NOT do a full recreate-rebuild-push cycle per edit. Accumulate the edits into the vault draft, then run ONE rebuild-push cycle for the batch. The PR does not need to be in sync after every single word change. Batching avoids N worktree lifecycles for N edits.

Mid-cycle arrivals: edits that arrive WHILE a batch is already in flight (worktree already recreated, build already running) fold into that SAME in-flight batch. Apply the new edit to the vault draft and proceed with the rebuild you were doing. Do NOT abort the running build, and do NOT queue a second rebuild-push cycle for the late edit (that would create the N-cycles-per-N-edits churn batching exists to prevent). If you already showed a diff and then a new edit arrives before commit, re-run the build so the committed diff reflects all edits.

## PR Title Format

- Blog post PR titles MUST include the publication date in [YYYY-MM-DD] format at the end of the title, so you can identify which posts are scheduled for which day at a glance from the PR list.

Format: `blog: <Post Title> [YYYY-MM-DD]`

`<Post Title>` is the post's exact title in Title Case, no `add`, no `(blog)` wrapper. Merged precedents:
- `blog: A data scientist walks into a GxP conference [2026-09-21]` (PR #303)
- `blog: A Multiplayer Chat Agent Is Just a Coding Agent in a Container [2026-09-25]` (PR #305)

Non-blog PRs keep their own type prefix plus the bracketed date when they carry a scheduled post: `feat(blog): Reviewing the PLAUD Note Pro [2026-09-20]`.

Steps:
1. Read the `pub_date` field from the post's `content/blog/<slug>/contents.lr`.
2. Set the PR title with the date in [YYYY-MM-DD] format via `gh pr edit <number> --title "<new title>"`.
3. If `pub_date` changes (rescheduling), UPDATE the PR title to match the new date. Do not let it go stale.

The commit message has no date; the PR title has the date.

## Post-PR URL Verification

- Once a blog post PR is OPEN and CI has built a deploy preview, do a full URL QA pass. The user explicitly requested this as a required step, not optional. Procedure:

1. Locate the deploy preview URL for the PR. The site is deployed to GitHub Pages (ericmjl.github.io); per-PR deploy previews (if present) appear in the PR checks/deployments or a comment from the deploy bot. If no preview environment exists, verify URLs against the PR's rendered content instead and say so.
2. Open the deploy preview (the full site) and confirm it loads.
3. Open the blog index page and confirm it loads.
4. Open the NEW blog post specifically and confirm it renders.
5. Extract EVERY link from the post body markdown (the body: field of content/blog/<slug>/contents.lr) and verify each resolves HTTP 200. This includes internal links to other blog posts (date-based: /blog/YYYY/M/d/slug/), external URLs, and image references (./name.webp). Internal blog-post links are the most error-prone: the date component must match the target post's pub_date, and the post must exist on the preview.
6. Report any broken link with its status and the source line in contents.lr.

Do this for ALL open blog PRs when asked, not just the one you authored. This step sits AFTER 'Monitor CI to green' in the Full Workflow (step 8) — the deploy preview must be live before URLs can be checked. Pairs with the existing 'Pre-publish Review' (which runs before commit) and memory entry on URL verification (which covers pre-publish factual claims).
- ### Accessing files across multiple PR branches

When the Post-PR URL Verification step requires reading contents.lr from MULTIPLE open PRs at once (e.g., extracting links from 4 posts to verify them in parallel), do NOT juggle git checkout on the main working tree and do NOT fall back to git show origin/<branch>:<path>. Parallel checkouts cause index.lock contention; git show works but is the wrong default.

Instead, create one worktree per PR branch (the SAME sibling-path worktree convention as publishing), then read files directly from each worktree filesystem:

```bash
cd ~/github/website
git fetch origin blog/<slug-a> blog/<slug-b> ...
git worktree add ~/github/website-verify-<slug-a> blog/<slug-a>
git worktree add ~/github/website-verify-<slug-b> blog/<slug-b>
```

This generalizes the Worktree Convention to the READ side, not just the publish side. The user has corrected this pattern twice (user-profile entry #8 + this session): worktrees are the tool for ANY multi-branch access in this repo. Notably, the blog post being inspected in this session was itself titled 'Git worktrees for beginners' — if you are touching multiple PR branches and not reaching for worktrees, treat that as a signal to reconsider.

## Conflict with opencode ephemeral pr-worktrees

- When recreating a worktree for an existing PR branch, the git worktree add command on line ~188 will FAIL with 'fatal: blog/<slug> is already checked out at .../pr-worktrees/pr-<num>-<slug>' if opencode has auto-created an ephemeral pr-worktree for that PR. opencode creates one pr-worktree per PR the user opens, so this conflict is EXPECTED whenever the user says 'in a new worktree' for an already-open PR. Resolution sequence: (1) git worktree list to confirm the branch is held by an ephemeral pr-worktrees path; (2) check if it is clean: git -C <pr-worktree-path> status --porcelain (empty output = clean); (3) if CLEAN, remove it and create the sibling-path worktree: git worktree remove <pr-worktree-path> && git worktree add ~/github/website-<slug> blog/<slug>; (4) if DIRTY, do NOT remove it — ask the user, since removal could lose uncommitted work. Never silently reuse the ephemeral worktree when the user asked for a new one; never force the user onto a path they did not request.

## Merging the PR

- The website repo (ericmjl/website) is configured for REBASE merges ONLY: allow_rebase_merge=true, allow_squash_merge=false, allow_merge_commit=false (verified 2026-06-16 via gh api). gh pr merge without an explicit flag will pick the wrong method or fail. Always use --rebase.

Before merging, verify the PR is ready:
1. gh pr view <num> -R ericmjl/website --json mergeable,mergeStateStatus,isDraft,statusCheckRollup
2. Confirm mergeable==MERGEABLE, mergeStateStatus==CLEAN, isDraft==false, and all checks pass (not UNSTABLE/pending).

Discover the repo's allowed methods at any time:
  gh api repos/ericmjl/website --jq '{squash:.allow_squash_merge, merge:.allow_merge_commit, rebase:.allow_rebase_merge}'

Merge and delete the branch:
  gh pr merge <num> -R ericmjl/website --rebase --delete-branch

This is the final step of the lifecycle after worktree removal (Full Workflow step 9). The user consistently asks to merge website PRs via rebase (PRs 256, 257, 263), matching the repo config. If checks are still running, wait for green before merging (per user-profile entry: always monitor CI to green).

- ### Merge timing: hold for Eric's preview review (2026-08-30, PR #294)

Opening the PR does NOT authorize the merge. A blog post is reader-facing live content: once the PR is open and CI is green, STOP and hand Eric the URLs — he clicks around the deploy preview and merges on his own signal ('merge' or 'can be merged, use rebase'). An explicit hold ('don't merge to rebase just yet') persists until he lifts it with an explicit merge command; a LATER generic 'Take action now' in the same arc is NOT merge authorization — it authorizes only the remaining hold-compatible close-out steps (URL delivery, verification pass, worktree removal). Proactively include BOTH URLs in the held-PR report so he never has to ask ('give me the url please', 2026-08-30 — he had to ask; deliver up front next time): (1) the final live URL https://ericmjl.github.io/blog/<YYYY>/<M>/<d>/<slug>/ (M/d NOT zero-padded, matching pub_date), and (2) the deploy-preview URL https://deploy-preview-<num>--ericmjl-github-io.netlify.app/ plus the post path on the preview. Run the Post-PR URL Verification pass BEFORE delivering, so the report can state that every in-body link resolves and all checks are green.
- MERGE GATE — blog posts are content Eric reads on the preview BEFORE they ship (2026-08-30, PR #294): after the PR is open, CI is green, and the Post-PR URL Verification pass passes, the terminal state is PR open + deploy-preview URL handed to Eric — do NOT rebase-merge in the close-out turn on your own initiative, even though merging is the lifecycle's final step. A pre-emptive hold ('don't merge to rebase just yet') or a bare 'give me the url please' means Eric is claiming preview review; the merge runs ONLY on his explicit commission ('merge' / 'can be merged, use rebase') — execute the full merge sequence above in that same turn. This is the website-blog instance of the reserved-review and copy/content carve-outs in the git-worktree-session-workflow skill (2026-08-27/28); blog content on a merge->production repo makes the merge a genuine user decision. Scope: TALKS pages equally, not just blog posts (2026-09-01 talk PR #295: the close-out drew Eric's generic 'Take action now.' — under this gate it authorizes only already-complete close-out steps, never the merge).
- ACCEPTANCE-HANDLE CARVE-OUT — reconciles this gate with the MERGE HOLD PRECEDENCE clause below (2026-09-13, errata PR #298; 3rd website-arc instance after blog PR #294 2026-08-30 and talk PR #295 2026-09-01): the word 'generic' in the clauses above means NO pending merge question was offered. When the CI-green close-out ITSELF ends with a one-word acceptance handle — 'Say the word and I'll merge with rebase', 'say merge and I'll rebase-merge' — Eric's terse 'Take action now.' is him INVOKING the handle: that IS the explicit merge commission (memory #1575 direct-answer test; both MERGE HOLD PRECEDENCE conditions already hold since the close-out delivered URLs + green checks and its pending question WAS the merge). Execute gh pr merge <num> -R ericmjl/website --rebase --delete-branch in THAT SAME TURN — do not cite this gate to refuse a handle-invoked merge. If the session exits at that interrupt before the merge runs, the arc is INCOMPLETE, not declined: the PR stays open + mergeable with merge already authorized — the next session (steering off the reviewer's resume-state memory) runs the merge + main sync + post-merge live-site verification WITHOUT re-asking. Errata PRs to already-published posts hit this shape routinely: Eric commissioned the correction and saw the register directive applied, so the verified close-out + handle is the standard completion path.
## Link Enhancement Pass

- When polishing a blog draft (initial or re-edit), proactively identify linkable entities in the body and offer to link them. This is a DISTINCT editing pass from URL verification (Post-PR URL Verification checks EXISTING links resolve); this pass CREATES new links. The user does this routinely ('find other things that can be linked') and often fires specific link requests in a rapid burst ('also link to X', 'and the repo too', 2026-08-30).

Procedure:
1. Scan the body markdown for proper nouns: tools/libraries (e.g. OpenCode, Convex, React, Remotion, marimo, Obsidian), methodologies (e.g. EARS specifications), people, Eric's own tutorials/talks, and references to PRIOR blog posts.
2. Categorize each candidate:
   - Prior blog post -> internal date-based link /blog/YYYY/M/d/<slug>/ (look up the target post's pub_date to build the URL correctly).
   - Tool/library/concept -> official homepage or canonical docs page (verified homepages: opencode.ai, convex.dev, remotion.dev, marimo.io, obsidian.md).
   - WORKFLOW / METHODOLOGY / TECHNIQUE Eric uses (HLD/LLD/EARS, design-driven-dev, a skill, a protocol) -> link to a CONCRETE PUBLIC EXAMPLE of where it was actually applied (e.g. github.com/ericmjl/build-deep-research-agent/tree/main/docs/designs for the HLD/LLD/EARS workflow), NOT an abstract external definition (Wikipedia, the original paper, the spec site). The user's phrase is 'link to source where we got it from' — the SOURCE of a workflow is where Eric operationalized it, not where it was first defined in the literature. If no Eric repo uses it yet, link to whatever public repo most concretely demonstrates it. Stated 2026-07-19 (SciPy 2026 blog, design-docs tip).
   - TUTORIAL/TALK BY ERIC mentioned in the post -> link the tutorial's PAGE ON THE WEBSITE (and/or the YouTube video directly). Every talk/tutorial has a site page — content/talks/<slug>/ or content/teaching/<slug>/ — carrying its YouTube embed in a '#### youtube #### url' field; find it by grepping those dirs for the tutorial title or video ID. Page URL is /talks/<slug>/ or /teaching/<slug>/ (default Lektor child URLs; only models/blog.ini defines slug_format). Stated 2026-08-30: 'the videos are on youtube and I also have those youtube videos linked in my website too.'
   - NAMED ACADEMIC -> their university faculty/home page (e.g. David Duvenaud -> his UofT page), or their PERSONAL SITE when that is the canonical home (e.g. Daniel Chen -> https://chendaniely.github.io/ — a personal site, not a university page). Stated 2026-08-30.
   - NAMED OPEN-SOURCE PROJECT/REPO -> its GitHub repo URL (e.g. autograd -> github.com/HIPS/autograd). Before linking any repo from public content, run the visibility check: `gh repo view --json visibility <owner/repo>` — private repos 404 for readers.
   - Person -> their site/handle ONLY if the user has confirmed it is public (see Confidentiality Check).
3. NEVER guess a URL. For each candidate either verify it resolves HTTP 200 before inserting, or ask the user for it. An unfamiliar name like 'Hermes' (a coding agent) is exactly the case to ask about rather than invent a link. Guessing links violates the verify-before-publish rule and creates broken links the URL-verification pass will then have to catch.
4. Present the candidate list with proposed URLs and ask which to link plus any URLs you could not verify. Do not mass-insert without confirmation. EXCEPTION — direct commissions skip the menu: when Eric names the links himself in the arc ('and then also link to david duvenaud's page at UofT', 'link to Daniel Chen's personal website'), those are ORDERS, not candidates — find, verify (HTTP 200), and insert them in the same arc without re-presenting a list. The confirmation step is for links the AGENT proposes, never for links Eric commissioned.
5. After links are added, run the Post-PR URL Verification pass on the deploy preview.

Pairs with: Post-PR URL Verification (verifies the links this pass adds) and the global memory rule on verifying claims/URLs before publishing.

## Deriving a published post's URL (never guess)

When you need the URL of an EXISTING published post (to link to it from another post, to verify a cross-reference, or to check it resolves), do NOT guess the slug or date. Both are deterministically available from the website repo via a 2-step lookup:

1. Find the SLUG (directory name):
   ```bash
   ls ~/github/website/content/blog/ | grep -i <keyword-from-title>
   ```
   Example: `ls ~/github/website/content/blog/ | grep -i multitask` yields `how-to-multitask-better-with-agent-harnesses`.

2. Find the DATE (pub_date field):
   ```bash
   grep '^pub_date:' ~/github/website/content/blog/<slug>/contents.lr
   ```
   Example: yields `pub_date: 2026-06-08`.

3. Build the URL: `https://ericmjl.github.io/blog/<YYYY>/<M>/<d>/<slug>/`
   Where M and d are NOT zero-padded: `2026-06-08` becomes `/blog/2026/6/8/<slug>/`.

The slug frequently differs from what you would guess from the title (the published post "How to multitask with agent harnesses" has slug `how-to-multitask-BETTER-with-agent-harnesses`). The date is often different from an assumption too. Guessing either wastes turns on wrong-URL debugging. Discovered 2026-07-12 (herdr blog post review): the agent guessed slug `how-to-multitask-with-agent-harnesses` (missing "better") and date `2026-06-14` (actual `2026-06-08`), requiring a correction pass.

## Renaming the slug when the title changes

- When you revise a blog post's title (initial draft or re-edit), the slug directory name must be updated to match the new title, NOT left as the old slug. Treat the slug rename as an inseparable part of the title-change operation, not an optional extra the user has to ask for separately. The user had to prompt 'update the file slug too...' in a 2026-06-17 session after the assistant changed the title text but left content/blog/agents-amplify-expertise/ unchanged.

Procedure:
1. Rename the content directory via git mv so git tracks it as a move (preserves history): git mv content/blog/<old-slug> content/blog/<new-slug>
2. No contents.lr edits are needed: Lektor derives the URL from the directory name, attachments (logo.webp, banners) move with the directory, and the body uses ./relative image refs that still resolve.
3. URL consequence: because Lektor derives the URL from the dir name, renaming changes the published URL. For a NEW/UNPUBLISHED post this is safe (no inbound links exist). For a PUBLISHED post, inbound internal links from OTHER posts use the date-based form /blog/YYYY/M/d/<slug>/ and will BREAK if the slug changes; grep the repo for the old slug before renaming a live post.
4. Worktree dir (~/github/website-<old-slug>) and branch (blog/<old-slug>) are local conventions. For an unpublished post, rename them too for tidiness (git branch -m, recreate worktree). For an open PR, weigh the disruption of renaming the PR branch vs. leaving it as the old slug (the branch name is internal; only the merged dir matters to the live site).
5. Commit as usual (blog: <subject>).

GOTCHA: a title change without a slug rename leaves the directory name lying about its contents. Always rename together.

## Evolving/living-document posts

- ## Evolving / "living document" posts: keeping pub_date while adding honest updates

Some blog posts document a compounding story (build logs, projects that keep
growing after the initial publish). The user may ask you to add updates to such
a post WITHOUT changing the `pub_date` field. This is the right call: `pub_date`
is the original publication anchor and downstream systems (blogbot URL
verification, date-based internal links) depend on it staying fixed.

The constraint: do this HONESTLY. The reader must be able to tell which content
is original vs newly added, but WITHOUT clunky "UPDATED:" markers, date-stamps
on paragraphs, or revision logs that break the narrative voice.

Technique — weave temporal signals into the PROSE:
- Use present-tense "as I write this" to anchor stats/features to the writing
  moment (late June) rather than implying they are as-of the original pub_date.
- Use a "first version went out a few weeks back; it has been compounding ever
  since" framing to signal the post has evolved since publication.
- If a "what comes next" item has since been completed, move it out of the
  future-tense "next" section (it is no longer next) rather than leaving a
  stale TODO. Completed one-off migration cleanups (e.g. re-pointing migrated
  symlinks) can simply be dropped from "what comes next" if they are too
  in-the-weeds for the narrative.
- Keep it skillful, artful, and masterful — like a good writer. The honesty
  lives in the sentence construction, not in metadata.

Stated 07-01: "find a way to keep the date as-is, but skillfully, artfully, and
masterfully (like a good writer) weave in the updates for me without
compromising on honesty/truthfulness."

## Dollar signs in body prose vs MathJax

- Dollar signs in body prose get grabbed as inline LaTeX math (MathJax). The site loads MathJax with `inlineMath: [['$','$'], ['\\(','\\)']]` (templates/layout.html), so a literal `$` in prose is treated as an inline-math delimiter. This bites currency/price references — e.g. `$0.000542 per second, or about $1.95 per hour` renders the span between the two `$` as broken math.

  **PREFERRED FIX (user-confirmed 07-02 "Just use the \$ thingy..."):** escape each literal dollar sign with a backslash — `\$1.95`. This is the ESTABLISHED site convention; existing posts already use it: `mastering-linkedin-connections` (`\$150,000`, `\$75/hr`), `the-selfish-reason` (`\$600,000`). The `\$` escape renders correctly and reads cleanly in source. Default to `\$` for all currency.

  Do NOT disable the dollar-sign math delimiter site-wide — real inline math (`$R^2$`, `$N(0,0.5)$`, display math `$$...$$`) is actively used across many posts and must keep working.

  Fallback forms if backslash-escape is problematic in a specific context: the HTML entity `&#36;1.95`, or wrap in a span (`<span>$1.95</span>`). But `\$` is the default.

  This is the INVERSE of Eric's own post `help-mathjax-render-correctly-by-wrapping-latex-in-span-tags` (which is about getting math TO render); here the goal is keeping plain currency OUT of math. Same class of body-authoring gotcha as the `---` horizontal-rule one above. User flag 07-02: "the dollar sign needs to be escaped so that we don't render as latex inline." Verify the rendered deploy preview shows literal `$1.95`, not a red math-error box, before merging.

## Consolidating to the main worktree

- ## Consolidating to the main worktree (alternative to the secondary-worktree recreate cycle)

The user may ask to consolidate work into the main worktree (`~/github/website`) instead of using a secondary worktree (`~/github/website-<slug>`) — either to make follow-up edits to an ALREADY-OPEN PR (replacing the recreate cycle in 'Updating an existing PR'), or to OPEN THE PR FROM THE MAIN CHECKOUT FROM THE START when a secondary worktree already exists but the user wants to skip worktree juggling. This is a legitimate alternative: keep all further work in the main worktree directly on the PR branch.

### When to use this

- The user wants to make follow-up edits to an OPEN PR from the main working directory, OR they want to OPEN A NEW PR from the main checkout when a secondary worktree already exists but they want to skip it
- They explicitly ask to 'switch the main worktree', 'target main worktree to the PR branch', 'work on the PR from the main checkout', OR say 'let's do it here, not within a worktree' / 'do it here' / 'open the PR from here' (here = main checkout, contrasted with the worktree)
- The secondary worktree (`~/github/website-<slug>`) is no longer needed (or the user wants to abandon it before opening the PR)

### Procedure

A git branch can be checked out in ONLY ONE worktree at a time. To move the PR branch into the main worktree.

**If the PR is NOT yet open and the work is UNCOMMITTED in the secondary worktree** (the 'open the PR from the main checkout from the start' entry point), you MUST commit in the secondary worktree FIRST so the work is saved on the branch, THEN remove the secondary worktree and check out the branch in the main worktree. Without the commit-first step, removing the secondary worktree loses the uncommitted work.

1. Verify the secondary worktree is clean (commit already pushed, no uncommitted work):
   ```bash
   git -C ~/github/website-<slug> status --porcelain
   ```
   Empty output = safe to remove. Non-empty = ask the user before removing (could lose work).

2. Remove the secondary worktree (releases the branch checkout lock):
   ```bash
   cd ~/github/website && git worktree remove ~/github/website-<slug>
   ```

3. Check out the PR branch in the main worktree:
   ```bash
   cd ~/github/website && git checkout blog/<slug>
   ```

Further edits commit directly on the branch and push with `git push` (tracks the PR).

### Gotcha: pre-existing uncommitted changes carry over

When you check out the PR branch in the main worktree, any UNCOMMITTED changes that were sitting in the main worktree on `main` (modified files, staged moves, untracked dirs) are carried over to the PR branch checkout by default — git preserves them across checkout when there is no conflict. So `git status` on the PR branch may show modifications (e.g. `M AGENTS.md`, a moved directory) that are NOT part of the PR diff against `main`. This is expected: the PR diff is `main...PR-branch` (commits unique to the PR branch), while `git status` shows the working-tree state including pre-existing uncommitted changes. Do not mistake these for PR-introduced changes, and do not commit them to the PR branch unless the user asks.

### Choosing between this and the recreate cycle

**Main-worktree consolidation is the DEFAULT for active iteration.** The Lektor dev server (`lektor server`, localhost:5959) serves the MAIN checkout's working tree. Work done in a secondary worktree does NOT reflect on localhost, so it breaks the fast-iteration feedback loop the user relies on. The moment you enter the active-edit/review loop on a single post, check out the PR branch in the main worktree rather than cycling a secondary worktree. Stated 07-24: 'this default worktree is on main, I need it to be the place we work on actively updated things for fast iteration, can't keep relying on other worktrees as it'll not reflect on my localhost.'

- **Recreate cycle** (Updating an existing PR): use ONLY when you need a CLEAN working tree per PR, or when juggling MULTIPLE PRs simultaneously (one worktree per PR). The localhost-dev-server gap means secondary worktrees are NOT suited for active iteration with live preview.
- **Main-worktree consolidation** (this section): the DEFAULT for single-PR iterative editing. One working directory, no worktree lifecycle overhead, AND localhost:5959 reflects your edits immediately.

Discovery history: (1) 2026-07-19 (xarray-in-biology-take-2 PR #269): user said 'help me target main worktree to the PR branch please' after PR creation. (2) 2026-07-19 (curiosity-at-the-wheel PR): user said 'open a new PR for this one please. let's do it here, not within a worktree'. (3) 2026-07-24 (going-bayesian-automates-data-analysis PR #275): user elevated consolidation to the DEFAULT — 'I need it to be the place we work on actively updated things for fast iteration, can't keep relying on other worktrees as it'll not reflect on my localhost.'

## Bidirectional sync check (before multi-pass prose editing)

- The vault draft is the source of truth, but drift accumulates when prose fixes land in contents.lr without being back-synced — e.g. quick typo or style fixes applied directly during a push, contraction fixes made in the worktree, or re-edit shortcuts. Symptom: the vault draft still has old text (uncontracted 'That is draining', stale attributions) while contents.lr has the fix.

BEFORE running any multi-pass prose editing on a published/in-flight post (coherent-writing, voice polishing, structural revision), VERIFY the vault draft body matches the contents.lr body. If they differ, back-sync the vault draft ← contents.lr FIRST (apply the missing fixes to the vault draft), run the editing passes on the synced vault draft, then propagate forward via the normal rebuild cycle.

Discovered 2026-07-19 running a coherent-writing pass on the 'Choose Your Own AI Policy' post: contraction fixes and a Daniel Chen attribution rewording applied to contents.lr in a prior session had not been synced to the vault draft, so the coherence pass was about to operate on stale text.

## Verify push landed on PR (gh pr view cache lag)

- Immediately after pushing a commit to a blog PR branch, 'gh pr view <n>' may report an OLD commit count and OLD 'latest commit' summary — e.g. showed 3 commits + 'blog: add banner image' as latest, when 5 commits including 'blog: coherence pass on ...' were actually on the remote branch. This is gh CLI METADATA CACHE LAG (seconds-to-minutes), NOT a failed push.

When the user asks 'did the edit actually land on the PR?' or whenever you verify a push, do NOT trust the commit-count or latest-commit fields of 'gh pr view'. Use ONE of these ground-truth checks instead:

- 'git ls-remote origin blog/<slug>' — shows the remote branch's HEAD sha (compare against your local 'git rev-parse HEAD')
- 'git fetch origin blog/<slug> && git log origin/blog/<slug> --oneline' — fresh remote log after fetch
- 'gh api repos/ericmjl/website/commits/blog/<slug> --jq .[0].commit.message' — fresh API call bypassing the view cache

Symptom signature: 'I just pushed commit X, but gh pr view shows commit Y as latest.' Resolution: trust git ls-remote / git log origin / gh api over gh pr view during the post-push window.

Distinct from the working-tree-HEAD-vs-commit-under-review guard (code-review workflow): THAT is about local HEAD being at a different commit than the one under review; THIS is about gh pr view's OWN cache lagging the remote state.

Discovered 2026-07-19 (Choose Your Own AI Policy PR #270): after pushing the coherence pass commit, gh pr view showed the previous 'blog: add banner image' as latest and only 3 commits, prompting a moment of doubt; 'git ls-remote origin blog/choose-your-own-ai-policy' confirmed the coherence commit (fe4f5b7) was in fact on the remote. Insert this verification between Full Workflow step 7 (push + open PR) and step 8 (monitor CI to green) whenever push-then-verify is required.

## Verify all commits landed on main after PR merge (mid-stream merge trap)

- A PR showing MERGED does NOT prove every commit you pushed actually reached main. Failure mode (PR #270, 2026-07-19): the PR was merged (via rebase) BETWEEN pushes — the merge snapshot took the commits pushed up to that point (initial post, tags/summary, banner), but three later commits pushed AFTER the merge trigger (stacked-PR prose weave, AGENTS.md move, coherence pass) landed on the BRANCH only, never on main. The PR badge said MERGED; main was missing the latest work.

TRIGGER: you are actively pushing commits to an open blog PR (e.g. during a re-edit / coherence-pass / batch-polish cycle) and the PR gets merged — by you in a prior turn, by auto-merge, by the user, or by a collaborator — before you have pushed ALL of your commits.

ROOT CAUSE: GitHub's merge takes a point-in-time snapshot of the branch at merge time. It does NOT follow the branch tip forward. Commits pushed after the merge trigger go to the branch ref, but the merged state on main is already locked in at the snapshot.

VERIFICATION (run after every blog PR merge, before declaring 'on main'):
  git fetch origin main
  git log origin/main --oneline --grep='blog' | head -20      # what's on main
  git log origin/<branch> --oneline | head -20                # what's on the branch
  # OR if the branch was deleted by --delete-branch:
  git log <your-local-branch> --oneline | head -20

Compare the two lists. Any commit on the branch that is NOT on origin/main (after fetch) is missing and needs a follow-up PR.

RECOVERY: open a follow-up PR from the same branch (if it still exists on the remote) or from a new branch off main with the missing commits cherry-picked. The follow-up PR's diff will show only the missing commits because main now contains the earlier ones. Use the same 'blog: <Post Title> [YYYY-MM-DD]' title format; mention 'follow-up to #<original-PR>' in the PR body.

This is DISTINCT from the 'Verify push landed on PR (gh pr view cache lag)' section: that section covers whether a single push reached the remote BRANCH; THIS section covers whether all your commits reached MAIN after the merge. Both checks belong in the post-merge workflow — verify push-then-cache-lag first, then verify merge-completeness.

Sibling to memory entry on PR-merge-completeness verification and to memory #4 (general verification discipline: 'don\'t trust surface done').

## Follow-up PR after rebase-merge: must rebase before opening

- The existing RECOVERY note ('open a follow-up PR from the same branch... The follow-up PR's diff will show only the missing commits because main now contains the earlier ones') is WRONG for rebase-merge-only repos (which this repo is — see Merging the PR section).

SYMPTOM (PR #271, 2026-07-19, follow-up to rebase-merged PR #270): opening a follow-up PR from the same branch yields mergeStateStatus CONFLICTING / DIRTY, NOT a clean diff of only the missing commits. The PR badge may say UNSTABLE while CI runs, but the mergeable state is the giveaway.

ROOT CAUSE: GitHub's rebase-merge replays the branch commits onto main with NEW commit hashes (the rebased copies live on main with different SHAs than the originals on the branch). When the follow-up PR compares branch vs main, it sees ALL original branch commits as 'new' (none of their hashes match anything on main), and they textually conflict with the rebased copies already on main.

FIX (the actual recovery procedure for a rebase-merge-only repo):
  git fetch origin main
  git rebase origin/main
    # git SKIPS the already-applied commits via patch-id detection
    # ('Commit N is empty — already applied, skipping') and replays
    # only the commits that did NOT make it into the first merge.
  git push --force-with-lease origin <branch>
  # PR now reports MERGEABLE / CLEAN. Merge via rebase as usual.

KEY INSIGHT: 'git rebase origin/main' on a branch whose earlier commits were rebase-merged onto main is a NO-OP for the already-applied commits and a REPLAY for the rest. Do NOT manually cherry-pick or rewrite; let git's patch-id machinery do the dedup. Force-push is REQUIRED because the branch history changes.

PREREQUISITE: a clean working tree (commit or stash unrelated WIP first). If the branch has unrelated uncommitted files (video-editor fixes, AGENTS.md edits), park them on a separate branch before rebasing, per memory #113 (split unrelated concerns into separate PRs from origin/main).

This corrects the prior RECOVERY paragraph in this section. Replace the claim 'The follow-up PR's diff will show only the missing commits because main now contains the earlier ones' with the rebase-and-force-push procedure above.

## Smallest-diff edits to published posts

- ## Smallest-diff edits to published posts

- When editing an EXISTING published blog post to add a new point or paragraph (e.g. adding a maintainer tip to a SciPy recap), prefer the SMALLEST POSSIBLE DIFF. Weave the new content into the existing paragraph structure — extend an existing paragraph, or insert ONE short bridge paragraph between two existing paragraphs — rather than restructuring sections, moving content, or reflowing the surrounding prose.

Procedure:
1. Identify the natural INSERTION POINT inside the existing flow (between two paragraphs, or appended to a thematically-related paragraph).
2. Write the new content to FIT that slot — match the surrounding paragraphs' length, voice, and transition style. A bridge paragraph is typically 2-4 sentences.
3. Do NOT reorganize the section, rename headings, or move existing sentences to 'make room'. The diff should show additions, not moves.
4. Apply the same edit to BOTH the vault draft and contents.lr (per the bidirectional-sync check, the vault draft is source of truth — edit there first, then rebuild).

Stated 2026-07-19 (SciPy 2026 blog, adding a design-docs/HLD/LLD/EARS tip): the assistant explicitly reasoned 'Given the user wants smallest diff, let me try to weave it into the end of paragraph 3 or as a very short bridge between paragraphs 3 and 4' rather than reorganizing the section. This is the EDIT-SCOPE facet of the minimal-disruption theme that also covers keeping pub_date fixed (Evolving/living-document posts section) and honesty-weaving updates — those govern WHAT stays constant; THIS governs HOW MUCH prose changes.

Apply by default to any revision of a published/shipped post. Larger restructuring (splitting sections, reordering points, rewriting intros) should be flagged to the user as a bigger change and confirmed before applying.

## Interactive HTML elements via iframe

- ## Interactive HTML elements via iframe

When a blog post needs an INTERACTIVE element (an animated demo, a slide deck, a carousel, a custom visualization), the site convention is to create a STANDALONE self-contained .html file in the SAME directory as the post's contents.lr, then embed it in the body markdown via an <iframe> tag.

### Pattern

1. Create the interactive HTML file alongside contents.lr:
   content/blog/<slug>/interactive-thing.html  (self-contained: inline CSS + JS, no external build step)
2. Reference it in the body: markdown (inline HTML is allowed in the body field):
   <iframe src="interactive-thing.html" width="100%" height="500vh" frameborder="0" scrolling="no"></iframe>

### Existing examples in the repo

- content/blog/arc-browser-first-impressions/arc-screenshots.html — embedded via <iframe src="arc-screenshots.html" width="100%" height="500vh"></iframe>
- content/blog/how-to-automate-the-creation-of-google-docs-with-python/slides.html — embedded via <iframe src="slides.html" width="100%" height="500vh" frameborder="0" scrolling="no"></iframe>

Both .html files sit next to contents.lr in the same content/blog/<slug>/ directory and are self-contained (no external dependencies).

### External embeds also use iframe (different use case)

The same <iframe> tag embeds EXTERNAL content (not local files):
- Excalidraw read-only links: <iframe src="https://link.excalidraw.com/readonly/<id>" width="100%" height="600px" style="border: none;"></iframe>
- YouTube embeds: <iframe width="560" height="315" src="https://www.youtube.com/embed/<id>" ...></iframe>
- LinkedIn embeds, figshare widgets, etc.

The LOCAL-file iframe is the one to use when you need custom interactive JS/CSS that cannot be expressed as an external embed.

### Gotcha: height attribute

The existing posts use height="500vh" in the HTML attribute. Note that 'vh' is a CSS unit, not a standard HTML attribute unit, so browsers may interpret it inconsistently. If the interactive element has a known pixel height, prefer height="<N>px" or a concrete pixel number. Test the rendered output on the deploy preview before merging.

### Discovery

Found by grepping the repo for <iframe in content/blog/**/*.lr. This is the established convention for interactive blog elements; reach for it before proposing a JS framework, a build step, or a separate subdirectory.

### Match interactivity to post type (2026-08-30)

- TRIGGER (standing): when the user says a post 'reads like a wall of text' or asks to 'make it more interactive', treat adding interactive elements as the DEFAULT expectation, in the spirit of the recent posts. Find precedent fast: grep content/blog for <iframe to list precedent posts and their widget counts (e.g. ai-turbocharges-learning, quantum-ml, going-bayesian).
- BEFORE choosing widget formats, determine the post's TYPE. Teaching/concept posts justify pedagogical widgets (build-before-you-read priming MCQs, concept demos). Marketing/positioning-type posts do NOT need a build-before-you-read widget - their interactive pieces should CARRY the post's argument or engagement, not teach a concept. Stated 2026-08-30: 'this is not a blog post that is teaching a concept, it's a bit more of a marketing-type blog post, we don't necessarily have to do a build-before-you-read widget here.'
- Once the widget plan is set, batch-build with one parallel widget-builder subagent per widget file, and apply the website-standalone-html-widget house style to each.
- **Match the visual format to what the explanation must show** (2026-09-26, seems post): a step-by-step mechanism walkthrough (how a parser tokenizes one line, how state transforms) wants a CUSTOM ANIMATED HTML WIDGET in the house style — an annotated line that animates what gets parsed — not a static Mermaid diagram. Mermaid suits static pipeline/architecture flow (boxes and arrows); when Eric first says "add a diagram" and the thing being explained is a process that unfolds on one input, propose the animated widget. Correction: "maybe mermaid is wrong. just custom html/js/css in there that animates how the seems parser is working?"
- ### Strict house style for interactive widgets (mandatory)

When creating a NEW interactive .html widget for a blog post, apply this exact design system. The user specified it VERBATIM, labeled "STRICT HOUSE STYLE (mandatory — copy this exactly)", across 5 separate widget requests in one session (learning-with-ai-survey, 07-27). Treat it as the canonical house style for new widgets unless the user overrides.

**Document shell:**
- Standalone `<!DOCTYPE html>` document, inline `<style>` and inline `<script>`. NO external CSS, NO external JS, NO libraries (no Tailwind, no React, no D3 — vanilla only).

**Global reset + body:**
```css
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: monospace; background: #fff; padding: 16px; color: #333; }
```

**Card container (the widget's outer box):**
```css
.card { /* or equivalent outer wrapper */
  background: #fafafa;
  border: 1px solid #ddd;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  max-width: 640px;
  margin: 0 auto;
  padding: 18px;
}
```

**Accent palette (use these exact hex values):**
- Primary blue: `#0066cc`
- Green text: `#1a9e5b` on green background: `#e7f6ee` (correct/success states)
- Red text: `#e85d3a` on red background: `#fdecea` (incorrect/error states)
- Grays for neutral/secondary text and borders.

**Typography:**
- Font sizes 12-15px, tight line-height.

**Iframe auto-resize:** widgets must resize their host iframe to fit content. Implement a `setHeight()` that measures `document.body.scrollHeight` (or the card's height) and posts it to the parent via `window.parent.postMessage({ type: 'setHeight', height: N }, '*')`, called on load and on any state change that changes height. (The existing iframe-embed examples in the repo use a fixed `height="500vh"` attribute; NEW widgets the user authors use the dynamic postMessage pattern so the iframe never shows a scrollbar or dead space.)

**Why mandatory:** these widgets are siblings of each other within and across posts. A consistent visual language (monospace, the #fafafa card, the blue/green/red palette) makes a multi-widget article read as one coherent interactive piece, and lets the user reason about "the widget style" as a single reusable template rather than re-specifying CSS per widget.
## Local testing on the Lektor dev server

- ## Local testing on the Lektor dev server (localhost:5959)

The Lektor dev server (`lektor server`) runs from the MAIN checkout (`~/github/website`, observed on port 5959) and serves whatever files are on disk in that working tree. The main checkout is on `main`. When you develop a blog post via the worktree workflow and then remove the worktree, the branch's files are NOT in the main checkout — so localhost:5959 returns 404 for the post, OR renders a STALE partial copy.

### To test a post locally after the worktree is removed

Extract the post's files from the PR branch into the main checkout:

```bash
git show blog/<slug>:content/blog/<slug>/contents.lr > ~/github/website/content/blog/<slug>/contents.lr
# Plus EVERY companion file in the post dir (interactive demos, images):
git show blog/<slug>:content/blog/<slug>/outlier-demo.html > ~/github/website/content/blog/<slug>/outlier-demo.html
```

Then refresh `localhost:5959/blog/YYYY/M/d/<slug>/`.

```` is a shell glob for the post directory. Use `git show blog/<slug>:content/blog/<slug>/` to list the branch's files if you don't know the companion filenames.

### GOTCHA: stale partial copies (the smoking gun)

If you wrote an initial `contents.lr` to the main checkout BEFORE switching to the worktree workflow, the main checkout has a STALE `contents.lr` AND is likely MISSING companion files (interactive .html demos, .webp images) that were only created inside the worktree. Symptoms: a 404 on a demo file, or the post body rendering an OLD draft version locally even though the PR has the latest. Fix: extract ALL of the post's files from the branch and overwrite the stale copies, not just contents.lr.

### These extracted files are local-test-only

They are UNTRACKED / MODIFIED in the main checkout on `main`. Do NOT commit them to `main` — the canonical versions live on the PR branch. Clean them up (`git checkout -- <file>` / `rm`) when done for a tidy main, or leave them for continued local iteration.

This is the LOCAL counterpart to the deploy-preview URL verification: the deploy preview tests the PR branch's rendered output; localhost:5959 tests your local working copy. They serve different purposes — reach for localhost when you want fast iteration without waiting on CI/deploy previews, especially for interactive elements where you need to actually click/drag in the browser.

## Live-preview server during interactive edit sessions (2026-09-26)

When Eric is reviewing edits in real time (a live editing session on the session worktree), START a Lektor dev server for that worktree on a random non-conflicting port and hand him the post URL BEFORE continuing edits. Declarative spec: "run web server on a rando port for me, and give it to me. then we'll continue. i wanna see changes live locally." Lektor auto-rebuilds on save, so after each edit, point at the URL again instead of describing the change.

This refines — does not contradict — the 2026-07-27 gotcha below about competing servers: that banned a SURPRISE worktree server fighting the main checkout's server for port 5959 while the user watched the main one. An EXPLICIT "run a server for me" commission means start one, on a random port so nothing conflicts. In an orca/attached worktree session the main checkout's 5959 server cannot serve the worktree's files anyway.

Example: `lektor server -p 31120` from the session worktree, backgrounded to survive the tool call (see the opencode-bash-background-process-survival skill), then hand over `http://127.0.0.1:<port>/blog/YYYY/M/d/<slug>/`.

Discovered 2026-07-24 (going-bayesian-automates-data-analysis post: main checkout had a stale contents.lr from an initial draft and was missing outlier-demo.html, so the interactive outlier demo rendered on the deploy preview but 404'd locally).

- Do NOT start a competing `lektor server` in a worktree. The user's main-checkout dev server already owns port 5959; a worktree-started server either fails to bind (port in use) or serves a host the user is not looking at, so the user reports 'I don't see the post at localhost:5959'. The fix is NEVER to start a second dev server — instead copy the post's files into the main checkout (or consolidate to the main worktree per that section). Discovered 2026-07-27 (learning-with-ai-survey): the assistant had backgrounded a worktree dev server on 5959 while the user's main-checkout server was already running there, leading the user to ask 'is this on a different worktree? can we bring it into this main worktree?'
## Runnable marimo notebook for reproducibility

- ## Runnable marimo notebook for reproducibility (alongside the iframe demo)

When a blog post ships an INTERACTIVE demo or analytical content, provide a RUNNABLE marimo notebook ALONGSIDE the post as a reproducibility artifact, in addition to (not instead of) the in-browser iframe demo.

### Convention

1. Place the .py notebook in the SAME directory as contents.lr:
   content/blog/<slug>/<demo-name>.py
   The notebook must use PEP 723 inline script metadata so deps auto-resolve under --sandbox, AND it must be a valid marimo notebook (starts with 'import marimo' + the marimo App scaffold).
2. Give readers the EXACT command to reproduce, in the post body:
   uvx marimo edit --sandbox --no-token <base-url>/<demo-name>.py
   - --sandbox: runs in an isolated uv-managed env (deps resolved from the notebook's PEP 723 metadata, no global install).
   - --no-token: no auth token needed, so a reader can just paste and run.

### This is DISTINCT from the embedded iframe demo

- The iframe .html demo (see 'Interactive HTML elements via iframe') is IN-BROWSER interactivity: the reader interacts without installing anything.
- The runnable marimo notebook is for the reader who wants to clone/re-run the FULL analysis LOCALLY and tweak it.

Provide BOTH. Stated 2026-07-24 (going-bayesian-automates-data-analysis): 'Fix the JS demo but also give me the marimo notebook in the same folder, so that we can give readers a uvx marimo edit --sandbox --no-token <base-url>/<some-notebook-name>.py as a command to reproduce the thing.'

### Do NOT offer a false either/or

When an interactive demo is broken or incomplete, do NOT propose degrading it to a STATIC SCREENSHOT as a substitute. The user wants interactivity PRESERVED. Fix the in-browser demo AND ship the runnable notebook; the static screenshot is not an acceptable replacement for either.

Pairs with memories #1530-#1533 (the blog-interactivity cluster: marimo WASM embedding, vanilla-JS-vs-WASM decision criterion) and #1537 (runnable-notebook reproducibility convention).

## Inlining interactive HTML directly into contents.lr

- ## Inlining interactive HTML directly into contents.lr (PREFERRED over iframe)

For NEW self-contained interactive demos (custom JS/CSS visualizations, click-to-exclude widgets, animated SVG demos), prefer INLINING the HTML directly into the body markdown over creating a separate .html file + <iframe>. Stated 2026-07-24 (going-bayesian-automates-data-analysis, outlier-dance demo): 'finally, it's an iframe? can we just embed the HTML directly? .lr files support that, the html in markdown gets passed through to rendering.'

Lektor's markdown processor passes raw HTML blocks through to rendering, INCLUDING <script> and <style> tags, so a self-contained demo works inline without the iframe indirection.

### Implementation caveats

1. Precede and follow the HTML block with BLANK LINES so markdown treats it as a contiguous HTML block (memory #120, the raw-HTML-block contiguity rule). Without surrounding blank lines, markdown may interleave the HTML with paragraph parsing.
2. Ensure the inlined HTML/CSS/JS contains NO standalone '---' lines. These are Lektor field separators and corrupt parsing exactly like a markdown horizontal rule written as '---' (see the 'Horizontal rules in the body corrupt field parsing' note above). If the demo's CSS/JS legitimately needs '---' (rare), reflow it.
3. Keep only <style> + body markup + <script>. Strip the <!DOCTYPE>, <html>, <head> wrappers — the page already provides those.
4. SCOPE all CSS selectors under a unique container. This is the #1 gotcha when converting an iframe widget to inline: the widget CSS was designed for its own document context, but inline it cascades across the ENTIRE host page. Two specific hazards:
   - GLOBAL RESETS kill the site layout: a bare `* { margin: 0; padding: 0; box-sizing: border-box; }` in the widget <style> resets EVERY element on the blog page, breaking terminal.css spacing, headings, and nav. Strip `*`, `html`, and `body` selectors entirely. The iframe provided its own document context; inline, the page's existing styles apply.
   - GENERIC CLASS NAMES collide: `.app`, `.titlebar`, `.dot`, `.toolbar`, `.btn`, `.meta` are common in widget CSS and may already be styled by terminal.css or the site theme. Wrap the entire demo in a container div with a unique ID/class (e.g. `<div id="outlier-dance-widget">`) and prefix every CSS selector with that container (`#outlier-dance-widget .dot { ... }`). This namespaces the widget without rewriting class names in the HTML/JS.

   Failure to scope produces a subtly broken page: the demo works, but the surrounding blog layout is corrupted (missing margins, wrong fonts, collapsed spacing). The deploy preview is the ground truth. Check the FULL page renders correctly, not just the demo region.

### Trade-off

The .lr file gets long and harder to read in source (~hundreds of lines for a rich demo). The benefit: no separate file to keep in sync, no iframe height/scrolling tuning, and the demo renders inline with the prose flow. Accept the longer .lr for the simplicity of a single source file.

### When to STILL use an iframe (the legacy/external cases)

The <iframe> pattern documented in 'Interactive HTML elements via iframe' above remains valid for:
- EXTERNAL embeds (YouTube, Excalidraw readonly links, figshare widgets, LinkedIn embeds) — these are not local files.
- Legacy posts already using a local-file iframe (arc-browser-first-impressions, how-to-automate-the-creation-of-google-docs) — do not rewrite them.
- When the demo is so large that inlining would make the .lr unmaintainable (judgment call; prefer inlining unless the user says otherwise).

For any NEW self-contained local demo, default to inlining. Insert this section BEFORE 'Interactive HTML elements via iframe' so the inlining preference is encountered first.

## inline-html-style-block-mangling

- GOTCHA (companion to caveat #4 CSS scoping): when you scope CSS under a container ID (the recommended pattern from caveat #4, e.g. '#outlier-dance-widget .curve { ... }'), the <style> block MUST be placed OUTSIDE/BEFORE the container <div>, separated from surrounding markdown by BLANK LINES. Do NOT nest <style> INSIDE the demo <div>.

ROOT CAUSE: Lektor's markdown processor interprets a leading '#' in CSS selectors (like '#ta * { margin: 0; ...') as a markdown H1 heading marker. When the <style> sits inside the container div, the markdown parser runs over its contents and mangles each '#'-prefixed rule into an <h1> tag. Symptom in rendered HTML: '<h1 id="ta-margin-0-padding-0-box-sizing-border-box">ta * { margin: 0; padding: 0; box-sizing: border-box; }</h1>' instead of raw CSS inside a <style> tag. The '#' becomes the heading, the selector body becomes the heading text, and the whole rule is destroyed.

FIX: structure the inline demo as THREE separate raw-HTML blocks, each surrounded by blank lines, in this order:

    (blank line)
    <style>
    #ta * { margin: 0; padding: 0; box-sizing: border-box; }
    #ta .curve { ... }
    ...
    </style>
    (blank line)
    <div id="ta">
      ...demo markup...
    </div>
    (blank line)
    <script>
      ...demo JS...
    </script>
    (blank line)

The <style> lives OUTSIDE the <div id="ta"> so the markdown parser treats the whole <style>...</style> as one raw-HTML block (no markdown interpretation of its contents). The blank lines are what make each tag a distinct raw-HTML block per CommonMark/Lektor rules (ties back to caveat #1 and memory #120).

VERIFICATION: after editing, curl the local dev server (http://localhost:5959/blog/.../<slug>/) and grep the rendered HTML for a stray '<h1 id="<container-id>' — if you see your container id buried in an <h1> id attribute, the <style> is still being mangled and needs to move outside the div. A correctly-rendered scoped <style> appears verbatim as CSS inside a <style> tag.

NOTE on inconsistent reproduction: a larger/longer HTML block may sometimes render correctly even with <style> nested inside the div (the parser switches to raw-HTML mode for sufficiently large contiguous blocks). Do NOT rely on this — it is fragile and content-dependent. Always place <style> outside the container div; it is the reliable fix regardless of block size. Discovered 2026-07-24 on the 'going-bayesian-automates-data-analysis' post (#ta widget mangled into <h1> tags; #od widget happened to render but for unverified reasons — moved #ta's <style> outside to fix reliably).

## inline-html-multi-block-merging-failures

- ## inline-html-multi-block-merging-failures (the three-block fix is NECESSARY but NOT SUFFICIENT)

The three-block structure from 'inline-html-style-block-mangling' above (style outside div + blank lines around each block) fixes the h1-mangling but does NOT fix ALL failure modes. For a COMPLEX multi-block demo (style + nested divs + script), Lektor/Mistune STILL mangles the output in additional ways, EVEN when every documented mitigation is applied:

1. **Closing </div> merged with next paragraph**: renders as '</div><p>Are you really going...' on one line — the markdown paragraph parser runs into the closing tag.
2. **Top-level <div> wrapped in <p>**: renders as '<p><div id="ta"></p>' — invalid HTML that breaks the layout. The container div gets paragraph-wrapped.
3. **Inner HTML parsed as markdown**: the contents of a container <div> get wrapped in <p> tags, <div class="ta-bar"> becomes '<p><div class="ta-bar"></p>', and closing tags nest incorrectly.
4. **</script> merged with next paragraph**: renders as '</script><p>The second problem...' — same merging as failure mode #1 but on the script close.

ROOT CAUSE: Mistune does NOT reliably pass raw HTML through for multi-block demos. It interleaves paragraph parsing into HTML block boundaries, especially when multiple raw-HTML blocks (style, div, script) sit in sequence. The blank-line separation required by CommonMark raw-HTML-block rules is necessary but Mistune's implementation does not honor it consistently across consecutive blocks.

FIX (the reliable approach for complex demos): use an iframe. Move the demo to a standalone .html file in the same content/blog/<slug>/ directory and embed it:

    <iframe src="outlier-dance.html" width="100%" height="600" frameborder="0" scrolling="no"></iframe>

The iframe isolates HTML/CSS/JS completely from the markdown processor — no merging, no paragraph wrapping, no h1 mangling. This SUPERSEDES the 'prefer inlining over iframe' default (section 'Inlining interactive HTML directly into contents.lr') for any demo with style + nested divs + script. Inlining may still work for SIMPLE single-block HTML (a bare <div> with inline styles, no <style> or <script>), but for complex interactive widgets, the iframe is the reliable choice.

VERIFICATION: after switching to iframe, curl the local dev server and confirm the demo renders inside the iframe and no surrounding paragraphs are mangled (grep for '</div><p>' or '</script><p>' patterns in the rendered HTML — they should not appear).

Discovered 2026-07-24 (going-bayesian-automates-data-analysis post, continuation of the earlier h1-mangling debug: the #od and #ta widgets still rendered incorrectly after applying the three-block + blank-line fix from the earlier gotcha; the assistant concluded iframes are the guaranteed-reliable approach).

## External citation verification (4th pre-publish review axis)

- Extends the Pre-publish Review (parallel read-only subagents) section. The existing three axes are: (1) privacy/confidentiality, (2) style/voice (write-like-eric), (3) technical accuracy (INTERNAL: cross-check against README.md, source code, data files). For SURVEY / LITERATURE-REVIEW / SOURCED posts that cite EXTERNAL authors (blog posts, papers, talks, docs), add a FOURTH axis:

4. **External citation verification**: fetch each cited URL and confirm the post's characterization matches what the source actually says. Report mismatches as "[claim in post] vs [what the source says]" with the URL. Pay special attention to: (a) dates (was the talk really July 2026? was the blog post published when you claim?), (b) attributions (did the RIGHT person say this, or did you conflate two authors?), (c) paraphrases (does the source actually make the argument you attribute to it, or are you overclaiming?), (d) quoted stats (do the numbers match — e.g. '60 million Copilot reviews', '>1 in 5 involve an agent'). This is DISTINCT from axis #3 (internal accuracy): #3 verifies your OWN project's claims against repo files; #4 verifies your characterization of OTHERS' work against the live source. Load-bearing for any post whose thesis leans on external sources.

Prompt template for the fact-checker subagent (mirrors the editorial-reviewer template):
"You are a READ-ONLY fact-checker. Do NOT edit any files. Read the blog post at <exact path>. Verify every factual attribution and quoted claim in the body against the cited source. For each, fetch the linked URL and confirm the post's claim matches the source. Report mismatches as: [claim in post] vs [what the source actually says], with the URL. Specific claims to verify (the load-bearing ones): <numbered list of claims with URLs>."

Not every post needs all four axes — pick the relevant ones. A dev-update post needs #1/#2/#3 (internal accuracy). A survey/lit-review post needs #1/#2/#4 (external citations). A post that dogfoods your own system AND cites others needs all four. Apply this DURING authoring too (verify attributions as you write, not just at review time) — this is the blog-post operationalization of the global memory rule: do NOT fabricate sources, URLs, or dates; verify every factual claim against a real fetched source.

Discovered 2026-07-24 (understanding-zeitgeist-ai-code-review post: a survey of ~10 external sources where the fact-checker subagent was dispatched alongside the editorial reviewer to verify each citation).

## Interactive HTML widgets (iframes)

- ## Interactive HTML Widgets (iframes)

The blog supports interactive HTML widgets embedded directly in the body. This is
an established pattern across multiple posts (going-bayesian-automates-data-analysis
has 5 widgets; the AI-PR-review post has 4).

### Convention

1. **Widget files live alongside contents.lr** in the same blog post directory:
   content/blog/<slug>/<widget-name>.html. Standalone HTML+JS+CSS, self-contained
   (inline everything; no build step / external deps).

2. **Embed via iframe** in the body markdown using this exact pattern (consistent
   across ALL existing widgets):

   ```html
   <iframe src="widget-name.html" width="100%" scrolling="no"
     style="border: 1px solid #ddd; border-radius: 6px;"
     onload="this.style.height=this.contentDocument.body.scrollHeight+'px'">
   </iframe>
   ```

   The onload auto-resize handler is CRITICAL: without it the iframe gets a fixed
   default height and content is cut off or shows scrollbars. It reads the
   widget's rendered body height and sets the iframe to match after load.

3. **Relative path only**: src="widget-name.html" (same dir as the post), never
   a full URL.

### Verification after inserting an iframe

1. Widget JS has no syntax errors (open in browser dev console, or extract and
   node --check the script block).
2. contents.lr separator count unchanged (6 for the standard field set). The
   iframe tag itself is fine (HTML, not a standalone --- line), but any prose
   edits around it must NOT introduce a --- markdown horizontal rule in the body
   (use *** or ___ instead — see the HR gotcha above).
3. No body --- HR introduced by surrounding prose edits.

### Placement strategy

Insert widgets at natural comprehension checkpoints throughout the post, not just
at the end: after a thesis paragraph the widget dramatizes (with a one-line
lead-in bridging prose to widget), in a Techniques section demonstrating a
just-described technique, and comprehension-check quizzes in a final section.
Breaking up walls of text is the point — a widget every few sections keeps the
reader active.

## dual-server-multi-worktree-preview

- QUALIFICATION to the 'NEVER start a second dev server' rule above (line ~622): that rule applies when the user ALREADY has a main-checkout dev server running and you start a competing one on the SAME port (5959) without asking. There is a legitimate, USER-CONFIRMED exception: previewing TWO posts from TWO different branches/worktrees simultaneously on DIFFERENT ports. Technique (CONFIRMED 2026-07-31, quantum-ML post A on port 5001 + mRNA post B on port 5002, both served HTTP 200 with correct titles and iframe widgets rendering):

1. Worktrees have NO .pixi/env directory (only the MAIN checkout at ~/github/website/ does), so 'pixi run lektor server' FAILS from a worktree. INSTEAD invoke the SHARED Lektor binary directly with CWD set to each worktree: /Users/ericmjl/github/website/.pixi/envs/default/bin/lektor server -p <port>. The binary's shebang resolves to the env's python so deps resolve regardless of CWD; Lektor finds its project file (.lektorproject) via CWD.

2. Lektor's DEFAULT port 5000 is OCCUPIED by macOS ControlCenter (AirPlay Receiver) on modern macOS — always pass an explicit -p with an alternate port (5001, 5002, ...). Port 5959 (the user's habitual main-checkout port) is also fine if free.

3. To preview TWO posts in two worktrees at once, start TWO servers on different ports (one per worktree), backgrounded via nohup with logs redirected (opencode-bash-tool-long-running-jobs): CWD=<worktreeA> nohup <lektor-bin> server -p 5001 > /tmp/lektor-dev/post-a.log 2>&1 & ; CWD=<worktreeB> nohup <lektor-bin> server -p 5002 > /tmp/lektor-dev/post-b.log 2>&1 &.

4. Verify each post serves (curl the URL, grep for HTTP 200 + the post <title>) BEFORE handing URLs to the user. Post URLs are date-based: http://localhost:<port>/blog/YYYY/M/d/<slug>/.

WHEN this exception applies vs the line-622 rule: the user EXPLICITLY asks you to start the server ('can you start the server for me?') AND wants to compare/preview two (or more) branches side by side AND there is no existing main-checkout server to collide with. If the user already has localhost:5959 running on main, do NOT start a second server there — use the main-checkout consolidation path instead. Lifecycle note (memory #62): the user generally prefers to OWN long-running server lifecycles, but EXPLICITLY requested starting this one — when the user asks you to start it, start it (and hand back PIDs + how to kill); do NOT narrate the plan instead of executing (the narration-without-action failure mode).

## Locating the target post (user refers to it by topic)

- When the user refers to a post by TOPIC ('the quantum computing blog post') rather than slug, do NOT infer the location from worktree or branch NAMES — branch blog/mrna-quantum-sampler held the mRNA-structure post while the user meant quantum-ml-for-the-probabilistic-bayesian, and the main worktree sitting on an unrelated feature branch (feat/learn-anything-promo) was a red herring (2026-08-14). Batch these diagnostics in ONE tool block:

1. List worktrees, then grep 2-3 distinctive phrases from the user's description across ALL worktrees' content/blog/ (not just the main checkout).
2. Determine published vs draft: git fetch origin main && git log origin/main --oneline -- <path>, plus read pub_date from contents.lr. A post found committed in the MAIN worktree while it sits on an unrelated branch is almost always already published (feature branches carry main's history — its presence in git log for the current branch proves nothing about which branch authored it).
3. gh pr list to check for an open PR touching that path.

Then branch accordingly: published post -> new branch/worktree off origin/main + PR (see Smallest-diff edits to published posts); open PR -> the re-edit workflow (Updating an existing PR). Symptom of guessing wrong: turns burned reconciling branch names ('wait, branch is blog/mrna-quantum-sampler... but the grep found it in the main worktree?') before the grep settles it. Also grep the post's companion *.html widgets at this stage when revision feedback is technical (see memory on body/widget claim consistency).

## Minimal-touch precision corrections (reviewer-feedback fixes)

- - When fixing TECHNICAL/PRECISION errors in a published post based on reviewer feedback (colleague review, fact-check, reader correction), each fix must be MINIMAL-TOUCH: keep Eric's original sentences, change only the inaccurate words, add at most ONE short clause or sentence of framing. Do NOT rewrite the surrounding paragraph, even in Eric's voice. Paragraph-balloon replacements get rejected as 'too verbose' (quantum-ML post corrections, 2026-08-14: full-paragraph before/after proposals were sent back with 'more compact' as the requirement; the accepted redrafts kept the original sentences and changed only 'amplitude squared' -> 'squared magnitude |a|²', 'sign' -> 'phase', and added one circle-vs-sphere framing clause).
- This applies at the PROPOSAL stage: when the user asks for proposed changes ('propose some changes', 'I need some guidance on how we can change the blog post'), draft the before/after pairs as compact edits from the start. Show the original sentence with only the wrong bits changed, not a wholesale rewritten paragraph.
- Extends the 'Smallest-diff edits to published posts' section (which covers ADDING new content) to the CORRECTION case. Pairs with the bidirectional-sync check: apply the same compact edits to the vault draft (Blog Draft/<slug>.md) so it does not drift from contents.lr.

## Vault draft drift on smallest-diff edits (strong-drift branch)

- The smallest-diff step 'apply the same edit to BOTH the vault draft and contents.lr' assumes the two are in sync. For a LONG-PUBLISHED post they may not be: the vault draft can be a much earlier draft that predates a substantial rewrite (discovered 2026-08-14, quantum-ML post: the vault draft was still anchored on the mRNA-paper framing while the published post had been rewritten around interactive widgets — none of the edit target strings existed in the draft).

Procedure:
1. BEFORE attempting vault edits, grep the vault draft for one or two of the exact old strings. If present, apply the same edits to both files as usual.
2. If the strings are ABSENT, the draft has STRONGLY drifted (it is a different, earlier version — not just missing recent fixes). Do NOT force line-by-line edits, and do NOT unilaterally rewrite/forward-sync the entire vault draft; that is a bigger action than the correction being made.
3. Proceed with the contents.lr-only smallest-diff edit and FLAG the drift to the user in the final report, offering a vault-draft refresh as optional cleanup.

This is the strong-drift sibling of the Bidirectional sync check section: THAT covers minor accumulated drift (back-sync the vault draft first); THIS covers the draft being an earlier draft version, where syncing is a rewrite decision for the user.

Post-edit QA for smallest-diff contents.lr edits: verify no em dashes were introduced AND the standalone --- separator count is UNCHANGED from before the edit (6 for the standard field set) — a changed count means an edit corrupted a field boundary (extends the separator-count rule in the Lektor format section to post-edit verification).

- AMENDMENT to step 3 (2026-08-14, PR #288): flag the drift in the final report but do NOT phrase the flag as a closing a-or-b offer question ('happy to refresh it or archive it if you want') — that exact closing drew a 'Take action now' interrupt. Either execute the reversible default in the same turn (refresh the vault draft from the published contents.lr; non-destructive, the vault is git-tracked) and report what was done, or state the flag declaratively ('the vault draft is a stale earlier version; left untouched') so the report ends on a statement. Refresh-vs-archive is not a genuinely destructive choice, so asking is not licensed; execution-discipline's ask-only-for-destructive-ops rule outranks this section's earlier 'offering' wording.
## editing-posts-with-companion-widgets

- Editing posts that have companion widgets: when correcting technical or precision claims in a post with embedded interactive .html files (see Interactive HTML elements via iframe), the widgets PARAPHRASE the post claims and drift out of sync — sweep the widget files for the SAME claims in the SAME PR. Procedure (2026-08-14 quantum-ML post session): grep the corrected terminology (e.g. "squared amplitude") across BOTH contents.lr AND *.html in the post dir; also point the pre-commit read-only reviewer at the WHOLE post directory, not just contents.lr — the reviewer pass catches residuals the direct edit pass misses (check-understanding.html lines 116/270/283 carried stale framing after contents.lr was fixed). Body and widgets can flatly contradict each other; fix them together. Sibling of memory #1119.

## Pre-PR build verification (pixi lektor tasks)

- ## Pre-PR build verification (pixi lektor tasks)

Before committing/opening a PR for ANY new or edited Lektor content (blog posts, talks, teaching entries), verify the site builds. The task names live in pyproject.toml [tool.pixi.feature.lektor.tasks] — do not re-read pyproject.toml to rediscover them:

- 'pixi run build' = 'lektor build --output-path site/' with depends-on ['build-search-index'] (runs scripts/build_search_index.py first). This is the canonical build check from the MAIN checkout.
- 'pixi run website' = 'lektor server --port 5959 --host 0.0.0.0' (the dev server).
- The default pixi environment includes the lektor feature, so no -e flag is needed.

WORKTREE GOTCHA (same root cause as the dual-server-multi-worktree-preview section): 'pixi run build' FAILS from a sibling worktree because worktrees have no .pixi/env directory. To verify a worktree's content before commit, invoke the shared Lektor binary with CWD set to the worktree (analogue of the documented 'lektor server' pattern, verified for server 2026-07-31):

  cd ~/github/website-<slug> && ~/github/website/.pixi/envs/default/bin/lektor build --output-path /tmp/site-build-test

Alternatively, complete the commit/push/PR from the worktree and let website CI (Lektor site build, ~5-10 min) be the verification, per the Monitor-CI-to-green step. Discovered 2026-08-23 (talks-page session: assistant stalled re-deriving the task name from pyproject.toml and was interrupted before running anything).

BUILD-ARTIFACT LEAK (2026-08-23, talks PR #290): the verification build (either path above) emits `assets/static/js/build-lunr-index.js` and `assets/static/js/search-index-data.json` as UNTRACKED files — those two paths are NOT gitignored (only `site/` is). After a build, `git status` shows them as `??`; NEVER stage or commit them with a content PR. Stage only the intended content/ files, and rm the two artifacts when done with the worktree. Fresh-worktree tell: every untracked file in a brand-new worktree was generated by YOUR verification build, not pre-existing — so this is never a reason to widen a commit.

## Draft resolved -> run the publish path in the same turn (2026-08-30)

- When a vault blog draft is complete and the open content questions (title, naming, framing) have been answered by the user, do NOT end the turn with 'when you're happy I'll take it through the publish path' or a paragraph-by-paragraph walk-through offer. That exact closing drew a 'Take action now' interrupt (retreat-announcement post session, 2026-08-30). The publish path (worktree off origin/main -> contents.lr from the draft -> PR with the [YYYY-MM-DD] title format -> CI watch -> URL verification pass) is REVERSIBLE: the open PR is the review surface, not a commitment to publish. Run it in the same turn and end the report pointing at the PR; Eric reviews there and requests edits via the re-edit workflow.
- Paragraph-by-paragraph refinement, if wanted, happens against the open PR branch, not as a gate before opening the PR.
- Same execution-discipline root as the 2026-08-14 amendment above: refresh-vs-walk-through and draft-vs-PR are not genuinely destructive choices, so asking is not licensed; execution-discipline's ask-only-for-destructive-ops rule governs.


## Outbound link verification in drafts

- Before inserting ANY outbound link in a blog draft (book, product, reference, vendor), verify the URL resolves AND its content is confirmable: curl -sIL -o /dev/null -w '%{http_code} %{url_effective}' <url> and expect 200 on the FINAL url (a 301 chain that lands on a 404 or an unrelated page = dead). If every candidate source fails (404, 301-to-404, publisher page gone) or returns a 200 JS shell whose content curl cannot verify (Goodreads), ship the mention UNLINKED and surface the open item to Eric in the close-out report - never insert a guessed, remembered, or content-unverifiable URL just because the sentence reads better with a link. Instance 2026-08-30: *Who Not How* link - benjaminhardy.com book page 301-to-404, Hay House page 404, Goodreads HTTP 200 but JS-rendered body unverifiable via curl - shipped unlinked with the flag; Eric can supply a preferred URL later. This is DRAFT-TIME outbound-link hygiene, distinct from the Post-PR URL Verification section (which checks RENDERED site URLs against the deploy preview after merge).

## sync-verify-expected-h1-diff

- ### Expected-diff rule when verifying vault-draft ↔ contents.lr sync (2026-08-30)

When verifying the bidirectional sync by diffing the vault draft (Blog Draft/<slug>.md) against the contents.lr body, exactly ONE diff line is expected and correct: the H1. The vault draft carries  as its first line; the contents.lr body omits it because the title renders from the  field. So 'the only diff is the H1 line' = bodies are IN SYNC, by design — do not 'fix' it by adding an H1 to the body (that would double the title on the rendered page) or by stripping it from the vault draft. Any OTHER diff line = real drift; back-sync per the bidirectional-sync check before proceeding. This turns the sync verification into a countable check: diff, expect exactly the H1 hunk, everything else must be empty.

## Merge holds and the proactive preview-URL report (2026-08-30, PR #294)

- PROACTIVE URL REPORT: the CI-green report for an open blog PR must include BOTH the deploy-preview URLs (site root https://deploy-preview-<num>--ericmjl-github-io.netlify.app/ AND the deep link to the post) AND the final post URL it will have after merge (https://ericmjl.github.io/blog/YYYY/M/D/<slug>/, validated against pub_date) WITHOUT being asked. Eric had to ask 'give me the url please' (PR #294, 2026-08-30) — he reviews the deploy preview before authorizing merge, so the URLs ARE the review surface; deliver them together with the checks status in the same report.

MERGE HOLD PRECEDENCE: on an explicit hold ('don't merge to rebase just yet'), keep the PR open and DO NOT merge. A subsequent 'Take action now' RELEASES the hold (= merge authorization) only when BOTH hold: (a) the preview URLs + green checks were already delivered, and (b) your last report's pending question was the merge itself ('say merge and I'll rebase-merge') — the interrupt then directly answers that pending question, per the memory #1592 direct-answer test (this is the merge-domain analog of the send-freeze rule). If either condition fails (URLs not yet delivered, checks pending), 'Take action now' is the anti-stall interrupt: execute the remaining agent-executable steps (deliver URLs, finish verification) and KEEP the hold. Observed 2026-08-30 (PR #294): hold -> URL delivery + full URL-verification pass -> 'Take action now' arrived at slice end.

## Blogbot generation step (tags, summary, banner) - mandatory, and the print-only gotcha

- After creating contents.lr (Full Workflow step 4), the summary:, tags: fields and the banner image MUST be generated via the blogbot skill's scripts, never hand-written and never skipped. User correction 2026-08-30: 'use my blogging skills (you should have used them...)'. This correction has recurred at least 5 times (2026-03-08 banner, 2026-06-12, 2026-06-15, 2026-07-19, 2026-08-30) because the Full Workflow never named the step. Insert as step 4.5, BEFORE the pre-publish review: (1) source ~/github/website/.env (worktree env gotcha in the section below); (2) run the WORKTREE's copies of tags.py, summary.py, banner.py SEQUENTIALLY (shared llamabot sqlite DB, parallel runs hit UNIQUE constraint failures); (3) banner.py writes logo.webp into the post directory itself. PRINT-ONLY GOTCHA (2026-08-30, verified in tags.py source): tags.py and summary.py are GENERATORS ONLY. They print the result to console in a Rich Panel and exit. There is NO --write flag and NO code path that writes contents.lr. Do not hunt for a write flag; writing the output into contents.lr is the agent's MANUAL step: capture the console output, apply the scrubbing rules (summary: strip [URL] placeholders and 'Read on' tails; tags: fix typos, curate generic/duplicate tags, diff old-vs-new per memory #656), then write the fields into contents.lr yourself. Completion check is filesystem state, not script exit: summary: and tags: populated in contents.lr AND logo.webp present in content/blog/<slug>/ (memory #730's three-artifact checklist).

## Blogbot content generation is part of publishing (summary, tags, banner)

- Generating the summary, tags, and banner image via the blogbot skill is a MANDATORY part of the publication workflow - do NOT treat it as a separate later step the user must request. Stated 2026-08-30: 'use my blogging skills (you should have used them...) to generate the summary, tags, and banner image' after a publish cycle completed without them.

Why mandatory: the contents.lr 'summary:' and 'tags:' fields and the banner logo.webp in the post directory are ALL blogbot script outputs. A post committed with placeholder summary/tags and no logo.webp is incomplete.

Workflow placement: AFTER creating/updating contents.lr, BEFORE the commit step (both initial publish and re-edit rebuilds):

1. source ~/github/website/.env && uv run ~/github/website-<slug>/.agents/skills/blogbot/scripts/summary.py <slug>
   Then scrub per blogbot's summary rules: no [URL] placeholder, no 'Read on' tail.
2. uv run ~/github/website-<slug>/.agents/skills/blogbot/scripts/tags.py <slug>
   Verify every tag for typos before writing into contents.lr.
3. uv run ~/github/website-<slug>/.agents/skills/blogbot/scripts/banner.py <slug>
   Saves logo.webp into content/blog/<slug>/.
4. Write the generated summary + tags into contents.lr, then commit - logo.webp ships with the post.

Run the scripts SEQUENTIALLY, never in parallel (shared llamabot sqlite UNIQUE constraint - see blogbot skill), and always from the WORKTREE copy of the scripts (see 'Running blogbot scripts in a worktree' above).

## Widget strategy by post type (2026-08-30)

- When asked to make a post more interactive, first classify the post's PURPOSE - the widget strategy depends on it. Teaching/concept posts warrant concept widgets that break up dense material (recent house precedent: 4-7 per post). A MARKETING-type post is not teaching a concept and does not need heavy didactic widgets like a build-before-you-read demo (Eric 2026-08-30: 'we don't necessarily have to do a build-before-you-read widget here'). Interactive pieces should BE the post's arguments, not decoration. Survey sibling posts on the same topic for precedent (widget count + kind) before deciding what to build.

### Match interactivity to post type (2026-08-30)

- VAULT-DRAFT MIRROR ANNOTATION (2026-08-30, retreat post PR #294): the vault draft (Blog Draft/<slug>.md) cannot render local-file <iframe> embeds, so represent each widget there as an HTML comment annotation at the embed position, e.g. <!-- widget: cognitive-debt -->, using the SAME comment convention already present in that draft (grep the draft for 'widget:' annotations before adding; match them). Keep the real <iframe src="./widget-name.html" ... style="border: none;"> tags ONLY in contents.lr. After swapping annotations in, run markdownlint on the vault file and count-check the 'Linting: N files' line. Under the bidirectional-sync check, a widget embed in contents.lr maps to its annotation in the vault draft, not a literal iframe.

## Post-merge LIVE-site verification (final lifecycle step)

- After a blog PR is rebase-merged, the verification chain has ONE more step beyond the mid-stream merge-trap commit check: verify the LIVE site, not the preview. Procedure (validated 2026-08-30, PR #294): (1) confirm all branch commits landed on origin/main (mid-stream merge trap check) and the post-merge 'Build website' GitHub Actions run completes successfully (gh run list -R ericmjl/website --limit 3). (2) TIMING EXPECTATION: the live URL https://ericmjl.github.io/blog/YYYY/M/D/<slug>/ does NOT serve immediately after merge — the Pages build takes ~3 minutes. A non-200 in the first minutes after merge is EXPECTED propagation delay, NOT a broken deploy; do not re-merge, re-push, or open recovery PRs over it. Poll with spaced curls (30-60s apart) until 200. (3) When it 200s, re-run the Post-PR URL Verification pass against the LIVE URL: page renders, every in-body link resolves 200 on production, iframe widget .html files serve at the post path, banner present. The preview pass proves the BRANCH; this pass proves the DEPLOYMENT — both belong in the final report. (4) Report the live URL to Eric as the terminal deliverable (he typically proceeds directly to distribution, e.g. emailing collaborators about the post). Worktree/branch cleanup ordering is NOT part of this step — it is covered by the git-worktree-session-workflow skill (remove worktree before merge; gh pr merge --delete-branch fails on a branch checked out in a worktree while the MERGE still succeeds).

## Post-merge live-site verification (GitHub Pages build delay)

- After the merge executes, the publish is NOT verified until the LIVE https://ericmjl.github.io page is confirmed. Chain, in order: (1) merge-completeness - fetch origin/main and confirm every branch commit landed (match by SUBJECT, not SHA: rebase-merge replays commits onto main with NEW hashes); (2) the post-merge 'Build website' CI run goes green; (3) WAIT for the GitHub Pages build: the live URL does not update the moment CI finishes - observed ~3 minutes after merge (PR #294, 2026-08-30). A 404/timeout/stale content on the live URL in the first minutes is EXPECTED, not a failure; poll the post URL with spaced sleeps (~30s) until HTTP 200 before diagnosing anything. (4) Re-run the verification pass ON THE LIVE URL (same checks as the Post-PR URL Verification pass: page 200, every in-body link resolves, iframe widgets serve, banner + summary/tags present) and report the live URL as the ship-report lead (memory #2347). Distinct surfaces: the deploy preview verifies the PR BEFORE merge; the live site verifies the shipped result AFTER merge; run both - neither substitutes for the other.

## Adding a new talk from a YouTube URL (content/talks/)

- TRIGGER: user pastes a YouTube URL of a new talk and asks to add it to the talks page / open a PR. Recurring task (2026-08-22 talks spruce-up, 2026-08-23 talks PR #290, 2026-09-01 Canvas Chat SciPy 2026 PR #295) — re-derived every session until 2026-09-02, when four parallel-reviewer duplicates of this section were consolidated into this one.

Procedure:
1. FETCH METADATA FIRST via yt-dlp (title, event, upload date, description) — title, slug, summary, and pub_date derive from the video, not the user. If YouTube bot-checks yt-dlp, pass chrome cookies + deno (verified 2026-08-22).
2. DUPLICATE CHECK before creating anything: grep content/talks/ AND content/teaching/ for the video ID (e.g. V42Hq61JgcY). Skip the add if it already exists.
3. WRITE the page: a talk is a SINGLE contents.lr at content/talks/<kebab-case-slug>/ with fields: youtube (the URL, in a '#### youtube #### url' block), description, summary, pub_date (the video's upload date), category, visible. READ THE MOST RECENT existing talk's contents.lr and mirror its exact field set and order before writing (talks differ from blog posts: no author/body/twitter_handle/tags unless the newest talk has them). No body markdown, no blogbot generation step for talks.
4. COMMIT with the talks convention 'talks: <subject>' — NOT the blog 'blog:'/'post:'/'content:' prefixes. Example: 'talks: add canvas chat scipy 2026 talk'.
5. WORKTREE + PR: worktree from origin/main, pre-PR build verification per the pixi lektor tasks section, never stage the two build-artifact files (build-lunr-index.js, search-index-data.json — a talks edit regenerates the lunr search index too), push, open PR, monitor CI with spaced polls (site build 5-10 min), verify the talk page renders on the deploy preview (Post-PR URL Verification pass), remove the worktree once the PR is open.
6. MERGE GATE: talks are reader-facing live content — the blog-post MERGE GATE above applies equally. The open PR + both URLs (deploy preview + future live path) is the close-out deliverable; the merge runs ONLY on Eric's commission. AUTHORIZATION TEST (2026-09-01 PR #295: the report ended 'Say the word (with rebase, as usual) when you want it merged' and Eric replied with the generic 'Take action now.'): under the MERGE HOLD PRECEDENCE direct-answer test that interrupt IS merge authorization when BOTH (a) URLs + green checks were delivered and (b) the report's pending question was the merge itself (memory #1592) - execute the rebase merge in that same turn (see the 'Talk-PR close-out' section below). A generic 'Take action now' is NOT merge authorization only when the report did NOT end with the merge as its pending question, or URLs/checks were not yet delivered - then it means finish the remaining agent-executable steps first, not merge.

## Talk-PR close-out: 'Take action now' answering the merge question = merge authorization (2026-09-01, PR #295)

- Extends the 'Adding a new talk' sections and the 'Merge holds and the proactive preview-URL report (2026-08-30, PR #294)' rule to the NO-HOLD case. On a talk PR, 'open a PR for me' = PR-opening commission so the first close-out correctly DEFERS the merge - but the deferral is only the first half of the arc. The CI-green close-out report must deliver BOTH URLs (deploy preview + final live path), green-check status, and end with the merge as the named pending question ('Say the word (with rebase, as usual) when you want it merged!' - Canvas Chat SciPy 2026, PR #295). When Eric then replies 'Take action now.', BOTH conditions of the MERGE HOLD PRECEDENCE direct-answer test are met ((a) URLs + green checks delivered, (b) the report's pending question was the merge itself, per memory #1592), so the interrupt IS merge authorization: execute gh pr merge <num> -R ericmjl/website --rebase --delete-branch in that same turn, then run the post-merge live-site verification (GitHub Pages ~3 min delay; poll until 200). Do NOT misread the blog MERGE GATE clause 'a LATER generic Take action now is NOT merge authorization' as a no-op verdict here - that clause is scoped to PR #294-style EXPLICIT holds; with no hold claimed and the merge as the pending question, the direct-answer test governs. Still overridden by an explicit review claim ('let me look at it') or a hold ('don't merge yet'); and if URLs/checks were NOT yet delivered, 'Take action now' means finish those agent-executable steps first, not merge.

## MERGE GATE exhaustion refinement: 'Take action now' over an EMPTY hold-compatible set (2026-09-01, PR #295 Canvas Chat talk)

- The 2026-08-30 MERGE GATE says a generic 'Take action now' after the held-PR report is NOT merge authorization — it authorizes only the REMAINING hold-compatible close-out steps (URL delivery, verification pass, worktree removal). Refinement from PR #295: when the held-PR report has ALREADY completed every hold-compatible step (both URLs delivered, preview verification run, worktree removed — i.e. the report was fully gate-compliant), the remaining-steps set is EMPTY, and a subsequent bare 'Take action now.' has nothing left to authorize except the deferred merge. In that state TREAT IT AS THE MERGE COMMISSION: run the full merge sequence (gh pr view <n> mergeable check -> gh pr merge <n> -R ericmjl/website --rebase --delete-branch), then verify all commits landed on main (see the mid-stream merge trap section) and report the final live URL. Do NOT re-ask 'do you mean merge it?' — that re-litigates an interrupt which IS the answer (drew the interrupt 2026-09-01 despite a fully gate-compliant report). Corollary for the held-PR report's final line: end with completed facts plus the one remaining decision stated crisply and concretely ('merge is your call: gh pr merge 295 -R ericmjl/website --rebase --delete-branch, or click through the PR') rather than an interrogative 'Say the word when you want it merged!' — the interrogative permission-question form is what reliably draws the reflexive 'Take action now' interrupt (execution-discipline pattern).

## Concept-post internal-consistency checks (accuracy axis, no-source variant)

- ## Concept-post internal-consistency checks (accuracy axis for posts with NO external source of truth)

Axis #3 (technical accuracy) is written for dev-update posts that cross-check against README.md / source code / data files. For a CONCEPT / OPINION / math-flavored post (a decision-making trick, a mental model) there is no repo source of truth — the accuracy reviewer instead verifies INTERNAL consistency. Check set (funny-money-allocation review, 2026-09-11):

1. **Arithmetic self-consistency**: every number the post states must add up (author bets 70 of 1,000 keeping 930 -> 70+930=1,000; others bet 900 leaving 100 split across two options -> 900+100=1,000). Cross-check summary-field numbers against body numbers ('70-point bet' in summary vs 70 in body).

2. **Math-claim-vs-framing verification**: when the post makes a mathematical claim ('dividing each allocation by 1,000 yields a probability distribution over the outcomes'), verify it GIVEN THE POST'S OWN FRAMING, not an external textbook. Enumerate the outcome set exactly as the post defines it (e.g. three options plus 'keep some or all of your points' functioning as a catch-all fourth outcome absorbing every unallocated point), then check the claim against that enumeration (nonnegative values summing to exactly 1). A claim can be correct under the post's framing while looking wrong under a narrower reading — the framing IS the spec. State the framing explicitly in the verdict so the check is auditable.

3. **Cross-post entity-URL consistency**: when the post links a person/org who appeared in a PRIOR post (e.g. Clayton Springer's LinkedIn URL vs variance-explained/contents.lr), grep the prior post and diff the URL byte-for-byte. The prior published post is the canonical source; a divergent URL is a typo even if it looks plausible. (Sibling-comparison principle: the existing post is the implicit spec.)

4. **Internal date-based link structure**: target folder must exist at content/blog/<slug>/contents.lr, the folder name must match the URL slug exactly, and the target's pub_date must map to the URL date components (unpadded month/day per slug_format: pub_date 2026-07-23 -> /blog/2026/7/23/<slug>/).

Report each check 'Verified: OK' or as a claim-vs-source mismatch with line numbers, same format as the other axes. A concept post gets axes #1 (privacy) + #2 (style) + this variant of #3.

## PR creation (multi-account gh), blogbot output, and post-PR dictated edits (2026-09-11, PR #300)

- ## PR creation (multi-account gh), blogbot output, and post-PR dictated edits

- **Multi-account gh**: PR creation from the worktree fails with a permissions error when the ACTIVE gh account is `eric-ormoni` (it lacks collaborator access on ericmjl/website; two accounts are logged in). Run `gh auth status`, switch to `ericmjl` for the PR call, then switch back. (2026-09-11, PR #300.)
- **blogbot scripts are suggestion-only**: `summary.py` and `tags.py` print suggestions to stdout — there is NO write flag; they never modify contents.lr. Adopt/adjust the suggestions by hand and verify contents.lr actually changed afterward. The generated suggestions are still worth adopting when sharper (e.g. the generated summary caught that 'settled' overstates — the reveal, not the bet, unblocked the discussion).
- **Post-PR dictated edits**: when Eric's review feedback arrives (voice-dictated corrections) AFTER the worktree was cleaned up, recreate the worktree on the PR branch (`git worktree list` first, then `git worktree add ../website-<branch> <branch>`), apply the edits, push — CI re-verifies. Apply dictated replacement text via the edit tool in the SAME turn the dictation arrives; never answer dictation with a restated reframe plan. Dictated sentences land verbatim; drop the spoken meta-commentary ('that's the key one', 'blah blah blah').
- **Post-PR URL verification**: LinkedIn links return a 301 bot-authwall redirect to curl/bots — expected, not a broken link. Verify the URL byte-matches the one already live on the published site rather than following the redirect. (2026-09-11.)

## Post-PR dictated edits: correction vocabulary + reference re-anchoring (2026-09-11 PR #300 later turns)

- - **Dictated-correction vocabulary** (2026-09-11 funny-money-allocation): each dictated remark is a distinct edit op — 'phrased a little bit too weakly' = state the rule directly and boldly, un-hedge it (e.g. ground rules should SAY you can allocate 1,000 points any split, or keep unspent points to express uncertainty); 'comes in a little too abruptly' = add a transition sentence before the pivot (Eric often dictates the transition verbatim); 'this is the same move X would prescribe' -> 'analogous to the kind of thing X prescribed' = attribution precision, analogous-to is NOT same-as; a bare count correction ('not one successful run, four or five') = factual update from Eric's memory, the new source of truth, apply as-is; 'the first point is nonsense' = DELETE the item entirely, never soften/rewrite; 'phrase it better for me' = polish delegated — intent binding, exact dictated words not. Within one dictation stream the FINAL phrasing wins ('no no no sorry. This is what I want.'); self-corrected intermediates are superseded. Keep-verbatim flags ('that's definitely a statement I want to keep') are binding.
- **Positional-reference re-anchoring on list edits**: when inserting/deleting a bullet in a list that surrounding prose references positionally ('The second point gets to the heart of the problem'), re-check every positional reference after the edit — insertion shifts positions silently. When Eric flags one item as 'the most important piece I'm trying to really convey', place the new item so the existing emphasis phrase lands ON it and say so in the report (2026-09-11: reveal-order anchoring bullet inserted at position 2, between 'collect independently' and 'sum publicly', so the closer's 'second point' pointed at the flagged item and the anchoring thread runs collect -> reveal -> sum).

## Scheduled Merges: the PR title date is an auto-merge trigger

- The website repo runs gr2m/merge-schedule-action. It reads the [YYYY-MM-DD] at the end of a blog PR title as a SCHEDULED MERGE DATE and squash-merges the PR (in Eric's name, via GHPAGES_TOKEN) the moment all checks pass on or after that date. The bracketed date is a timer, not a label.

Rules:
- NEVER put today's date (or a past date) in the title unless Eric explicitly asked for same-day publish. Same-day schedule = the bot merges about 2 minutes after the PR opens = zero review window. This published an unreviewed, slop-heavy draft straight to production (2026-09-22, PR #304; reverted in bb71ae0 and re-opened as #305 dated 3 days out).
- Default: set pub_date 2-3 days in the future so Eric gets a review window on the Netlify deploy preview. He merges early when satisfied; the scheduler is the deadline, not the goal.
- Hold longer: gh pr edit <num> --title 'blog: <Title> [<new-date>]'. Ship early: merge the PR manually at any time.
- If a scheduled merge fires in error: revert the commit on main, let CI redeploy, then open a fresh PR from the branch with a future date and leave a paper-trail comment on the old PR. After the revert, production may keep serving a stale 200 from CDN cache for a while; verify removal via a 404 at the pages source or a fresh non-cached request, not a single curl.
- Before opening the PR, run the unslop audit on the body. Style gates (unslop banned-phrase scan, no em dashes, voice check) are PRE-PR steps, not post-publication repair: Eric reviews the draft on the deploy preview, and slop in what he reviews wastes the review cycle (2026-09-22 correction: 'it reads really sloppy, there is a lot of AI slop'). Eric's standing rule (2026-09-22): every draft gets a TRIPLE unslop pass before the PR opens, and EACH unslop invocation is preceded by updating the skill from source first (npx skills update unslop). One pass is not enough; he called a single-pass draft 'a lot of AI slop'.

## Mermaid diagrams in posts

- templates/layout.html already loads mermaid v10 (startOnLoad: true, theme 'neutral', securityLevel 'loose') and styles .mermaid as a white padded card. To add a flowchart to a post, embed a raw HTML block in the body markdown - <div class="mermaid"> graph TD ... </div> with blank lines separating it from surrounding markdown - and NO per-post script import (the older 5-retrieval-strategies post imports its own v11; that predates the layout wiring, don't copy the pattern). Verified 2026-09-22: the multiplayer-chat-agent mention-path flowchart renders at its natural size in both light and dark mode. Use quoted node labels (["..."]) and keep labels short; unquoted parentheses/colons in labels break parsing.

## Appending generated fields to contents.lr (separator gotcha) + Chrome visual check

- When APPENDING a generated field (tags: or summary:) from blogbot output to an existing contents.lr, the standalone --- separator line MUST be written BEFORE the new field. Lektor does NOT error on a missing separator: the field is silently parsed into the PRECEDING field's body (2026-09-22 multiplayer-chat-agent: tags appended after the body with no separator rendered as a stray 'tags: ai discord ...' blob at the end of the post body and the post had no tags field). AGENTS.md documents the separator rule, but appends are where it actually bites. After any append, verify the field layout before pushing: confirm every canonical field name appears (title, author, body, pub_date, twitter_handle, summary, tags) and the standalone --- separator count matches the most-recent post's contents.lr. Then run the VISUAL pass the user requires (stated 2026-09-22 as a required step): open the deploy preview in real Chrome via the agent-browser skill and confirm it LOOKS right — header meta renders (tags as field links, date, banner image), the body ends clean with no stray blob, and code blocks + dark mode render correctly. This rendering pass complements Post-PR URL Verification: that pass checks link HTTP status; this pass catches field-parsing corruption and layout breakage that a 200 status hides.

- When EDITING or REGENERATING an existing contents.lr field via regex/substitution (not appending), a pattern that assumes a blank line after the `---` separator SILENTLY NO-OPS: Lektor fields run `---\nbody:` with the next field starting immediately after the separator line, so a blank-line expectation matches nothing and the substitution succeeds while changing nothing (2026-09-25, PR #309: the regeneration regex no-opped, git reported nothing to commit, and the stale mental model of the committed file survived until an empirical diff). After ANY scripted edit to contents.lr, verify the diff actually landed: `git diff --stat` must show the expected +/- and a grep for the new phrasing must hit; nothing-to-commit right after an intended edit is the tell. Prefer line-based replacement keyed on the field name over multi-line regexes. Corollary on verification asserts: when a post-write self-check fails on an expected string, first confirm the string ever belonged to THIS file — sibling drafts (issue draft vs blog draft) contaminate expectations, and the assert failure can be a false alarm rather than truncation (PR #309: the missing string came from the issue draft, which the post never contained).
