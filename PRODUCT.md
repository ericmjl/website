# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Three audiences, confirmed as a mix by Eric:

1. Data scientists and Python practitioners who watch talks, adopt tutorials, and use the teaching materials.
2. Conference organizers and potential collaborators scanning Eric's speaking/teaching range before reaching out.
3. Learners who buy the Leanpub books or work through the companion websites and notebooks.

## Product Purpose

Eric J Ma's personal academic site: blog, books, talks, teaching, open source, bio. It exists to share his work openly. Success means a visitor finds an artifact and engages with it: watches a talk, buys or reads a book, clones a repo, invites Eric to speak or teach.

## Positioning

Eric's blend of Bayesian statistics, network science, and agentic AI practice, taught openly and in the open source ethos. No neighboring personal site can copy the depth and breadth of freely shared teaching artifacts backed by real conference history.

## Operating Context

Lektor 3.4 static site (pixi-managed) deployed to ericmjl.github.io. Content authored in `contents.lr` files; collections (books, talks, teaching, projects) all share the `projects` model and one template. YouTube videos embedded via flowblocks. Site-wide light/dark toggle persisted in localStorage. Analytics: GA4 + PostHog.

## Capabilities and Constraints

- Books (2) are sold on Leanpub. Cover artwork must be **linked from Leanpub, never duplicated into this repo** (Eric, confirmed 2026).
- Talks (20) and teaching (12) items each carry a YouTube video; thumbnails may be pulled/hotlinked from the YouTube CDN (i.ytimg.com), which is their by-design serving mechanism.
- Projects (7) are repos and essays without video; a few have authored images in-repo.
- All four collection pages share one core theme and should be redesigned together as a coherent system.
- Both light and dark mode must work everywhere (site-wide toggle).

## Brand Commitments

- Terminal aesthetic: terminal.css 0.7.2 with self-hosted Berkeley Mono (woff2). This identity is durable and pinned by AGENTS.md.
- Dark mode implemented per AGENTS.md: `body.dark-mode` class + variable overrides in custom.css, never in layout.html.
- Voice: direct, first-person, natural (per AGENTS.md).

## Evidence on Hand

- Real YouTube URLs for all 32 talk/teaching items → real thumbnails available.
- Leanpub book pages with cover art served from Leanpub's CDN.
- A handful of authored images in-repo (testing-talk.webp, network-analysis.webp, wildlife camera-trap photos).

## Product Principles

- The artifact leads: videos, books, and code are the content; chrome recedes.
- Open sharing is the point; teaching materials stay free.
- The terminal identity is a feature, not a costume.

## Accessibility & Inclusion

- Maintain readable contrast in both light and dark themes; keep meaningful alt text on imagery.
