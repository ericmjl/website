---
name: Eric J Ma's Website
description: Personal academic site of Eric J Ma — terminal-native, artifact-first.
colors:
  paper: "#ffffff"
  slate-dark: "#222225"
  ink: "#151515"
  ink-dark: "#e8e9ed"
  signal-blue: "#1a95e0"
  signal-blue-dark: "#62c4ff"
  quiet-gray: "#727578"
  quiet-gray-dark: "#a3abba"
  code-wash: "#e8eff2"
  tile-dark: "#2a2a2d"
typography:
  display:
    fontFamily: "'Berkeley Mono', monospace"
    fontSize: "1.6rem"
    fontWeight: bold
  display-compact:
    fontFamily: "'Berkeley Mono', monospace"
    fontSize: "1.3rem"
    fontWeight: bold
  title:
    fontFamily: "'Berkeley Mono', monospace"
    fontSize: "1.05rem"
    fontWeight: bold
    lineHeight: 1.3
  body:
    fontFamily: "'Berkeley Mono', monospace"
    fontSize: "1rem"
    lineHeight: 1.55
  body-small:
    fontFamily: "'Berkeley Mono', monospace"
    fontSize: "0.9rem"
    lineHeight: 1.5
  label:
    fontFamily: "'Berkeley Mono', monospace"
    fontSize: "0.85rem"
  artifact-initial:
    fontFamily: "'Berkeley Mono', monospace"
    fontSize: "2.25rem"
    fontWeight: normal
rounded:
  none: "0"
spacing:
  xs: "0.5rem"
  sm: "0.75rem"
  md: "1rem"
  card: "1rem 1.1rem 1.1rem"
  lg: "1.5rem"
components:
  collection-masthead:
    textColor: "{colors.ink}"
    typography: "{typography.display}"
    padding: "0 0 0.75rem 0"
  artifact-card:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "{spacing.card}"
  card-artwork:
    backgroundColor: "{colors.paper}"
    rounded: "{rounded.none}"
    size: "aspect-ratio 16/9, full card width"
---

# Design System: Eric J Ma's Website

## Overview

**Creative North Star: "The Wall of Equals"** (user-pinned direction, 2026-09-20; supersedes the earlier "Front Page" hierarchy)

This site is a terminal that publishes. The committed world is terminal.css 0.7.2 with self-hosted Berkeley Mono, flat surfaces, 1px hairlines, and one blue accent — light mode on paper white, dark mode on slate #222225. The collection pages (Books, Projects, Talks, Teaching) read as a wall of equals: every artifact is one identical card in a uniform responsive grid. No item is featured; no item is a footnote. Artwork is real and linked at the source — YouTube posters from the ytimg CDN, book covers from Leanpub's CDN — never duplicated into the repo, never replaced by invented imagery.

**Key Characteristics:**
- One accent color per theme; everything else is ink, ground, and hairlines.
- Square corners everywhere in the collection system; no border radius.
- Real artifacts (video posters, book covers) are the only large imagery.
- Lowercase, prompt-style action links separated by middots.
- Depth appears exactly once: a soft shadow under a physical book cover.
- Equal emphasis: identical cards, aligned artwork, actions on a shared baseline.
- Artwork-less artifacts show their real command as a terminal prompt, not a placeholder.

## Colors

One accent per theme; the rest of the page is ink on ground ruled by hairlines.

### Primary
- **Signal Blue** (#1a95e0): links, action labels, hover accents, focus rings, card borders on hover. Light mode.
- **Signal Blue (Dark)** (#62c4ff): the same role in dark mode; brighter to hold contrast on slate.

### Neutral
- **Paper** (#ffffff): light-mode ground.
- **Slate Dark** (#222225): dark-mode ground.
- **Ink** (#151515): light-mode text; titles stay ink, never accent.
- **Ink (Dark)** (#e8e9ed): dark-mode text.
- **Quiet Gray** (#727578): light-mode secondary — meta lines, counts, venue lines, summaries.
- **Quiet Gray (Dark)** (#a3abba): dark-mode secondary.
- **Code Wash** (#e8eff2): tinted tile behind contained book covers and initials fallbacks. Dark mode: **Tile Dark** (#2a2a2d).

### Hairlines
- Light: `rgba(21,21,21,0.18)` standard, `rgba(21,21,21,0.34)` structural (masthead rule, card borders).
- Dark: `rgba(232,233,237,0.16)` standard, `rgba(232,233,237,0.32)` structural.

### Named Rules
**The One Voice Rule.** The accent blue appears only on interactive elements — links, borders-on-hover, focus rings. Never on titles, never on large surfaces. Titles are ink in both themes.

## Typography

**Single Font:** Berkeley Mono (self-hosted woff2), `monospace` fallback. One face, no pairing.

**Character:** Machine-set precision; every size step is a role, not a decoration.

### Hierarchy
- **Display** (bold, 1.6rem): collection mastheads (`h1.collection-title`); drops to **1.3rem** (Display Compact) under 600px.
- **Title** (bold, 1.05rem, 1.3): card titles (`h2.card-title`).
- **Body** (1rem, 1.55): prose on detail pages.
- **Body Small** (0.9rem, 1.5): card summaries.
- **Label** (0.85rem): meta lines, artifact counts, venue·date lines.
- **Artifact Initial** (2.25rem): the initials tile fallback for artwork-less artifacts.

### Named Rules
**The Prompt Case Rule.** Action links are lowercase (`watch`, `slides`, `github`, `book`, `website`, `details`) — they read as shell commands, separated by a middot `·` drawn as an `::after` pseudo-element on every non-last item.

## Layout

Single centered container (site-wide, 65em cap). Collection pages stack two regions: masthead row (title left, artifact count right) over a structural hairline, then the card grid. The grid is `repeat(auto-fill, minmax(280px, 1fr))` with a 1.5rem gap; cards stretch to equal heights per row (flex column, actions pinned by `margin-top: auto`). At 600px the gap tightens to 1rem and the masthead drops to 1.3rem. Spacing rhythm: 2rem between masthead and grid, 1rem/1.1rem inside card bodies.

## Elevation & Depth

Flat by default. Depth is carried by hairline rules and the tinted tile, not shadows. The single sanctioned shadow is physical: a book cover resting on the tinted artwork zone gets `0 6px 24px rgba(0,0,0,0.18)`. Nothing else in the collection system casts a shadow.

### Named Rules
**The Physical Object Rule.** A shadow is reserved for depicted physical objects (a book cover); UI chrome never casts one.

## Shapes

Square corners (radius 0) across the collection system. Borders are 1px hairlines: structural (0.34/0.32 alpha) for card outlines and the masthead rule, standard (0.18/0.16) for the artwork zone's bottom rule and row rules. Card borders turn Signal Blue on card hover.

## Components

### Collection Masthead
- **Shape:** full-width row, structural hairline bottom border.
- **Content:** collection title (Display) left, artifact count (Label, Quiet Gray) right.

### Artifact Card (the system's core component)
- **Shape:** flex column, 1px structural hairline border, square corners, equal heights per grid row.
- **Artwork zone — poster (default):** 16:9, full card width, hairline bottom rule; video posters `object-fit: cover` (hqdefault's letterbox bars crop away exactly).
- **Artwork zone — command plate (no artwork):** the quiet tile carries the artifact's real command set as a terminal prompt — `$ git clone ericmjl/<repo>` for repos, `$ open <host>/<path>` for sites — in Body Small, `$` in Quiet Gray, command in Ink, with the Signal Blue block caret after it (blinks on card hover; static under `prefers-reduced-motion`). Monospace is used here for actual commands, never as costume.
- **Body:** title (Title, ink, links to the artifact), meta line (Label: venue · month year, or category), summary (Body Small, clamped to 3 lines), action row pinned to the card bottom.
- **Hover:** card border → Signal Blue, 140ms ease-out. Nothing lifts, nothing grows.
- **Behavior:** artwork zone is one link, `tabindex="-1" aria-hidden="true"` — the title link is the accessible entry.

### Book Shelf Variant
Cover-bearing artifacts (books) render as **horizontal shelf cards** on a wider grid (`minmax(440px, 1fr)`, single column under 980px): the cover sits at its natural portrait ratio on the tinted tile (38% card width, 1.25rem padding, physical-cover shadow) with the content beside it. A portrait cover is never letterboxed into a 16:9 zone.

### Action Row
- **Shape:** horizontal prompt list; lowercase links in Signal Blue, `·` separators via `li:not(:last-child)::after` with `position: static`.
- **Hover:** underline only; the background stays transparent.
- **Legend:** terminal.css positions every `li::after` absolutely at the `li`'s top-left and pads `li` 20px for markers; inside this system both are defeated (`position: static; padding-left: 0`) or the separators land on top of the first link.

### Focus
- `:focus-visible`: 2px Signal Blue outline, 2px offset, on every collection link.

## Do's and Don'ts

### Do:
- **Do** keep titles ink-colored in both themes and reserve blue for interactive elements.
- **Do** hotlink artwork from its source (ytimg CDN, Leanpub CDN `s_hero2x`); derive cover URLs from the Leanpub book slug.
- **Do** strip terminal.css `li::before/::after` markers (`content: none`) and its 20px `li` padding inside any list you restyle.
- **Do** verify Leanpub URL parsing with `| string` before substring checks in Jinja.
- **Do** keep card heights equal per row (flex column + `margin-top: auto` on the action row).

### Don't:
- **Don't** copy artwork files into the repo (user constraint, 2026).
- **Don't** feature one artifact above the others — every item gets one identical card (user constraint, 2026).
- **Don't** let a link hover paint a background block. terminal.css's `a:hover` inverts links with a solid primary background; within this system every hover is `background: transparent` plus a color shift. A title hover that lands on the primary background is unreadable.
- **Don't** animate layout properties (`max-height`, `height`); use `grid-template-rows` or static clamps.
- **Don't** fabricate imagery for artifacts that lack it — use the command plate.
- **Don't** letterbox a portrait cover into a landscape zone; give covers the shelf layout.
