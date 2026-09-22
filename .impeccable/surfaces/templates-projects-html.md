---
version: 1
slug: "templates-projects-html"
primary_target: "templates/projects.html"
related_targets: ["templates/macros/project.html","assets/static/css/custom.css"]
---

# Surface brief: collection pages (books, projects, talks, teaching)

Scope: the shared collection listing served by `templates/projects.html` (models `projects`/`projects-carousel`) for the Books, Projects, Talks, and Teaching sections. Detail pages are out of scope.

Visitor mode: Experience. The artifact leads; chrome recedes.

Audience: data scientists watching/adopting artifacts, organizers scanning range, learners buying books. Job: compare artifacts as equals and act on any of them (watch / read / buy / clone).

Direction: "The Wall of Equals" — user-pinned equal-emphasis uniform card grid (2026-09-20, Eric rejected the featured/archive hierarchy: "there should be equal emphasis on each item in each page"). Supersedes dealt option card-frontpage from the same surface round (seed key `1f0fb6c4`); a user-pinned decision beats the roll. Code-led.

Memorable moment: a clean wall of identical artifact cards — artwork tops aligned, actions pinned to a shared baseline — where no item outranks another.

Artwork rules (Eric, binding): Leanpub covers hotlinked from Leanpub's CDN (s_hero2x), never copied into the repo; YouTube posters hotlinked from i.ytimg.com.

Link hover rule (Eric, binding, 2026-09-20): title links must never inherit terminal.css's invert-block hover (`a:hover` paints a solid primary background); a title hover is a color shift on a transparent background, always readable.

Unresolved decisions: none.

## Direction contract

THESIS: Equal emphasis. Every artifact in a collection is one identical card in a uniform responsive grid — nothing is featured, nothing is a footnote. It refuses both the incumbent's full-width slabs and the front page's featured-versus-ledger hierarchy.

OWN-WORLD: terminal.css variables in both themes, Berkeley Mono, square corners, 1px hairline borders that turn signal blue on card hover, ink titles, lowercase prompt-style action rows pinned to a shared baseline, real linked artwork on top of every card.

STORY: The visitor scans a wall of equals; each card offers artwork, title, venue and date, a three-line summary, and its own actions. Every card's title and artwork link to the full record.

FIRST VIEWPORT: Masthead (collection title + artifact count) over a hairline; below it, the grid's first full row of equal-height cards with artwork zones aligned at 16:9 and action rows on a shared baseline. Motion grammar: the grid fades and rises once as a whole (400ms exponential ease-out); card hovers shift border color 140ms; title hovers shift color only, never a background block. Blinking cursor retired with the featured plate. prefers-reduced-motion disables the entrance.

FORM: Equal-emphasis uniform card grid, pinned by Eric over the dealt hand (card-frontpage rejected, card-dossier and card-gallery not chosen); surface round seed key `1f0fb6c4`; code-led build.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance.
