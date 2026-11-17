---
name: website-standalone-html-widget
description: >-
  Build standalone interactive HTML widgets for the website's Lektor blog posts (iframe-loaded, auto-resizing height). Use when the user asks to build/create a widget, interactive element, carousel, calculator, quiz, survey, visual demo, simulation, forgetting-curve chart, or animated diagram for a blog post, or batch-build widgets via subagents. CONSTRUCTION-TIME RULES: (1) BAR-CHART AXIS HONESTY — bar heights on an ABSOLUTE scale anchored to the domain ceiling (100% probability, |amplitude|=1), never relative to the data max; (2) FIXED container heights (never min-height) for dynamic-content containers, or the iframe fit() auto-resize expands the frame as state changes; (3) FIXED width for text spans whose content changes length when sitting next to flex siblings; (4) legend swatch chips rendered as ::before on the classed span ITSELF — a descendant selector like '.lg-intake span::before' never matches. House style, iframe fit() workaround, and post-build checks live in the body. For connector lines/arrows between rendered DOM elements (gantt arrows, org-chart lines, flowchart edges), see references/dom-anchored-svg-connectors.md — anchor on measured rects, never de-novo coordinate math.
created_by: autolearn
created_at: "2026-07-27"
---

# Website Standalone HTML Widget

Build standalone interactive HTML widgets for embedding in the website's Lektor
blog posts. Each widget is a single self-contained `.html` file loaded by the
blog post via `<iframe>`. This skill captures the established house style and
the non-obvious iframe height-resize workaround so every widget matches the
reference post and resizes correctly inside its frame.

## When to use

- The user asks to build/make/create a widget, calculator, quiz, survey,
  visual demo, simulation, forgetting-curve chart, animated diagram, or any
  interactive HTML artifact for a blog post.
- The user asks to "build the N widgets in parallel" or "batch-build widgets"
  for a post or post series.
- You are extending or restyling an existing widget and need the canonical
  template.

## House style (apply to every widget)

These are fixed conventions derived from the reference post. Deviate only when
the user explicitly asks for something different.

### Structure

- **Standalone** `<!DOCTYPE html>` file. One file per widget.
- **Inline** `<style>` and inline `<script>`. No external CSS or JS files.
- **No frameworks, no dependencies.** No React, Vue, Tailwind, D3, Chart.js,
  no CDN `<script src>`. Plain HTML/CSS/vanilla JS only.
- **IIFE wrapper** around all script: `(function(){ 'use strict'; ... })();`
- **Plain DOM manipulation.** `document.getElementById`, `addEventListener`,
  `classList.toggle`, `style.prop = ...`. No virtual DOM, no templating.
- Target length: a single self-contained file, typically 150-250 lines.

### Visual / CSS

- `font-family: monospace` (the widget's defining aesthetic).
- Global reset + `box-sizing: border-box` on `*`.
- **Card container** (the widget's outer frame):
  - `background: #fafafa`
  - `border: 1px solid #ddd`
  - `border-radius: 8px`
  - `box-shadow: 0 2px 8px rgba(0,0,0,0.06)`
  - `max-width: 640px`, centered (`margin: 0 auto`).
- **Accent palette** (use these exact values unless told otherwise):
  - Primary blue: `#0066cc`
  - Green (positive/correct): text `#1a9e5b`, background `#e7f6ee`
  - Red (negative/wrong): text `#e85d3a`, background `#fdecea`
  - Neutral grays for secondary text and borders.
- **Typography sizing**: small, but with a HARD FLOOR of 12px. Body
  12-15px, padding 6-16px, `line-height: 1.4`-`1.55`. Headings only
  slightly larger; avoid large display type.
- **Minimum font size — 12px HARD RULE (all widgets, all surfaces)**: the
  user stated this as a non-negotiable rule: "minimum font size should be
  12pt font on the blog, on all widgets." No text anywhere in a widget may
  render below 12px. This covers: body text, axis/series labels, button
  labels, tooltips, AND inline SVG annotations (the α/β/α'/β'/H labels on a
  quantum-gate diagram, axis tick labels on a chart, legend text). If a
  layout (e.g. a dense SVG data-flow diagram) was sized for 7.5-9px fonts,
  RESCALE the whole diagram (widen the viewBox, enlarge background rects,
  re-space the elements) rather than ship sub-12px text. NOTE: earlier SVG
  examples in this skill (svg-diagram-label-bounding-box-overlap-and-alignment,
  svg-label-opaque-background-over-crossing-lines) used F=7.5px and now
  PRE-DATE this rule — do not copy that font size into new widgets; use
  F>=12 and recompute the glyph math (rect width = charCount × ~0.6F + 2×padding,
  rect height = ~1.2F + 2×padding) at the larger size.

  RECURRENCE WARNING (2026-08-07, the DAY AFTER this rule was stated):
  despite the 12px rule being stated verbatim AND captured in memory, a
  follow-up session applied an 11px FLOOR instead — bumped every sub-11
  value to 11px (CSS 10px→11px, 10.5px→11px, SVG font-size="10"→"11")
  and LEFT 11px values in place, violating the 12px floor. Prose priming
  alone did NOT prevent the 11px-vs-12px error — exactly the failure mode
  that motivated the greppable self-check for bar-chart-axis-honesty. The
  fix is the same: a deterministic grep gate.

  CONSTRUCTION-TIME SELF-CHECK (the greppable gate — run before declaring
  any widget built or edited): grep the file for every font-size
  declaration and confirm each value is >= 12:
    rg -n 'font-size' <widget>.html
  Check BOTH surfaces: (a) CSS `font-size: Npx;` and (b) SVG
  `font-size="N"` attributes on <text> elements. The recurring offenders
  are CSS 10px / 10.5px / 11px and SVG font-size="9" / "9.5" / "10".
  Values of exactly 11px are the SILENT violation — they look 'close
  enough' but violate the 12px floor. If ANY font-size is < 12, bump it
  to 12 (do NOT stop at 11). When fixing a batch of widgets, check ALL
  files, including widgets that are no longer embedded in the post (their
  stale CSS still violates the rule and may be re-embedded later).

## Iframe height auto-resize (the fit() workaround) — REQUIRED

The widget lives inside an `<iframe>` in the blog post. The iframe must grow
and shrink to fit the widget's content height, including when the user
interacts and the content height changes (carousel advance, expanding answer,
strategy switch, etc.).

**The naive approach fails:** setting `frameElement.style.height =
document.body.scrollHeight + 'px'` directly does NOT shrink the iframe when
content gets shorter, because `scrollHeight` is computed against the current
height and stays sticky on shrink.

**The canonical fit() — collapse to 0px ONLY on a real height change, guarded
so an unchanged height writes nothing:**

```js
function fit() {
  var f = window.frameElement;
  if (!f) return;              // not in an iframe — no-op
  var h = document.body.scrollHeight + 'px';
  // Write only on a real change: resetting to 0px on every call resized
  // this frame's own viewport and fired our resize listener, which called
  // fit() again in an endless loop that pinned the page.
  if (f.style.height === h) return;
  f.style.height = '0px';      // force scrollHeight to recompute from 0 (shrink case)
  f.style.height = h;
}
```

WHY THE GUARD IS LOAD-BEARING: the unconditional reset writes the frame's
style on EVERY call. A style write resizes the child document's viewport,
which fires a `resize` event inside the widget; a widget that listens for
`resize` (to re-fit after a window resize — the retreat-week carousel
pattern) then calls `fit()` again, whose reset fires another resize: an
ENDLESS LOOP (~5,700 events/s) that seizes page scrolling. Observed
2026-11-10 on two learning-how-to-learn-anything-together widgets; root
diagnosis and measurement evidence in the autolearn wiki pattern
`iframe-resize-clamps-page-scroll-range`. The guard
`if (f.style.height === h) return;` makes fit() write nothing when the
height is already correct, so the loop dies after at most one real change.
The 0px collapse is still required for the shrink case — it must simply
never run when the height already matches. The same guard landed in the
prompt-card template (ericmjl/skills PR #56); the two specs must stay
identical.

Call `fit()`:
- Once on init (after DOM is built).
- After EVERY interaction that can change content height: carousel
  Back/Next, strategy switch, expand/collapse, reset, answer reveal.
- On `window.resize` if needed — safe ONLY with the guarded write; a resize
  listener plus the unguarded reset is the infinite loop above.

If you forget `fit()` on a state change, the iframe will either clip content
or leave dead space below the widget.

## Batch-building multiple widgets in parallel

When the user asks for N widgets (e.g. "build the 5 widgets for this post"):

1. Write the **shared house-style spec** once (this skill IS that spec — quote
   it back to each subagent verbatim, or point them at this skill).
2. Dispatch **one subagent per widget** in a SINGLE tool block (parallel, not
   sequential). Each subagent owns exactly one file, so there are no write
   conflicts.
3. Give each subagent: the widget's content spec (questions, data, behavior),
   the house-style block above, and the explicit instruction to call `fit()`
   on init and every state change.
4. After the subagents return, **verify each widget**:
   - Read the file back, confirm structure (DOCTYPE, inline style/script,
     IIFE, fit() calls).
   - If the widget encodes math (a curve, a percentage, a distribution),
     run the numbers through `node -e` / `uv run python -c` and confirm
     the displayed values match the spec BEFORE declaring done. Surface
     values are not ground truth for arithmetic.

- When N subagents build N widgets from a shared house-style spec, they predictably DIVERGE on two axes even when the spec is quoted verbatim: (1) CRITICAL IMPLEMENTATION TRICKS — one widget omitted the fit() reset-to-0px step (named its function resize() and set height directly to scrollHeight, so the iframe never shrank on shorter screens); (2) NAMING — fit() became resize() in one widget. This is the parallel-author coordination-drift failure mode: each author in isolation can produce a correct-looking file that diverges from the shared brief on exactly the details that must be uniform. MITIGATION — after the batch returns, run a CROSS-WIDGET CONSISTENCY REVIEW (read-only) that checks: (a) every widget names the resize function fit() and uses the GUARDED reset pattern — grep for the guard 'if (f.style.height === h) return;' across all files, and flag any '0px' height write that has no height-equality guard before it (the unguarded reset self-fires resize loops — see the fit() section); (b) every widget calls fit() on init AND after every height-changing interaction; (c) palette hex values match exactly (grep '#0066cc' / '#1a9e5b' / '#e7f6ee' / '#e85d3a' / '#fdecea'); (d) card-container CSS matches. A per-widget correctness check (step 4 above) catches WITHIN-widget bugs; the cross-widget consistency review catches ACROSS-WIDGET drift. Do both. Discovered 2026-07-27 reviewing 5 widgets for the ai-turbocharges-learning post: before-you-read.html had the missing-reset and fit->rename defects while the other 4 were correct.

## cross-widget-diagram-visual-harmonization

- EXTENDS the cross-widget consistency review above (which covers fit() / palette / card-container CSS) to DIAGRAM-LEVEL VISUAL IDIOMS across sibling widgets of the same type in the same post. When a post embeds N widgets that share a visual element (e.g. 4 Bloch/unit-circle widgets, 3 bar-chart widgets), the DIAGRAM-LEVEL properties must be HARMONIZED, not just the card chrome:
  - Circle/diagram FILL: all sibling circles use the SAME fill rule (e.g. fill="none" unfilled). Do NOT leave one filled and another unfilled.
  - AXIS LINE STYLE: all sibling axes use the SAME stroke-dasharray (e.g. dotted "2 2"). Do NOT leave one solid and another dotted.
  - AXIS/STROKE COLOR: all sibling axes use the SAME hex (e.g. #ccc). Do NOT leave one at #bbb and another at #ccc.
  - SVG viewBox DIMENSIONS: all sibling circles share the SAME viewBox (e.g. 0 0 140 140) and the SAME CSS width/height.

  THE PATTERN: widgets built at different times (or by different subagents) DRIFT on diagram-level details even when their card-container CSS matches. A reader scrolling through 4 circle widgets in the same post notices immediately if one has a filled circle, another unfilled, a third with solid axes, a fourth with dotted — the visual inconsistency reads as carelessness even when each widget is individually correct.

  THE FIX: when you change a diagram-level visual property in ONE sibling widget (circle fill, axis dash, axis color), PROPAGATE the change to ALL sibling widgets of the same type in the post. This is the cross-widget analogue of the within-widget visual-idiom-consistency rule (below) — THAT governs sibling ELEMENTS inside one diagram; THIS governs sibling WIDGETS across the post. Discovered 2026-08-07 harmonizing 4 quantum-ML circle widgets (bloch-circle, spread-dial, interference, entanglement) whose circle fills, axis dash styles, and axis colors had drifted across separate authoring sessions.

## Verification discipline

A widget that "looks right" can still have wrong math. Before declaring a
widget done:

- Read the file back; confirm line count and structure.
- For any numeric content (percentages, curve points, distributions),
  re-derive the values independently (node/Python eval) and confirm they
  match the spec's stated values, including the tolerance bands the user
  gave (e.g. "75-80%" or "approximately 1%").
- Confirm `fit()` is wired to every height-changing interaction.

## What NOT to do

- Do not pull in a CDN framework (React, D3, Chart.js, Tailwind). The house
  style is dependency-free on purpose: the widget must still render in 10
  years without an external host.
- Do not write the iframe height unconditionally. The canonical fit() guards
  the write (`if (f.style.height === h) return;`) BEFORE the reset-to-0 +
  recompute pair; an unguarded reset fires a resize event on every call and
  loops forever in any widget with a resize listener. (The pre-2026-11-10
  advice — 'reset to 0px FIRST, unconditionally' — was itself the loop bug.)
- Do not write a single giant widget that bundles multiple unrelated
  interactives. One file, one widget.
- Do not hand-build the same house style into every subagent prompt from
  memory. Quote this skill's spec block verbatim so palette values, padding
  ranges, and the fit() pattern stay identical across the batch.

## pedagogical-design-before-you-read-and-quiz

- Complements the HTML house-style section above (which governs LOOK/FEEL + fit()); THIS governs the PEDAGOGICAL/UX DESIGN of the two recurring educational widget types, stated by Eric 2026-07-30 (quantum-ML post review):

BEFORE-YOU-READ widget — must be a set of EXACTLY 3 KEY FOCUS QUESTIONS that frame what the reader should hold in mind while reading (mapping to the post's throughline/sections). NOT a bullet list of prerequisites or a 'you'll get more out of this if...' list. The questions focus attention on what the post will resolve. Example shape: numbered list (1/2/3) with each a genuine conceptual question ending in '?'.

CHECK-UNDERSTANDING / QUIZ widget — FIVE requirements: (1) PAGINATION. Show ONE question at a time with Next/Back navigation, never all questions displayed together on one screen. (2) PER-OPTION EXPLANATIONS AUTHORED (data richness). Author an explanation for EVERY option — the correct answer AND each wrong distractor — so the data is rich and each wrong answer explains why IT is wrong. Do NOT collapse to a single generic per-question explanation. (3) SELECTED-ONLY REVEAL (UX) — the critical author-vs-reveal distinction. On answer, REVEAL ONLY the explanation for the option the user SELECTED: green = why their pick is right, red = why their pick is wrong. Do NOT reveal all options' explanations at once — that wall of text overwhelms the reader (Eric 2026-07-31 correction: 'the check-understanding quiz is overwhelming me with why all other options are right/wrong, not just why my option is right/wrong'). The correct answer may still be highlighted green so the reader learns the right one, but exactly ONE explanation block appears per answered question. NOTE: the earlier version of this rule (2026-07-30) specified only 'per-distractor explanations' and omitted this reveal distinction, which caused the all-at-once reveal the user rejected.  Authoring richness (req 2) and reveal restraint (req 3) are both required. (4) ANSWER-POSITION RANDOMIZATION — the correct answer must NOT be hardcoded to a fixed position (the first option, or any deterministic slot). Shuffle each question's options array on every page load so the correct answer lands in a random position each time (Fisher-Yates shuffle at init, before first render), and reshuffle again on "Try again." Hardcoding correct=true on options[0] in every question creates a detectable pattern a reader can exploit without learning the material. IMPLEMENTATION NOTE: the shuffle is only safe if scoring/grading is FLAG-BASED (check options[i].correct), never POSITION-BASED (never assume a fixed index is the answer). The .correct flag travels with its option through the shuffle, so flag-based scoring remains correct after randomization. Author the data with correct: true on the right option regardless of where it sits; let the runtime shuffle move it. (Eric 2026-07-30 first raised this after noticing every correct answer sat at index 0; 2026-07-31 confirmed as canonical quiz design: "that is actually how the quizzes should be built now.")
(5) QUESTION-BANK ARCHITECTURE (random subset per session). Author a LARGE bank (~15 questions) covering every section of the post, then surface only a RANDOM SUBSET (~5) per session. The bank/subset split serves two goals: (a) COVERAGE without FATIGUE - a reader sees a tractable 5 questions but the bank covers the full arc of the post; (b) REPLAYABILITY - each page load and each 'Try again (new questions)' re-rolls BOTH the question selection AND the per-question option order (combines with req 4), so a returning reader gets fresh questions and cannot memorize positions. Each surfaced question still follows reqs 1-4 (pagination, per-option explanations, selected-only reveal, option shuffle). Distinct from req 4: req 4 shuffles option ORDER within one question; req 5 shuffles WHICH questions appear. Stated by Eric 2026-08-07 (quantum-ML check-your-understanding rewrite): 'a bank of ~15 questions ... we only surface up a random subset of 5 questions of the 15 (at random) ... be sure to make sure that the options order is shuffled per question.' This is the default for new check-understanding quiz widgets unless the user specifies a different bank/subset ratio.


Apply these as DEFAULTS to new before-you-read and quiz widgets unless the user specifies otherwise. When batch-building a series of posts with these widget types (as parallel subagents), give BOTH the house-style block AND this pedagogical block to every subagent so the widgets do not diverge on pedagogical structure (the cross-widget drift the skill already warns about applies to design choices, not just CSS).

CORRECTION 2026-07-30 (SUPERSEDES the BEFORE-YOU-READ spec above): before-you-read is NOT 3 focus questions. It is a PRIMING activity of exactly 3 MULTIPLE-CHOICE questions, each with 4 options that are ALL PLAUSIBLE BUT SUBTLY INCOMPLETE BY DESIGN (no single correct answer; every option has a kernel of truth but misses something). The reader picks one; reveal ONLY the selected option's feedback (selected-only discipline, like the quiz), and the feedback uses a NEUTRAL BLUE accent (the #eef4ff/#0066ff panel), NOT green/red, because there is no correct/wrong here; it names the plausible kernel AND the gap the post fills. Paginated one question at a time (Back/Next), closing with a line that points the reader into the post. Stated by Eric 2026-07-30: 'This is supposed to be a set of 3 mcq questions, each with 4 possible answers, and each answer is plausible but subtly incomplete (by design).'

CONCEPT-DEMO SCALING (2026-07-30): when a demo illustrates a structural point (e.g. exponential state-space growth), let the reader SCALE the key parameter (1 -> 2 -> 3 qubits, exposing 2 / 4 / 8 joint outcomes), not just the minimal 1-unit case, so the structure is visible. Stated by Eric 2026-07-30 about qubit-measurement: 'we only show it for one probabilistic bit. how about two probabilistic bits, or three, just to show the point?'.

## Pre-reading priming MCQ widgets

- When a blog post needs a 'Before you read' pre-reading activity, build a PRIMING MCQ widget (not focus/prompting questions): 3 multiple-choice questions, each with 4 options that are ALL plausible but SUBTLY INCOMPLETE by design — no single fully-correct answer. The point is to prime the reader's priors and motivate them to read the post for the complete picture; the incompleteness of every option IS the lesson. Distinct from a comprehension quiz (one right answer). Use the standalone-widget house style (inline <style>/<script>, IIFE, monospace, card container) and the fit() height-resize workaround. Related pedagogical rule: for demos of scale-dependent phenomena (qubit/probabilistic-bit measurement, exponential state space), let the reader set n=1/2/3 and SHOW the general case via a joint-outcome histogram — a single n=1 demo doesn't make the point.

## clarity-no-unexplained-jargon

- ## Clarity: no unexplained jargon (applies to ALL interactive widgets, not just MCQ/quiz types)

The pedagogical section above governs MCQ/quiz/before-you-read widget STRUCTURE. THIS section governs the plain-language clarity of ALL interactive widget UI copy — preset names, pattern labels, axis labels, and formula text. It is the widget-specific application of Eric's plain-English voice.

RULE: every preset name, pattern label, and formula in a widget must be self-explanatory to a FIRST-TIME reader. Do NOT introduce domain jargon that the reader has no way to decode. If a reader's first reaction to a label is 'I don't know what that means', the label is wrong.

Three acceptable patterns (pick one per label):
1. NAME BY WHAT THE PATTERN VISIBLY IS. Instead of an abstract term like 'parity pattern', describe the visible shape, e.g. 'every other bar flips sign'.
2. SHOW THE PATTERN EXPLICITLY so the name is secondary. Render the +/− sign over each outcome directly in the UI; the name then just indexes what the eye already sees.
3. GLOSS IN PLAIN ENGLISH in-place. If a term is necessary, define it the moment it appears ('parity pattern — the sign alternates +/−/+/- across the outcomes').

Formula labels must state WHAT the quantity IS in plain terms, not just the math. A label like 'combined amplitude (1+alpha*sign)' fails if the reader cannot tell what the y-axis represents. Rewrite as a plain lead-in: 'Size of each outcome after the two paths combine (1+alpha*sign)' — naming the axis meaning first, formula second.

VALIDATE THE PHYSICS/MATH MODEL before shipping. When a widget models a real domain (quantum mechanics, probability, statistics), the intuitive abstraction it draws must be physically VALID, not merely intuitive-looking. A flat 'add +1 to every amplitude' contribution, for example, is NOT a valid quantum operation (amplitude vectors are transformed by UNITARY gates; adding a constant vector is not unitary/reversible), even though it makes a clean-looking bar chart. If you cannot map every widget operation to a legitimate operation in the domain, the model is misleading and must be redesigned or the widget scoped to explicitly label itself as a simplified illustration, not a faithful circuit.

Stated by Eric 2026-07-31 reviewing an Interference widget: 'I don't know what "parity pattern", "first-bit split" and "spotlight on 000" are supposed to mean when I first read the blog post. we have introduced jargon.' and, on the physics, 'is adding +1 a valid operation in quantum computing? I know flipping signs is, but is that how we introduce interference too?'

GENERALIZATION (Eric, 2026-08-28, fortnight weight-review dashboard — this rule now reaches beyond widgets): 'write clearly, you gotta write clearly in everything you give me actually' — the no-unexplained-jargon rule covers EVERY deliverable's user-facing text, not just widget UI copy: pure-HTML analysis dashboards (stat-card labels, chart legends, story-in-numbers sections), reports, any artifact Eric reads. Metabolic/data shorthand fails exactly like physics jargon: 'Apple's basal runs ~20% hot' is undeclarable by a first-time reader ('I don't know what you mean by that man') — write 'the calories Apple says your body burns at rest read about 20% high'. Translations that worked: basal → 'the calories your body burns just existing', TDEE → 'total daily burn', claimed deficit → 'what the devices say', calibrated/true deficit → 'the number that matches the scale'. Code-side text (JS comments, variable names like TRUE_TDEE) is exempt; only rendered copy is governed.

## verification-discipline-dynamic-computational

- EXTENSION to Verification discipline (for DYNAMIC/INTERACTIVE computational widgets whose output changes with user input, e.g. a phase-flip interference widget, a calculator with sliders, any widget implementing a mathematical TRANSFORM rather than displaying fixed values): the static 're-derive the stated values' check is insufficient because there is no single spec value — the output spans an input space (2^n patterns). Instead verify the COMPUTATION across its input space via three layers: (1) KNOWN-ANSWER CASES with analytic ground truth — pick inputs where you can hand-derive the exact expected output (e.g. all-+ phases → 100% probability on outcome 000, rest 0; a single phase flip → 0.5625 on 000 / 0.0625 elsewhere) and assert the widget's recomputation matches to the decimal. (2) INVARIANT TESTING over RANDOM inputs — generate N (e.g. 200) random input patterns and assert the domain invariant holds for every one (e.g. probabilities sum to exactly 1 because the transform is unitary). This catches normalization bugs that a few hand-picked cases miss. (3) SYMMETRY / GLOBAL-PROPERTY TESTING — assert the transform respects the domain's symmetries (e.g. flipping ALL phases reproduces the all-+ result, because global phase is unobservable). Implement as a standalone node -e / uv run python check script that replicates the widget's core computation function INDEPENDENTLY (re-derive the formula from first principles, do not import the widget's JS) so the check is not self-confirming. Discovered 2026-07-31 building the quantum interference widget (3 qubits, 8 outcomes, Walsh-Hadamard recombination): the static-value check from the skill would have missed a normalization drift that only surfaced over random patterns.

## make-the-mechanism-visceral-break-open-black-box-steps

- ## make-the-mechanism-visceral-break-open-black-box-steps

- Governs the CORE design test for any interactive widget that applies an OPERATION/TRANSFORM (a gate, a recombination, a matrix multiply, a filter). Stated by Eric 2026-07-31 reviewing the 3-qubit interference widget: 'the problem with this one is that it doesn't show what the hadamard gate is doing... include something that would allow the hadamard gate to be viscerally felt by the user... we aren't really "show"-ing what the hadamard gate is doing.'

THE TEST (ask before shipping any widget that computes something): is there a step that turns inputs into outputs as a BLACK BOX — a button like 'Recombine (Hadamard)' that maps phases straight to a probability pattern with no visible intermediate? If yes, that step FAILS the test. A reader who flips inputs, presses the button, and watches the bars change has NOT seen what the operation does; they have seen that it does SOMETHING. The operation must be made VISCERAL — the reader should FEEL the mechanism by watching its internals move.

THE FIX — break the operation into a visible SIGNED SUMMATION, shown for a user-SELECTED output one at a time. For a linear transform T acting on input vector v to produce output k: each input j contributes a VOTE = (kernel sign of T at (j,k)) × (input coefficient at j), which is + or −. Show all votes (e.g. for an 8-input transform, 8 rows: each input's gate-kernel sign in gray, combined with the user's phase shown as up/down arrow, yielding a blue + or red − vote). The votes sum to the net = output amplitude; amplitude squared = probability. Render the net as a readable formula line, e.g. '000: 8 add − 0 subtract = net +8/8 = amplitude +1.000 → probability 100%'.

WHY THIS MAKES IT VISCERAL: destructive interference stops being an asserted fact and becomes a VISIBLE event — select a 0% outcome and the panel reads '4 add − 4 subtract → balanced votes cancel to zero'. The cancellation is right there in the +/− vote split. Flip an input and the votes + net update LIVE, so the reader watches the gate re-sum in real time.

GENERALIZES beyond Hadamard to any orthogonal/unitary transform in a widget (Fourier, Walsh-Hadamard, any matrix multiply): do not display only the before-vector and after-vector. Display the SUMMATION that connects them, one selected output at a time, as signed votes. This is the interactive-widget surface of the broader mechanism-exposure preference (let the reader SEE and FEEL how things work, never hide the wiring behind a convenience wrapper or a magic button).

VERIFICATION (pairs with the verification-discipline-dynamic-computational section): the breakdown's per-output probability (net/8 squared) MUST equal the bar in the recombined-probability row for EVERY output — assert this across N (e.g. 300) random input patterns in a standalone node check that re-derives both from first principles, not by importing the widget's JS. Also assert probabilities sum to 1 and the analytic known-answer cases (all-+ → 100% on 000; 4/4 split → 0%; single flip → 56.25%) are exact.

DISCOVERED 2026-07-31: the first version of the interference widget had a 'Recombine (Hadamard)' button that was a pure black box — phases in, probability pattern out, with no visible intermediate. The user rejected it with the quote above; the signed-vote breakdown panel (click an output bar → see it assembled from 8 signed contributions → net → amplitude → probability) was the fix that passed.

- GENERALIZES to PARAMETRIZED-STATE DECOMPOSITION (not just transform summation): the 'black box' can also be a CONTROL PARAMETER that mixes sub-states, where the reader sees only the final basis amplitudes change but not WHY. Example: a 2-qubit entanglement dial η that produces joint amplitudes a00..a11. The reader asks 'why does amplitude appear on |11> as I turn the dial?' and the widget shows only the final amplitudes moving — a black box. THE FIX: display the Schmidt decomposition LIVE — 'state = 0.71 · |u0 v0> + 0.71 · |u1 v1>' with the two coefficients (cos η, sin η) and the constituent product states (|u0>, |u1> etc.) shown as live values. The reader can then RESOLVE each basis amplitude: at θ_A=θ_B=0, |u0>=|0>, |u1>=|1>, so the first term contributes to |00> and the second to |11>; the coefficients directly ARE the amplitudes. This is the state-decomposition sibling of the signed-vote summation: votes show how a TRANSFORM assembles each output from inputs; the Schmidt display shows how a CONTROL assembles the state from product terms. Both make the wiring visible instead of asserted. TRIGGER to apply this: a reader asks 'why does X change when I move Y?' about a widget where Y is a continuous control and X is a derived quantity — answer by showing the intermediate decomposition (coefficients + constituent terms) live, not just the final result. Discovered 2026-08-05 quantum-ML entanglement widget: user asked 'why do I get 0.71 amplitude on |11> at max entanglement?' — the fix was a live Schmidt decomposition panel, not a prose explanation.
- RECURRED + CONFIRMED at the SINGLE-QUBIT level (2026-08-07, hadamard-gate widget): the user independently re-derived the exact wave-interference mechanism ('we kinda create two new waves from the H gate: for component |0>, we create two new waves... and the two new waves constructively interfere') and asked 'is there a good visual way to show on the widget that this is exactly what's happening?' — i.e. they requested the path-breakdown panel for H acting on |+> = (|0>+|1>)/sqrt(2). This is the N=1, 2-PATH instance of the rule: input superposition has TWO components (the |0> part and the |1> part), each is mapped by H to a (+0.5, +/-0.5) pair, and the two contributions to each output either reinforce (|0>: +0.5 + +0.5 = +1, constructive) or cancel (|1>: +0.5 + -0.5 = 0, destructive). The path-breakdown panel therefore applies even at the SIMPLEST qubit count — do NOT assume the signed-vote summation is only for multi-qubit (8-vote) transforms. A 1-qubit gate widget needs the SAME two-row breakdown (one row per input component, showing its signed contribution to the selected output, summing to the net amplitude). User also confirmed the displayed 0.71 values are rounded 1/sqrt(2) while the underlying math (1/sqrt(2))^2 = 1/2 is EXACT — when a reader asks 'is that the exact math?', confirm exactness and clarify which displayed figures are rounded vs exact.
- ESCALATION: the signed-summation breakdown panel described above (rows of +/− votes summing to a net) is a FIRST PASS, not the ceiling. It shows the VOTES but not the FLOW — a reader sees a table of numbers and still cannot trace HOW each output component is derived from the inputs. When the user says the breakdown is 'still not visual enough' (recurred 2026-08-07 hadamard-gate widget, after shipping exactly the panel this section prescribes), escalate from a summation TABLE to a CIRCUIT-DIAGRAM LAYOUT — the standard quantum-circuit visual idiom (wires + gate boxes + end-cap dots). DELETE the textual breakdown panel and replace it with a spatial data-flow diagram.

THE CIRCUIT-DIAGRAM LAYOUT (stated verbatim by Eric 2026-08-07 for the single-qubit Hadamard gate): in VERTICAL order — (1) Bloch circle / unit circle with the draggable input arrow; (2) input SIGNED AMPLITUDE bars (alpha, beta); (3) a GATE BOX for H sitting BETWEEN the input and output amplitude columns, with WIRES (lines) carrying end-cap DOTS that connect the |0> and |1> input amplitude components to the gate box; (4) the gate box OUTPUTS the pre-interference values on its outgoing wires with the MATH shown inline (e.g. alpha/sqrt(2), beta/sqrt(2)); (5) output SIGNED AMPLITUDE bars on the right showing the recombined results; (6) PROBABILITY bars at the bottom are fine to keep. The load-bearing addition is step 3-4: a literal gate box with wires+dots connecting each input basis component to each output, so the reader can FOLLOW the path (input component -> wire -> gate -> wire -> output contribution) rather than reading a summation table.

THE PRINCIPLE BEHIND THE ESCALATION: what the user is missing is 'how we get each component from 0 and 1' — the DERIVATION FLOW. A summation panel enumerates the contributions; a circuit diagram makes the FLOW a visible object (a wire you can trace with your eye). The wire + end-cap dot idiom is how quantum circuits are drawn in every textbook and paper, so it is the idiomatic, immediately-decodable representation. When the summation panel is rejected as insufficient, the next step is always the spatial data-flow diagram with explicit gate boxes and connecting wires — never a denser/more-detailed table.

GENERALIZES beyond single-qubit Hadamard to ANY widget visualizing a TRANSFORM/GATE/MATRIX-MULTIPLY the user finds insufficiently visual: if the signed-vote panel (the section's original FIX) gets a 'still not visual enough' rejection, escalate to the circuit-diagram idiom (input basis -> wires+dots -> gate box -> output basis), with per-wire value labels and inline math. The two failure modes are distinct: (a) BLACK BOX (no intermediate shown at all) -> fix = signed-summation panel; (b) SUMMATION PANEL PRESENT but reader still cannot trace the flow -> fix = circuit-diagram layout with gate boxes and wires. Do not assume the panel is always sufficient.
## bar-chart-axis-honesty-absolute-scaling

- - Governs the Y-AXIS SCALING of any bar chart / histogram in a widget that encodes a magnitude or probability (the interference widget's 'After the Hadamard gate -> probability' row, any outcome histogram, any magnitude comparison). Stated by Eric 2026-08-01 reviewing the 3-qubit interference widget: 'the bar chart right underneath that should use absolute y-axis to be very clear.'

RULE: render bar heights on an ABSOLUTE scale where FULL BAR HEIGHT = the MAXIMUM POSSIBLE value (e.g. 100%% probability, or the theoretical ceiling of the quantity), NOT a relative/normalized scale where the CURRENT MAX always fills the track (i.e. height = prob / maxProb * fullHeight is FORBIDDEN).

WHY RELATIVE SCALING IS DISHONEST: with relative scaling, the tallest bar ALWAYS fills the track regardless of its actual value. A 12.5%% probability bar fills the whole track if it happens to be the local maximum, making a one-eighth chance look like a certainty. The visual lies about magnitude: the reader can compare bars RELATIVE to each other but cannot judge any bar's ACTUAL size, which defeats the purpose of a probability chart ('to be very clear').

THE FIX: height = Math.round(prob * fullHeight), anchored so prob=1.0 -> fullHeight and prob=0.0 -> a minimal gray stub (1-2px) so canceled/zero outcomes remain visible as 'gone' rather than vanishing entirely. A 12.5%% bar is then visibly 1/8 of full height; a 100%% bar fills the track; the eye reads true magnitudes.

GENERALIZES beyond probability to ANY magnitude-encoding bar chart in a widget (amplitudes, counts, intensities): the axis must be anchored to the domain's real ceiling, not auto-normalized to the current data's max. Distinct from make-the-mechanism-visceral (which governs SHOWING the summation that produces the bars) — THAT governs how a bar is EXPLAINED; THIS governs how the bar's HEIGHT is SCALED so the height is truthful. Both apply to the interference widget: break open the black box AND scale the result honestly.

- RECURRENCE WARNING (2026-08-05): this rule escaped AGAIN on the quantum-ML spread-dial widget, 4 days after first capture (2026-08-01). The SAME relative-scaling bug appeared in BOTH the amplitude bars (signedStrip) AND the probability bars (probBars) of the spread-dial widget, and the assistant self-flagged 'This is the same fix as the interference widget earlier.' The frontmatter description of this skill now names this rule as a load-time trigger precisely so it primes at construction, not just at review.

CONSTRUCTION-TIME SELF-CHECK (run for EVERY magnitude-encoding bar before declaring a widget done): for each bar-rendering function, ask 'what is the denominator in the height formula?' If the answer is 'the current max of the displayed values' (height = v / maxOfCurrentValues * fullHeight), STOP — that is relative scaling and is FORBIDDEN. The denominator MUST be a FIXED domain ceiling (1.0 for probability; 1.0 for |amplitude|; the theoretical max count for a histogram), independent of what is currently displayed. height = v / DOMAIN_CEILING * fullHeight.

AMPLITUDE BARS (the 2026-08-05 recurrence surface — signed magnitudes, |amp| <= 1): the absolute scale max is 1.0 (a single basis state has amplitude 1). height = |amp| / 1.0 * HALF (HALF = half the track height; the bar grows up for +amp, down for -amp from a centerline). At |000>: amp_0=1 -> full half-height; others 0 -> 1-2px stub. At uniform 3-qubit superposition: each amp = 1/sqrt(8) ~= 0.354 -> ~35% of half-height. So as the state spreads, the tall bar visibly SHRINKS (1 -> 0.354) and the short bars GROW (0 -> 0.354). With relative scaling none of this is visible — the tall bar always fills, hiding the redistribution that is the WHOLE POINT of the demo.

PROBABILITY BARS (the 2026-08-01 original surface): height = prob / 1.0 * fullHeight. At |000>: prob_0=1 -> full; uniform 3-qubit: each prob = 0.125 -> ~12.5% height. The tall bar shrinks 1 -> 0.125 as the state spreads.

This is the interactive-widget surface of memory #22 (enforcement-escalation): the rule was in skill prose with a checkable signature (denominator = current max) yet escaped twice in one project. The construction-time self-check above is the deterministic gate that catches it at build time rather than at review.
- DETERMINISTIC CODE SIGNATURE (the greppable smoking gun — catches this at write time, not review time): any bar-rendering function that computes a MAX from the CURRENT DATA before computing bar heights is using RELATIVE SCALING and is FORBIDDEN. The canonical wrong shape is a max-finding loop immediately before the height assignment:

WRONG (relative, the bug):
  var maxAbs = 0; values.forEach(function(v){ if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v); });
  if (maxAbs < 1e-9) maxAbs = 1;
  bar.style.height = Math.max(1, Math.abs(v) / maxAbs * PX) + 'px';

CORRECT (absolute, domain ceiling = 1.0):
  bar.style.height = Math.max(1, Math.abs(v) * PX) + 'px';

The signature is the PRESENCE of a max/dataMax/maxAbs variable derived by scanning the displayed values. If your render code contains 'var max...= 0' followed by a forEach that finds the data max, DELETE that computation entirely — the denominator is the FIXED domain ceiling (1.0 for both probability and |amplitude|), never the data max. The correct code has NO max-finding step at all; it multiplies the raw value by a constant. This recurred a THIRD time on 2026-08-06 (two-qubit-entanglement widget, joint amplitudes: the maxAbs loop + /maxAbs division) despite the frontmatter priming added 2026-08-05 — proving that prose priming alone does not prevent the bug; the code-signature grep does. When writing OR reviewing any bar-render code, search for '/ maxAbs' or '/ max' or 'dataMax' in the height formula — every hit is this bug.
## Iframe element styling (borderless) — REQUIRED

- The <iframe> element that embeds a widget in a blog post's contents.lr MUST be borderless: use `style="border: none;"` on the <iframe> tag. Do NOT add `border: 1px solid #ddd; border-radius: 6px;` or any border to the iframe itself. Rationale: the widget's inner card container already provides the visual frame (1px solid #ddd + border-radius: 8px, per the house style); a bordered iframe DOUBLE-FRAMES the widget. The iframe's only job is to host the widget and auto-resize its height (see fit() below). When embedding a widget, the iframe tag looks like: `<iframe src="./widget-name.html" width="100%" scrolling="no" style="border: none;" onload="..."></iframe>`. If you are retroactively cleaning up an existing post, replace `border: 1px solid #ddd; border-radius: 6px;` with `border: none;` across all iframes in contents.lr (replaceAll). Stated 2026-08-02 (quantum-ml + mrna-structure posts: 'make sure my iframes don't have borders on both blog posts').

## single-sample-multiple-representations

- ## sample-once-reuse-across-representations (sampling/Monte-Carlo widgets)

When a single user action (clicking a 'Measure' / 'Simulate' / 'Sample' button) produces a RANDOM OUTCOME that must drive MULTIPLE visual representations at once — e.g. a persistent tally counter AND a transient collapse animation, OR a histogram bin AND a status flash — sample the outcome EXACTLY ONCE and pass that single value to every representation. Do NOT call Math.random() / the sampler separately inside each representation's update path.

THE BUG: a measure(n) function that increments the tally with one random draw, then independently re-draws a SEPARATE random for the collapse-flash animation, produces a desynchronized display: the tally shows |0> incremented while the flash visually snaps to |1>. Each call site is individually 'correct' but they disagree because they sampled twice.

THE FIX: capture the outcome once at the top of the handler, then reuse:
  function measure(n) {
    var a = Math.cos(state.phi); var p0 = a*a;
    if (n === 1) {
      var o = Math.random() < p0 ? 0 : 1;   // single sample
      tally[o]++; renderTally(); collapseTo(o);   // both consume the SAME o
    } else {
      for (var i=0;i<n;i++) tally[Math.random()<p0?0:1]++;
      renderTally();
    }
  }

GENERALIZES to any widget where one probabilistic action fans out into several UI effects (counter + animation + sound + label). One outcome, many consumers — never one consumer per random call. The smoking gun that reveals the bug: the counter and the animation disagree about what just happened on the same click.

Discovered 2026-08-02 self-caught in the quantum-ml Bloch-circle widget's measure() function (tally increment vs collapse animation drew independent randoms).

## fixed-container-height-fit-stability

- Governs a PRECONDITION of the fit() auto-resize mechanism (see the 'Iframe height auto-resize' section above): for the iframe height to stay STABLE as the user interacts, every container that holds dynamic/varying content (bar tracks, chart areas, result panels whose children grow/shrink) MUST use a FIXED height, NEVER min-height.

THE BUG: a bar/chart container styled with min-height grows and shrinks to fit its tallest child. When the widget's state changes (user moves a spread dial, flips a phase, slides a probability), the bars inside grow or shrink, the container's height follows them, document.body.scrollHeight changes, and fit() (which sets the iframe height to scrollHeight on every state change) propagates that growth to the iframe. The whole iframe visibly expands and contracts as the user drags a slider. This is jarring and looks broken even when the math is correct. Stated by Eric 2026-08-05 (quantum-ML spread-dial widget): 'it seems like our y-axis height within the visual is not fixed? it is causing the height of the entire iframe to change as I change the spread towards uniform.... not good, you should fix it.'

THE FIX: give every dynamic-content container a FIXED height (e.g. .strip { height: 100px; } and .bars { height: 90px; }), not min-height. Bars then grow/shrink WITHIN a fixed-height track, the container height never changes, scrollHeight stays constant, and fit() reports a stable iframe height across all states. This COMPOSES with the bar-chart-axis-honesty rule (absolute bar scale) without conflict: absolute scale pins each bar's height to a domain ceiling; fixed container height pins the track; together they yield honest bars inside a stable frame.

CONSTRUCTION-TIME SELF-CHECK: after wiring fit(), sweep the widget through its full input range (min, max, uniform/midpoint) and watch the iframe's bottom edge. If the edge moves at all during the sweep, some dynamic container is using min-height instead of height. A stable frame across the full state space is the passing condition. This is the sibling gate to the axis-honesty denominator check.

GENERALIZES to any standalone widget whose content height depends on state (histograms, tally rows, expanding result blocks, animated diagrams): fix the OUTER track height so inner variation is absorbed within a constant total. Distinct from the fit() guarded-reset workaround (which governs HOW fit recomputes scrollHeight on shrink) — THAT is about scrollHeight stickiness; THIS is about the container heights that feed scrollHeight in the first place.

## fixed-width-dynamic-text-flex-sibling

- Governs the WIDTH-sibling analogue of the fixed-container-height rule above. Where THAT rule pins HEIGHT so the iframe stays stable via fit(), THIS rule pins WIDTH so a dynamic-text element does not shove its FLEX SIBLING (a range slider, an input, a button row) around when its text content changes length. Stated by Eric 2026-08-06 (quantum-ML entanglement slider widget): 'when I drag the entanglement slider all the way to the right, when "maximally entangled" shows up, the slider suddenly shrinks. this is bad ux. we should find a way to reveal maximally entangled subtly without causing jumps.'

THE BUG: a value-readout span (e.g. `<span class="v">0 (independent)</span>`) next to a range input sits in the same `.srow` flex row. As the user drags the slider, the span's textContent is updated to reflect the value ('0 (independent)' -> '45 (maximally entangled)'). The long string is ~170px at 11px monospace; the short one is ~70px. If the span's width is driven by its content (the default) or by a too-small `min-width`, the span GROWS when the long text appears, and because the row is flex, the sibling slider's available width SHRINKS. The slider visibly jumps/shrinks at exactly the moment the longest label appears — jarring and read as broken, even though the logic is correct.

THE FIX: reserve a FIXED width on the text span that accommodates the LONGEST possible string it will ever show: `.srow .v { min-width: <enough-for-longest-text>; white-space: nowrap; }` (e.g. min-width: 170px for '45 (maximally entangled)'). With white-space: nowrap the text never wraps, and with min-width >= longest-string the span's box is constant across all states, so the sibling slider never resizes. The slider is somewhat smaller than it would be without the reservation, but it is CONSISTENT (no jump) — that consistency is what matters. (Alternative if the long label is optional: shorten the label text AND reserve the width, e.g. '(max)' instead of '(maximally entangled)' — but the width reservation is the load-bearing fix; text-shortening alone does not prevent the jump if any two states differ in length.)

CONSTRUCTION-TIME SELF-CHECK: for every text element whose textContent is updated by an interaction handler (a value readout, a status line, a result string) that shares a flex row with a slider/input/button, (1) enumerate the set of strings it can show across the full input range, (2) identify the LONGEST, (3) confirm the element's CSS has `min-width` >= that longest string's rendered width AND `white-space: nowrap`. Sweep the slider across its full range and watch the SIBLING element's width — if the sibling resizes at any point, the text span is under-reserved. This is the width-sibling gate to the fixed-container-height gate.

GENERALIZES beyond slider readouts to ANY dynamic-text element adjacent to a flex/inline sibling whose geometry you want to keep stable: a status label next to a button, a count next to a progress bar, an error message next to a form field. The principle: when text length is a function of state and the text shares a row with a sibling whose size should NOT depend on that state, reserve the text's width up front. Distinct from fixed-container-height-fit-stability (HEIGHT / iframe stability via fit()) — THAT governs the vertical dimension and the outer frame; THIS governs the horizontal dimension and the in-row flex sibling. Both compose: a widget can need both (fixed-height bar tracks AND fixed-width readout spans). Discovered 2026-08-06 entanglement-slider widget.


## legend-swatch-chip-selector

- Governs the CSS selector form for legend color chips rendered as ::before pseudo-elements on classed label spans — the standard legend idiom for charts keyed by colored chips next to text labels. Stated by Eric 2026-08-28 (fortnight weight-review dashboard, daily-energy-balance chart): 'There are no colors next to the legend. So something's a little off there.'

THE BUG: the legend markup puts the class ON the label span itself (`<span class="lg-intake">Intake (cal)</span>`), but the per-series color CSS was written as a DESCENDANT selector: `.lg-intake span::before { background: var(--cyan); }`. That selector means 'a span INSIDE an element with class lg-intake' — but there is no child span; the class is on the span itself. The selector matches zero nodes, every chip's ::before keeps the shared geometry (from the matching `.legend span::before` rule) but gets NO background, and the legend renders bare labels with empty/transparent chips. The failure is SILENT: the page lays out, the chip boxes exist, only the color is missing — a casual 'did it render' glance passes.

THE FIX: target the span ITSELF with a compound selector — `span.lg-intake::before` (or `.lg-intake::before`) for each per-series color, keeping the shared chip geometry in the container-scoped rule (`.legend span::before`). Rule of thumb: when the class is on the element you want to decorate, attach the pseudo-element with a compound selector on THAT element; a descendant combinator (`.lg-item span::before`) is only correct when the markup really wraps an inner span (`<span class="lg-item"><span>...</span></span>`).

CONSTRUCTION-TIME SELF-CHECK: for every legend/per-series rule, strip the `::before` and trace the remaining selector against the actual DOM — does `.lg-intake span` match any node? Grep the HTML for `class="lg-` and look at what each classed element actually contains. Zero matches = dead CSS that will fail silently at render time.

GENERALIZES to any pseudo-element decoration keyed by class: key chips, badges, axis markers, bullet icons. Root pattern: a descendant combinator (`A B`) written where a compound selector (`B.A` / `AB`) was meant — the rule parses fine, matches nothing, and produces no error anywhere. Distinct from the cross-widget shared-element drift rule (THAT keeps parameters consistent ACROSS widget files; THIS is a within-file selector/target mismatch). Discovered 2026-08-28 on the fortnight-review dashboard (vault pure-HTML deliverable, per the 'pure HTML page' commission); applies to iframe widgets and pure-HTML dashboards alike.

## auxiliary-visual-element-label-coincidence

- ## auxiliary-visual-element-label-coincidence

- Governs the ADDITION of auxiliary/guide visual elements (a secondary arrow, a reference line, a projection, a complement vector) to a labeled SVG diagram. Stated by Eric 2026-08-06 (quantum-ML entanglement Bloch-circle widget): 'what is the dashed line at |1> exist? I do not understand where that is coming from?'

THE BUG: you proactively add an auxiliary element to make a concept visible (e.g. a dashed u1 arrow at 90 degrees from the main u0 arrow, to show the Schmidt partner). But when the MAIN parameter sits at a basis/extreme state, the auxiliary element's position COINCIDES with an axis/basis LABEL. A dashed arrow pointing straight up at the |1> label reads as 'this arrow is about |1>' or 'what is this line at |1>?' rather than 'this is the orthogonal Schmidt direction u1'. The reader cannot decode an auxiliary element whose position overlaps a label — the label CAPTURES the element's meaning. A caption saying 'the dashed arrow is u1' does not rescue it; the visual coincidence overrides the text. Proactively-added embellishments that confuse the reader are WORSE than no embellishment at all — the reader's attention is spent decoding the mystery element rather than the main concept.

THE FIX (three options, in preference order): (1) REMOVE the auxiliary element entirely and explain the concept in TEXT only (the formula display + caption). This is the cleanest when the auxiliary element is not load-bearing for the interaction. (2) If the auxiliary element IS load-bearing, ensure it NEVER coincides with any label at any parameter value — offset it, shorten it, or reposition it so no sweep state lands it on a label. (3) Label the auxiliary element in-place ('u1' next to the dashed arrow) so its identity is self-declaring even when near a basis label — but this adds clutter and only works if the in-place label is unambiguous. In the 2026-08-06 case, option (1) was correct: the u1 concept stays in the Schmidt formula text and the caption; the circles just show u0.

CONSTRUCTION-TIME SELF-CHECK (run before declaring any labeled-diagram widget done): enumerate every auxiliary/guide element you added beyond the main interactive element. For each, sweep the main parameter through its full range (min, max, and every basis/extreme state — 0deg/90deg/180deg/270deg for a circle). At each sweep state, ask: does the auxiliary element's position land ON or OVERLAP any axis/basis label? If yes at ANY state, apply one of the three fixes above. The smoking gun that you missed it: a reader asks 'what is the [line/arrow] at [label]?' — the question itself reveals the coincidence captured the element's meaning.

ALSO CHECK (label-truncation, same sweep): while sweeping the diagram, confirm every SVG TEXT LABEL sits fully WITHIN the viewBox bounds, accounting for the text's width on BOTH sides of its anchor (x) point. A label at x=124 in a viewBox of width 128 gets clipped on the right; the fix is to move it inward (e.g. x=118) so the full glyph string renders. This is the static-position sibling of the fixed-width-dynamic-text rule (which governs text that CHANGES length); here the text is constant but positioned too close to the edge. Discovered 2026-08-06 (quantum-ML Bloch-circle |0> label truncated at x=124 in BOTH circles; fixed to x=118).

GENERALIZES beyond quantum Bloch-circles to ANY labeled SVG diagram with a parameterized main element plus derived auxiliary elements (a projection vector, a complement, a rotated copy, a shadow, a normal): the auxiliary element must not visually collide with axis/basis labels at any reachable state, and all labels must sit within the frame. Distinct from make-the-mechanism-visceral (which governs SHOWING the computation that produces a result) — that is about explaining mechanism; THIS is about the spatial/visual clarity of auxiliary guide elements relative to labels. Both can apply to the same widget: the Schmidt decomposition TEXT panel makes the mechanism visceral, while the circles must not confuse the reader with label-colliding arrows.
## visual-idiom-consistency-across-sibling-elements

- Governs the ENCODING CONSISTENCY of sibling visual elements (two arrows, two lines, two bars) that share a visual idiom (length, color, stroke style). Stated by Eric 2026-08-06 (quantum-ML entanglement Bloch-circle widget): 'we're mixing visual idioms here. length means something for the solid line and something else for the dotted line (i.e. nothing). what can we do to make the visual, geometric idioms more consistent?'

THE BUG: two arrows on the same diagram, one solid and one dashed. The solid arrow's length encodes a quantity (purity, magnitude, weight). The dashed arrow's length encodes NOTHING — it is always full-length, serving only as a direction marker. A reader sees two arrows that both have a length and naturally assumes length means the SAME thing for both. When one changes and the other doesn't, the reader either (a) invents a false meaning for the static one, or (b) fails to decode the dynamic one. The mixed idiom silently misleads: the geometry the reader is asked to reason about is incoherent because the two arrows are read against different invisible rulers.

THE FIX: when two sibling elements share a visual idiom (length, color, stroke), make them encode the SAME KIND of quantity so the reader can compare them on equal footing. In the 2026-08-06 case: the solid arrow = cos eta (weight of the u0 term in the Schmidt decomposition) and the dashed arrow = sin eta (weight of the u1 term). Both now MEAN 'Schmidt coefficient weight' — a coherent shared idiom. The result is a SEE-SAW: as eta grows, the solid arrow shrinks (1 -> 0.71) and the dashed arrow grows (0 -> 0.71), directly mirroring the formula 'state = cos eta |u0 v0> + sin eta |u1 v1>' shown above the diagram. At maximal entanglement (eta=45deg) BOTH arrows are 0.71, making 'both terms equally present' a visible fact rather than a formula assertion.

CONSTRUCTION-TIME SELF-CHECK: enumerate every pair of sibling elements that share a visual idiom (any two arrows whose length you can compare, any two bars, any two lines). For each pair, ask: does element A's length and element B's length encode the SAME kind of quantity (both = weights, both = magnitudes, both = probabilities)? If A's length means something and B's length means nothing (it is a fixed direction/reference marker), the idiom is MIXED and the reader will be misled. Fix by either (1) giving B a length that encodes a quantity of the SAME kind, so both are read against one ruler; or (2) making A's length ALSO purely directional (drop the magnitude encoding) if no coherent shared quantity exists. The smoking gun that you missed it: a reader says 'length means X for the solid line and nothing for the dotted line' or 'why is one arrow changing and the other isn't?' — the question itself reveals the mixed idiom.

GENERALIZES beyond quantum Bloch-circles to ANY labeled diagram with sibling elements sharing an idiom: two vectors whose lengths the reader compares, two trend lines, two histograms overlaid. Distinct from bar-chart-axis-honesty (which governs the SCALE/ruler each bar is measured against — THAT fixes the ruler; THIS fixes whether the two elements are even measured against the SAME ruler and encode the same kind of thing). Distinct from auxiliary-visual-element-label-coincidence (which governs WHERE an auxiliary element sits relative to labels — THAT is spatial; THIS is semantic/encoding). All three compose: a diagram can need honest absolute scaling AND label-clear placement AND idiom-consistent encoding.

SUPERSEDES NOTE: an earlier design decision for this same widget series recorded 'keep the dashed u1/v1 arrows STATIC regardless of n, because they are Schmidt basis directions.' The user's idiom-consistency critique reversed that: the static-dashed version mixed idioms (solid = magnitude, dashed = nothing) and was replaced by the see-saw (solid = cos eta, dashed = sin eta, both = Schmidt weight). When a geometrically-'pure' design (each arrow represents its domain-true quantity, even if that means one is static) conflicts with a reader-decodable design (both arrows share one coherent idiom), prefer the latter in an EDUCATIONAL widget — reader-decodability beats domain-geometric purity, because the widget's job is to teach, and a mixed idiom teaches the wrong thing. (Discovered 2026-08-06; the earlier static-dashed decision was the same day and did not survive first user review.)

## quantum-state-vector-completeness

- For quantum state-vector widgets (bloch-circle, many-qubits, two-qubit-entanglement, hadamard-gate, any Hadamard/gate visualizer), ALWAYS show all THREE linked visual representations together and keep the visual connection tight between them: (1) CIRCLE/ARROW — the geometric state on a unit circle (draggable for input); (2) SIGNED AMPLITUDE BARS — the alpha/beta values as signed bars with a midline, NOT just text labels — the SIGN must be visually represented because the sign of an amplitude is what drives quantum interference (this is the whole point of the Hadamard gate); (3) PROBABILITY BARS — amplitudes squared. The chain is: arrow projections = amplitudes (motif 1→2), amplitudes squared = probabilities (motif 2→3). A widget that shows only probabilities, or shows amplitudes as plain text instead of signed bars, is INCOMPLETE — the user corrected this explicitly on the Hadamard gate widget (2026-08-07): 'not just probabilities, but also amplitudes. the tight connection needs to be kept between the three visual motifs.' Recurs across the quantum widget series.

## fixed-container-height-tall-enough-for-stack

- The COMPLEMENT to fixed-container-height-fit-stability (which says use FIXED height, never min-height, so the iframe stays stable via fit()). THAT rule says 'use a fixed height'; THIS rule says 'make that fixed height TALL ENOUGH for the full stacked content at max bar extent.'

THE BUG: a vertical stack of elements (motif-title + bar-track + value-readout + label) lives inside a fixed-height container. The bars can grow to a known max extent (e.g. 22px up + 22px down from a midline = 44px track). The container's fixed height was sized for a SHORTER content stack than the full-height case. When a bar reaches max amplitude, it overflows the track's allocated space and overlaps the title above OR the value/label below. Smoking gun: the user reports 'the bars cover the [TITLE] text' or 'it covers the value on the bottom' — text and bars are colliding because the container height < sum of stacked elements at max extent.

Concrete instance (2026-08-07, hadamard-gate widget): .ampstrip container = 64px; content stack = motif-title (~14px) + amp-track (44px) + amp-val (~14px) + amp-lab (~14px) = ~86px. 64 < 86 -> overflow -> amplitude bars covered the 'AMPLITUDES' title and the amplitude value text. Fix was to increase .ampstrip height to 84px and amp-track to 56px so the full-height bars fit with margin.

THE FIX: when setting a fixed container height for a vertical stack that includes dynamically-sized bars, compute the SUM of all stacked elements at their MAXIMUM extent (tallest bar + every static element: title, value, label) plus inter-element margins, and set the container height >= that sum. Do NOT eyeball a height that looks right at the resting state — the bars must fit at their FULL extent without touching neighbors. Also grow the bar-TRACK height in proportion so max-bar-height <= half-track-height with margin to spare.

CONSTRUCTION-TIME SELF-CHECK: sweep the widget to the state that produces the TALLEST possible bar (max |amplitude| = 1). Visually confirm: (a) the top of the tallest up-bar does not touch/overlap the title above it; (b) the bottom of the tallest down-bar does not touch/overlap the value/label below it. The user's 'just give it more height' is the tell that the fixed height was undersized for the max-extent stack.

GENERALIZES beyond amplitude bars to ANY fixed-height container holding a vertical stack where one element is dynamically sized (bars, a growing chart, an expanding panel): the fixed height must accommodate the tallest possible child PLUS all its static stack-mates plus margins. Distinct from fixed-container-height-fit-stability (which prevents the IFRAME from growing/shrinking) — THAT governs the OUTER track's height for scrollHeight stability; THIS governs the INNER container's height for non-overlap of stacked siblings. Both compose: use fixed height (outer stability) AND make it tall enough (inner non-overlap).
## svg-diagram-label-bounding-box-overlap-and-alignment

- Governs the PLACEMENT of STATIC text labels (operations, coefficients, values, annotations) along SVG line/path geometry in a labeled data-flow diagram (a butterfly/circuit diagram with crossing and straight lines, a matrix-multiply visualization, any diagram where annotations sit on wires). Stated by Eric 2026-08-07 reviewing the single-qubit Hadamard butterfly diagram: 'it's just a tad crowded? ... move all of the +0.5 -0.5 (i.e. the values) off to the right a tad bit, so that they don't overlap with the dotted lines. and then the operation (x1/sqrt(2) etc.) to the left, so they don't overlap with the lines. finally, middle-align each pair (operation, value), so that they're visually uniform. use bounding boxes to ensure that there's no overlaps, and that appropriate center alignments (for operations and for values respectively) and middle alignments (for operation + value pairs) are applied.'

THE BUG: labels placed by eyeballing pixel coordinates land ON TOP of the dotted/straight lines they annotate (a "+0.50" value sitting on a dashed crossing line), collide with sibling labels, and rest at inconsistent y-baselines (each operation at a slightly different height). The diagram reads as 'crowded' / cluttered even though the math is correct. The clutter is a BOUNDING-BOX problem: nobody computed whether each label's rendered rectangle clears the lines and its neighbors.

THE FIX (four parts):
1. GROUP-SEPARATE from the line geometry. Place one label group (operations) to the LEFT of the central element (gate box / crossing point) and the other group (values) to the RIGHT, at x-coordinates chosen to clear ALL line segments in that half of the diagram. Widen the viewBox if the labels need more room — do not cram labels into the same span as the wires.
2. DERIVE each label's y from the LINE's y at the label's x (interpolate the path), so the label sits centered ON its line at that x rather than approximately near it. For a straight line from (x1,y1) to (x2,y2), y at x = y1 + (y2-y1)*(x-x1)/(x2-x1).
3. CENTER with SVG text ATTRIBUTES, not hand-nudged coordinates: `text-anchor="middle"` (horizontal center on the anchor x) + `dominant-baseline="central"` (vertical center on the anchor y). This gives pixel-correct centering without measuring glyph widths by hand, and applies uniformly to every label in the group so the group forms a tidy aligned column.
4. VERIFY by BOUNDING-BOX MATH. At font-size F, each glyph row is ~F*1.25 px tall and ~0.6*F px wide per character. Confirm: (a) the vertical gap between any two labels at the same x >= glyph row height (no label-label overlap); (b) the horizontal gap between a label group and the nearest line >= half the label's rendered width (no label-line overlap); (c) every operation+value pair on the SAME path sits at the same y (middle-aligned). These are checkable arithmetic assertions, not visual guesses.

CONSTRUCTION-TIME SELF-CHECK: after placing all labels, enumerate every label pair that shares an x (within the operations column and within the values column) and assert the y-gap >= F*1.25; enumerate every label and assert its rectangle clears the nearest line segment; for every operation+value pair, assert identical y (both interpolated from the same path). The smoking gun that you missed it: the user calls the diagram 'crowded' / 'a tad crowded', or says values/operations 'overlap with the dotted/straight lines', or explicitly asks you to 'use bounding boxes to ensure that there's no overlaps' — the phrase 'bounding boxes' IS the request to reason about label rectangles rather than eyeball positions.

GENERALIZES beyond the Hadamard butterfly to ANY labeled SVG data-flow diagram (quantum circuit diagrams, network/flow diagrams, matrix-multiply visualizations, signal-path diagrams) where annotations sit along crossing and non-crossing lines: group-separate (left/right of center), derive y from the line, center via text-anchor+dominant-baseline, verify by bounding-box arithmetic. Distinct from auxiliary-visual-element-label-coincidence (an auxiliary VISUAL ELEMENT hitting an axis LABEL — THAT is element-vs-label coincidence; THIS is label-vs-line and label-vs-label OVERLAP and alignment). Distinct from fixed-width-dynamic-text-flex-sibling (dynamic text that CHANGES length in a flex row next to a slider — THAT governs length-change stability; HERE the labels are static and the issue is placement/overlap/alignment). Distinct from the make-the-mechanism-visceral escalation (which governs WHETHER to use a circuit diagram at all and WHAT it shows — the wires+dots+gate-box idiom; THIS governs how to place the TEXT LABELS that annotate those wires once the diagram exists). Discovered 2026-08-07 single-qubit Hadamard-gate butterfly widget: operations (x1/sqrt(2), x-1/sqrt(2)) and values (+0.50, -0.50) annotating 4 lines (2 solid straight, 2 dashed crossing); first pass crowded them at the lines' midpoints and duplicated the 'H' label; fix widened the viewBox to 200x100, parked operations at x=50 and values at x=150 (clearing the x=11-139 line span), derived each y from its line, centered with text-anchor=middle + dominant-baseline=central at font-size 7.5, and spaced y's at 14-19px gaps (>9.4px glyph height).

## svg-label-opaque-background-over-crossing-lines

- The COMPLEMENT to svg-diagram-label-bounding-box-overlap-and-alignment (the section directly above). THAT section governs placing labels so they CLEAR the lines (group-separate left/right, derive y from the line, center via text attributes). THIS section governs what to do when a label CANNOT fully clear a line: in a butterfly/crossing diagram the dashed lines MUST cross through the middle, so any label positioned on a crossing line's path has the line passing through its glyphs, making the text illegible. Stated by Eric 2026-08-07 reviewing the single-qubit Hadamard butterfly (same diagram as the bounding-box-overlap rule): 'now the operations and values cross the lines. put those in a rounded corner box with opaque background so they're visible.'

THE BUG: labels are correctly positioned (group-separated, centered on their lines via text-anchor=middle + dominant-baseline=central), but the SVG `<text>` renders with NO background — the glyphs are transparent, so any `<line>`/`<path>` drawn BEFORE the text in document order shows through behind the characters. Where a crossing line passes through a label like '+0.50', the line bisects the digits and the value is unreadable. The bounding-box-overlap rule prevents label-label and label-line OVERLAP at the rectangle level, but it does NOT prevent a line passing through a label's OWN rectangle when the diagram geometry forces a crossing (a butterfly's diagonal wires must cross the center where the labels sit).

THE FIX: wrap each affected label in an opaque `<rect>` background (fill matching the card/page background, e.g. fill='#fafafa' for the widget house style) with rounded corners (rx='2'), rendered BEFORE the `<text>` in document order so the text sits on top. The rect masks the portion of the line that would otherwise pass through the glyphs. The line visibly enters and exits the box on either side, so the reader can still trace the wire; the label is fully readable inside the box.

SIZING (deterministic, not eyeballed — reuse the glyph-math from the bounding-box-overlap section): for a label at font-size F with text-anchor='middle' and dominant-baseline='central', center the rect on the SAME (x, y) as the text. Rect width = charCount × ~0.6F + 2×padding (~2px each side); rect height = ~1.2F + 2×padding. At F=7.5px a 5-char label ('+0.50', '×1/√2') → width ≈ 30px, height ≈ 12px, rx=2, so rect x = textX − 15, rect y = textY − 6. Insert each `<rect>` immediately before its `<text>` (or all rects before all texts in the group) so the z-order is rect-behind-text.

CONSTRUCTION-TIME SELF-CHECK: after placing labels on a diagram that has crossing lines, for each label ask 'does any line segment pass through this label's rectangle?' (the bounding-box-overlap section's arithmetic answers this). If yes at ANY label, add an opaque background rect to it. Do NOT blanket-box every label — only those a line actually crosses (blanket-boxing clutters the diagram and hides wiring the reader should see). The smoking gun that you missed it: the user says the labels 'cross the lines' or are 'not visible' / 'illegible' and asks to 'put those in a rounded corner box with opaque background' — that phrase IS the request to mask the lines behind the glyphs.

GENERALIZES beyond the Hadamard butterfly to ANY labeled SVG diagram where the geometry forces lines to cross through label rectangles: network/graph diagrams with crossing edges, signal-path diagrams, matrix-multiply visualizations with crossing factor lines, any butterfly/circuit diagram, any Sankey/flow with overlapping paths. Distinct from svg-diagram-label-bounding-box-overlap-and-alignment (which tries to AVOID line-crosses-label by REPOSITIONING — THAT is the first line of defense; THIS is the fallback when geometry makes avoidance impossible). Apply both in sequence: first place labels to clear lines (the overlap rule), then for any label a line still crosses, add an opaque background rect (this rule). Also distinct from auxiliary-visual-element-label-coincidence (an auxiliary VISUAL ELEMENT landing on an axis LABEL — THAT is element-vs-label coincidence; HERE a structural LINE crosses through a TEXT LABEL). Discovered 2026-08-07 single-qubit Hadamard-gate butterfly widget: after the labels were repositioned to x=50 (operations) and x=150 (values), the crossing dashed lines still passed through them; the fix was an opaque #fafafa rect (30×12px, rx=2) behind each of the 8 labels that a line crossed (4 operations + 4 values).

## cross-widget-consistency-audit

- When a single blog post embeds MULTIPLE iframe widgets that share a VISUAL ELEMENT (a Bloch circle / unit-circle SVG, a color palette, an axis convention, a font stack), that shared element DRIFTS across widgets because each widget file is authored independently: viewBox scales diverge (130 vs 140 vs 300), fills flip (none vs #f4f7fb), stroke widths and label positions creep. AUDIT PROCEDURE when the user says 'harmonize the X styles across all iframes': (1) grep the shared element's SVG/HTML tokens across ALL widget files in the post directory to enumerate every instance (viewBox=, cx=, r=, fill=, font-size=); (2) build a comparison TABLE of the parameters per file so discrepancies are visible at a glance; (3) pick the canonical spec (usually the most-recently-corrected widget, e.g. viewBox 0 0 140 140, center 70/70, radius 46, fill #f4f7fb, stroke #dde6f5 width 2); (4) edit each file to match, carrying the SAME parameters AND updating the JS that references center/viewBox constants (the SVG and the JS that draws on it must stay in sync); (5) verify with a grep that all instances now share the spec and curl the page to confirm it serves. This is the CROSS-WIDGET counterpart to the per-widget construction-time rules below (bar-chart-axis-honesty, fixed-container-height, fixed-width-for-dynamic-text): those govern HOW to build one widget correctly; THIS governs keeping a SHARED element consistent ACROSS the widgets in one post. Smoking gun you skipped it: a side-by-side reader notices one Bloch circle has a filled disk and an adjacent one is transparent, or labels sit at different margins despite 'looking the same' in isolation.

## font-size-floor-relaxed-to-11px

- - REVISION (2026-08-07): the font-size floor has been RELAXED from 12px to 11px. The user stated for the quantum-ML widget family: 'all of the iframes should have at least font size 11 throughout.' Treat 11px as the confirmed acceptable minimum going forward; 12px remains a conservative default but is NO LONGER the hard floor. When bumping sub-11 font sizes in existing widgets, target 11 (not 12). This SUPERSEDES the 12px HARD RULE above for the quantum-ML widgets and future widget work unless the user re-states 12. The spirit (no tiny <9px text; rescale dense SVG rather than ship unreadable labels) still holds at the 11px floor.
assistant picks ONE sibling's attributes (a filled disk) and applies them to all the others — overwriting an established convention (non-filled disks) that the user actually prefers. The work then gets reverted. An unprompted FILL change is especially costly because it flips the visual idiom (solid disk vs wireframe outline), which is a more fundamental change than a stroke-width or viewBox tweak.

THE RULE: before applying a harmonization pass, IDENTIFY which attribute(s) are being unified (fill? stroke? viewBox? font-size?) and CONFIRM the target value with the user when the attribute is one that changes the visual idiom (fill on/off, solid vs dashed). 'Make them consistent' means make them AGREE, not make them FILLED — consistency can be achieved by setting ALL to non-filled just as well as all to filled.

PREFERRED BLOCH-CIRCLE CONVENTION (the concrete target for quantum widgets): NON-FILLED (transparent/none fill on the disk) with DOTTED LINES along the |0> and |1> axes. Do NOT fill the Bloch disk. This is the user's stated aesthetic; harmonize all Bloch circles TO this non-filled + dotted-axis style, not away from it.

GENERALIZES beyond Bloch circles to any cross-file visual harmonization (bar-chart palettes, axis styles, label fonts): confirm the idiom-changing attributes (fill, color family, solid-vs-dashed) before bulk-applying, because those are the ones a user is most likely to revert. Stroke width, viewBox dimensions, and numeric proportions are lower-risk to unify directly. Distinct from visual-idiom-consistency-across-sibling-elements (which governs TWO elements WITHIN one diagram encoding the same quantity) and auxiliary-visual-element-label-coincidence (auxiliary elements colliding with labels) — both are about internal coherence of ONE diagram; THIS is about confirming the harmonization TARGET across a FAMILY of files before doing the bulk edit.

## paginated-widget-state-machine-render-and-persist

- CONSTRUCTION-TIME RULE for ANY paginated widget (quiz, before-you-read MCQ, multi-step wizard): ALL navigation (Next/Back) must flow through a single render() function that draws the current item from state — NEVER mutate the index without re-rendering — AND per-item interaction state (selected option, revealed feedback) must be PERSISTED per item index (e.g. answers[qIndex] = selectedIdx) and RESTORED on render. Failure observed 2026-08-30 building a before-you-read MCQ: go() advanced current without calling render, so Next/Back appeared completely dead (question content never swapped); separately, feedback state lived only in the visible DOM, so navigating Back to an already-answered question showed it as unanswered with its feedback panel gone. Fix: render(current) is the ONLY place DOM content is written; navigation handlers just update current and call render(); selection handlers write answers[current] then render(); render() reads answers[current] to restore any saved selection + feedback panel. Self-test the state machine by walking the full navigation cycle: answer Q1 -> Next -> Next -> Back x2 -> confirm Q1 still shows its selection and feedback. Distinct from the fit() height rule (those govern frame sizing; THIS governs content swapping and state restoration under pagination).

## paginated-quiz-render-discipline

- - PAGINATED WIDGETS (check-understanding quiz, before-you-read priming MCQ): route EVERY state change — navigate (Next/Back), select an option, reset — through ONE render() function that redraws the question body, nav-button disabled states, and any feedback panel from state. Do NOT mutate a position variable inside a go()/next() handler and assume the DOM follows: index mutation without a re-render call leaves Next/Back visibly dead (the UI never swaps questions). Root cause: two parallel code paths (one mutating state, one drawing DOM) never wired together; the file ships looking complete because the buttons exist and click. CONSTRUCTION-TIME CHECK: after writing, confirm every handler's LAST line calls render() (or that select handlers write state then call render()).

- PER-QUESTION ANSWER PERSISTENCE: store selections keyed by question index (e.g. answers[i]) and have render() restore them — returning to an answered question via Back must re-show BOTH the highlighted option AND its saved feedback panel. Without persistence, Back silently drops answered state and the reader must re-answer on return. CONSTRUCTION-TIME CHECK: grep the file for 'answers[' reads inside render() — absence means answered state is lost on navigation.

- COLOR-NAME GREP FALSE POSITIVE in the no-green/red self-check (the before-you-read neutral-blue rule): a bare substring grep for 'red' matches 'shared', 'bordered', 'hundred', 'covered' (any word ending -red or containing red) and forces a wasted re-check cycle; done loosely, it can also be wrongly trusted in the other direction. FIX: grep word-boundary regex '(\\bgreen\\b|\\bred\\b)' case-insensitive, or better, inventory all hex literals ('#[0-9a-fA-F]{6}') and confirm the palette list contains ONLY approved colors (e.g. #0066cc/#0066ff/#eef4ff for before-you-read). Discovered 2026-08-30 self-checking a before-you-read widget where the 'red' check matched the word 'shared'.

## color-audit-grep-word-boundaries

- SELF-CHECK PRECISION for the 'no green/red' color audits (neutral-blue feedback widgets, before-you-read MCQs): grep for forbidden color words with WORD BOUNDARIES, not bare substrings — pattern '\b(red|green)\b' (case-insensitive). A bare 'red' substring matches inside common words ('sha**red**', 'decla**red**', 'cove**red**', 'configu**red**') and produces a false positive that wastes a re-check cycle; bare 'green' can match 'gree**n**'-adjacent tokens similarly. Observed 2026-08-30: a done-report self-check flagged 'red' inside 'shared' until the regex was tightened. Best practice: audit COLORS by hex/name inventory scoped to CSS color-bearing properties (color:, background, border, fill, stroke) rather than free-text grep, then eyeball the inventory; use the word-boundary regex only as a coarse first pass.

## paginated-widget-render-discipline

- Two construction-time failure modes hit 2026-08-30 building before-you-read.html (priming MCQ, 3 questions, Back/Next pagination, 12 authored feedback panels). Both were self-caught during the post-write review, which is the point: bake them into the build so the review pass has nothing to catch.

(1) STATE MUTATION WITHOUT RE-RENDER. TRIGGER: a paginated widget whose navigation handler advances the page index directly (go() does current += 1) without routing through the render path. ROOT CAUSE: state mutation decoupled from DOM update, so Back/Next changed the index silently (content never swapped) and per-question answer state (saved selection + feedback) was never re-applied when revisiting an answered question. FIX: ONE render() function owns the DOM; it draws the current question AND re-applies that question's saved selection/feedback from the answers store; EVERY state mutation (option select, Back, Next, reset) ends by calling render(). CONSTRUCTION-TIME SELF-CHECK: answer a question, page forward, page back to it, confirm BOTH the selected option highlight AND its feedback panel are restored; click Next/Back on an unanswered page and confirm the question content actually swaps. Content that does not change on Back/Next is this bug.

(2) ABSENCE-CHECK SUBSTRING FALSE POSITIVE. TRIGGER: verifying a spec like 'no green/red anywhere' by grepping the widget source for the substring red (or green). ROOT CAUSE: color words are substrings of ordinary English words in copy and code, so the check false-alarms (red matched inside shared; same family: covered, rendered, declared) and trains you to distrust the check. FIX: word-boundary regexes (\bred\b, \bgreen\b) for word tokens, or stronger: extract the full hex inventory (grep -oE '#[0-9a-fA-F]{3,6}') and classify every hex value against the allowed palette; the hex inventory is the authoritative gate because correctness coloring lives in CSS hex values, not English words. Report the check as a per-hex classification (value, role), never as a bare substring count.

## second-lektor-repo-and-post-build-syntax-gate

- SURFACE: widgets are NOT exclusive to ~/github/website. A SECOND Lektor blog repo, ~/github/website-learn-anything-blog, hosts widget HTML files DIRECTLY in content/blog/<post-slug>/ alongside contents.lr (observed 2026-08-30: learning-how-to-learn-anything-together-in-the-age-of-ai/ contains before-you-read.html, cognitive-debt.html, does-this-resonate.html, retreat-week.html + logo.webp). When a widget task names a path in that repo, write there; its repo AGENTS.md carries the same voice rules (no em dashes). POST-BUILD SYNTAX GATE (add to the verification suite): extract the <script> block (sed -n '/<script>/,/<\/script>/p' | sed '1d;$d' > /tmp/widget.js) and run node --check on it — structural read-back does NOT prove the JS parses. TIER/THRESHOLD LOGIC: enumerate EVERY boundary input through the tier expression, not sample points (e.g. on=0..5 -> tier map 0->0 1->0 2->1 3->1 4->2 5->2; assert no mismatches), and verify gated elements (e.g. a CTA link shown only at the top tier) flip at the exact boundary. LINE BUDGET: if a written file exceeds the target length (e.g. 240 vs 150-220), compact CSS blocks (merge declarations onto shared lines, collapse one-rule blocks) rather than dropping features, then RE-RUN the structural checks on the compacted file (audit-before-rewrite applies: enumerate features before compacting, confirm all present after). NEW WIDGET PATTERN (non-quiz self-check): N statements, each a checkbox toggle DEFAULTING OFF (honest neutral state), result panel REVEALED only after every item has been touched (track a touched map on change events, not checked state, so un-toggling a previously-touched item does not re-lock the panel), tier copy selected by count thresholds with warm non-judgmental language. Instance 2026-08-30: does-this-resonate.html, 5 statements, tiers 0-1/2-3/4-5, retreat-site link gated to top tier only.

## warm self-check widget type (resonance toggles, not a quiz)

- ## Warm self-check widget type (resonance toggles, NOT a quiz)

A third reader-facing widget type alongside before-you-read (priming MCQ) and check-understanding (scored quiz): the WARM SELF-CHECK ('Does this resonate?') for announcement/invitation posts where the goal is RECOGNITION, not assessment. Spec as built (learn-anything does-this-resonate.html, 2026-08-30, in the second Lektor repo per the SURFACE note above):

- N (5) stacked statement rows; each row is a real checkbox visually styled as a toggle switch (input.tgl + label.switch with a .knob span + label.stmt, wired via for=id so the whole row is clickable). ALL SWITCHES OFF by default — off is the honest neutral state; the reader flips voluntarily.
- A live chip counter ('0 of 5 resonate') updates on every change.
- A reflection panel appears ONLY after EVERY row has been touched: track a touched{} map keyed by checkbox id, not just the checked-count — flip-on-then-off still counts as engaged. Until all-touched, the panel stays hidden (no partial judgment).
- The panel shows ONE of 3 TIERS keyed to the on-count (on<=1 / on<=3 / else) with warm second-person copy ('You are the exact personality the room is being curated for'). NO scores, NO correct answers, NO shaming; even the 0-of-5 tier is warm and no-pressure ('the ideas in this post are still yours to use today').
- The CTA link inside the panel is shown only on the TOP tier (flink.classList.toggle('on', tier === 2)).
- Footer: 'No wrong answers here.' + a Reset button that clears checkboxes AND the touched map, then update().
- Copy rules: warm, second person, no em dashes; the intro states the contract explicitly ('Nothing is scored and nothing is graded'). House style + fit() resize apply; call fit() inside update() so the iframe height tracks panel reveal.

Contrast with the quiz type: a quiz grades recall (correct/wrong, per-option explanations); a self-check reflects identity back to the reader, so feedback must reward honesty at EVERY tier and the interaction must be fully reversible (Reset) with no persistence.

## carousel-widgets-stage-floor-and-qa

- WIDGET TYPE (day-by-day flip-through; recurred 2026-07-24 quiz carousels, 2026-08-30 retreat-week.html): ONE card visible at a time, Back/Next buttons, a row of clickable day dots showing position, and a 'Day X of N' counter. Governs any Back/Next one-card-at-a-time UI (day-by-day timelines, quiz flip-throughs, Q&A slides, step decks).

- THE BUG - CAROUSEL HEIGHT JIGGLE: panels with UNEQUAL content lengths make the stage/iframe height change on every flip, because each slide change rewrites document.body.scrollHeight which fit() faithfully propagates (measured 2026-08-30 retreat-week: content cards 321px vs closing card 278px - every advance visibly resizes the frame and the dots/footer row jumps). A correct fit() SURFACES the jiggle rather than causing or hiding it.

- THE FIX - FLOOR THE STAGE TO THE MEASURED TALLEST PANEL: after building all panels (hidden), measure EVERY panel's natural height at the stage's actual rendered width (offscreen probe div set to stage.clientWidth; clone each panel + closing with 'hidden' removed; read offsetHeight or getBoundingClientRect().height), take the MAX, and floor the stage once: stage.style.minHeight = Math.max(measuredMax, saneMin). Measured, never eyeballed - card text length varies per post, so a carousel stage cannot be hand-pinned to a magic constant; the measured-max floor IS the fixed height, and every state then reports the SAME scrollHeight to fit(). All states share one stable stage; only the copy changes between slides.

- RESIZE WIRING: fit() on init + state changes is not enough - a window resize rewraps text and changes panel heights, so also wire window.addEventListener('resize', ...) to re-floor + fit(). Keep the per-state fit() calls (init, after Back/Next/dot-nav, after reset): flooring removes the between-slide jiggle; fit() still handles genuine size changes (and no-ops outside an iframe).

- RELATION TO THE FIXED-CONTAINER-HEIGHT RULE: that rule forbids min-height on a container whose child GROWS inside one track; the carousel variant is the sibling where the VISIBLE CHILD CHANGES between fixed children - the floor to the tallest measured panel supplies the fixed height. Both compose: fixed height AND tall enough for the tallest state.

- SYNTAX GATE BEFORE QA: extract the inline <script> and run node --check (or a headless load / console-error check) BEFORE behavioral QA. The 2026-08-30 retreat-week first draft shipped malformed markup, an incomplete fit(), and an undefined state variable that a read-back missed - caught only when the QA pass started. A 2-second syntax check prevents writing a known-broken draft.

- QA CHECKLIST (headless browser, NATIVE clicks, before declaring done - structural read-back is not enough): (1) initial state: N panels == N dots, counter text exact ('Day 1 of N'), Back hidden-or-disabled on first card, dot 0 active; (2) full forward walk: counter increments per Next, active dot tracks, each panel visible; (3) closing/last card renders fully, Next behaves per spec (no-op or loop); (4) Back boundary: no-op/hidden at first card; (5) dot-jump: clicking any dot lands that day with counter + active dot synced; (6) reset returns to the first card; (7) HEIGHT-STABILITY ASSERTION: record the card/stage height at EVERY state - all values equal proves the floor works; a per-slide delta like 278 vs 321 IS the jiggle (apply/verify the floor and re-drive).

- STATE-TRANSITION HEIGHT SUB-BUGS (2026-08-30, same widget): (1) a JS height override must never UNDERCUT the CSS design floor - the measured tallest panel was 158px while CSS said min-height:240px, so overriding with the raw measurement made panels SHORTER than CSS alone (compute maxH = Math.max(measuredMaxH, CSS_FLOOR)); (2) the closing state must not SHRINK by a removed controls row (43px) - keep the row's LAYOUT SPACE in every state (visibility:hidden or an invisible spacer, never display:none/removal). RESULT: constant 321px across all states; the verification habit is asserting identical container height across every state (initial -> each step -> closing).

- QA AUTOMATION GOTCHAS (2026-08-30, retreat-week carousel QA): (a) SINGLE-QUOTE NESTING breaks inline eval loops - a loop that quotes button labels inside single quotes collides with the eval/shell quoting layers and dies; prefer the automation tool's find-role clicks with clear visible marker text over building eval loops. (b) BACK-STEP ARTIFACT - a click aimed at step N's button can land on step N-1's still-mounted UI (state lag), so a transition appears to pass against the WRONG state; after any suspicious step, re-verify the real sequence using the button labels that actually EXIST at each step (confirm the target button is present in the CURRENT DOM before clicking it).

## widget-content-fidelity-source-only-paraphrase

- When a widget card or panel copy is sourced from a blog post or document (retreat-week carousel, quiz built from an article), paraphrase ONLY that source: no invented facts, numbers, names, day counts, or claims, no embellishment beyond the source spec. Commission phrasing to watch for: paraphrase ONLY the content of X (no new facts). During QA, verify every card claim traces back to the source spec, not to plausible-sounding filler.

## genre-appropriate-widget-selection

- ## Genre-appropriate widget selection (marketing vs teaching posts)

- Classify the post's GENRE BEFORE proposing widgets. Teaching/concept posts: before-you-read priming MCQ + check-understanding quiz are the defaults (see the pedagogical sections above). MARKETING/invitation posts (product announcements, retreat promos, waitlist pushes): the reader is being INVITED, not taught — a priming quiz 'reads like a classroom'. Eric 2026-08-30 (learn-anything post): 'this is not a blog post that is teaching a concept, it's a bit more of a marketing-type blog post, we don't necessarily have to do a build-before-you-read widget here.'
- Marketing-post widgets must do MARKETING WORK — each widget IS one of the post's arguments, not decoration: (1) the core argument made visceral (e.g. a cognitive-debt visual); (2) the product made walkable (e.g. a retreat-week day carousel); (3) a warm does-this-resonate self-check that lets the reader qualify themselves (recognition, not assessment — see the warm self-check section).
- Standing trigger: when a draft 'reads like a wall of text', propose interactive breaks in the spirit of the most recent posts (ai-turbocharges-learning 6 widgets, quantum-ml 7, going-bayesian 4) — but pick widget TYPES by genre, and when the user pre-steers ('we don't necessarily have to do X here'), X must not appear in the proposed widget set anyway.

## vault-draft-widget-annotations

- VAULT DRAFT EMBEDS ARE COMMENTS, NOT IFRAMES: the brain42 vault draft of a widget-bearing post cannot render iframes (Obsidian markdown), so each embed spot in the vault draft carries an HTML-comment annotation — '<!-- widget: <widget-file>.html — <one-line what-it-does> -->' — NOT a raw <iframe> tag. Only the website side (contents.lr) gets the real iframe embeds (border:none, repo convention). When syncing widgets into BOTH surfaces, keep annotation and iframe in 1:1 correspondence (same widget file, same position in the post); if a raw iframe leaks into the vault draft, swap it for the comment form and re-run markdownlint on the draft. Observed 2026-08-30 (learn-anything retreat post, PR #294): the vault draft's existing widget annotations used the comment convention and a fresh embed initially inserted a raw iframe before being swapped — the convention existed in practice but was undocumented.

## clipboard-on-file-and-notes-loopback

- CLIPBOARD IS SECURE-CONTEXT-GATED: `navigator.clipboard` is unavailable or rejects on `file://` pages — and locally-opened self-contained HTML tools (review consoles, offline calculators, repo-local QA dashboards) are ALWAYS `file://`, so a bare `navigator.clipboard.writeText()` copy button silently fails there. Every copy button needs the fallback ladder: try `navigator.clipboard` first; on rejection/absence, create a hidden `<textarea>`, select it, `document.execCommand('copy')`, remove it; surface a visible success/failure status either way so a silent clipboard dead-end never masquerades as a working button. (2026-09-23, legal-docs tc-review-console copy-notes-for-chat button.)
- NOTES-BEARING TOOLS MUST LOOP BACK INTO CHAT: any annotation/persistent-notes UI needs a one-click 'Copy notes for chat' affordance NEXT TO the file-export button — Eric's workflow is export-to-Downloads THEN paste his comments back into the conversation for triage, so export alone does not close the loop (his words: 'you should give me a way to just copy/paste all comments I left into here as well'). The copy block should be compact (section → note lines, from the same localStorage the notes UI persists to), not the full markdown brief the Export button produces.

## deploy-preview-verification-gate-post-push-pre-handover

- Deploy-preview verification gate (post-push, pre-handover): after pushing widget or iframe commits to an open PR and CI goes green, verify the DEPLOYED preview before reporting done; a green build does not prove the assets serve. One pass per PR: (1) each widget asset URL on the preview host returns 200 (curl -s -o /dev/null -w "%{http_code}"); (2) the post page returns 200 AND its HTML embeds every iframe — grep the src filenames against the fetched page HTML; (3) the site root returns 200; (4) re-run the house-style grep gates against the committed files: fit() guarded write present in each widget (height-equality guard before the 0px collapse), palette tokens exact, 12px font floor, iframe border: none, contents.lr --- separator count intact. TRANSIENT-000 RULE: one curl 000 (connection failure) on a URL that returned 200 on the immediately preceding probes is a transient network or CDN blip, NOT a deploy regression — re-probe once and report the confirmed state; do not alarm or re-trigger the pipeline over a single blip. Trigger: 000 on the latest probe with 200s on prior probes of the same host; root cause: transient network-path failure; fix: re-probe and confirm. Instance 2026-08-30 website PR #294 (three-widget learning-how-to-learn post): all three assets and the post page 200 with all three iframes embedded, one transient root 000 re-confirmed 200. Pairs with the ship-report rule: lead the completion report with the verified preview URL.
