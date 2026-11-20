# Marimo islands on this site

Verdict: feasible, and worth it for a single interactive viewer inside a post. This change adds the plumbing and switches the protein viewer in [Agent-assisted data science with marimo notebooks](https://ericmjl.github.io/blog/2026/10/14/agent-assisted-data-science-with-marimo/) from a full marimo WebAssembly iframe to an inline island. The notebook source is unchanged. Drag to rotate and scroll to zoom still work. Python still runs in the reader's browser.

Islands are a preview API in marimo. The generator and the CDN package are pinned to marimo 0.25.2 so a later release does not silently change the fragment.

## How the site is built

The site is [Lektor](https://www.getlektor.com/) 3.4 (`personal-site.lektorproject`). Posts are `content/blog/<slug>/contents.lr` files. The body field is Markdown. Raw HTML blocks in that Markdown are passed through, which is how the old iframe was embedded. Templates live in `templates/`. `templates/blog-post.html` renders `templates/macros/blog.html`. Styles are terminal.css plus `assets/static/css/custom.css`, including a `.dark-mode` class on `<body>`.

`pixi run build` writes the search index, then `lektor build --output-path site/`. Pushing `main` runs `.github/workflows/build-website.yml` and publishes `site/` to `ericmjl/ericmjl.github.io`. Pull requests run `.github/workflows/preview-website.yml`, which builds the same way and deploys a Netlify preview.

## How the old embed worked

`protein-anywidget.py` is a three-cell marimo notebook: imports, an anywidget `ProteinViewer` whose JavaScript loads 3Dmol and fetches PDB 1UBQ from rcsb.org, and a cell that constructs the viewer. `protein-anywidget.html` was the output of `marimo export html-wasm --single-file` for marimo 0.25.1, mode `read`. The file is about 29 KB. It does not contain Python or Pyodide. It points the browser at `@marimo-team/frontend@0.25.1` (a long modulepreload list) and embeds the notebook source in a `<marimo-code>` element. The post loaded that file in an iframe, 540 px tall. The iframe kept marimo's CSS, fonts, and the "made with marimo" chrome out of the blog page. It also meant the viewer scrolled as a nested document.

## How islands are generated

[`MarimoIslandGenerator`](https://docs.marimo.io/guides/exporting/#embed-marimo-outputs-in-html-using-islands) turns notebook cells into `<marimo-island>` elements plus a JSON payload (`application/vnd.marimo.islands+json`). On load, `@marimo-team/islands` boots Pyodide, installs the notebook's PEP 723 dependencies, and runs the cells.

`scripts/build_marimo_islands.py` does that offline:

```bash
uv run scripts/build_marimo_islands.py content/blog/<slug>/notebook.py
```

The script re-invokes itself with `uv run --with marimo==0.25.2` and whatever else the notebook declares. It writes `<notebook-stem>.islands.html` beside the notebook. The fragment is the islands `<head>` tags, every cell, and the payload, wrapped in `<div class="marimo-islands-host">`.

Cells are not executed during generation. `generator.build()` does run them, and for this anywidget the preview HTML includes a new `random-id` and `model_id` on every run. The browser executes the cells regardless, so committing that preview would churn the git diff for no reader-visible gain.

The stock init island (`render_init_island()`) was left out on purpose. In marimo 0.25.2 it drew "Initializing..." and, in a headless Chrome check, left that spinner on screen after the kernel had finished and the canvas was visible. The fragment has its own one-line status, removed once a canvas exists.

## Where it fits in the Lektor build

The post body keeps a one-line placeholder:

```html
<div class="marimo-islands" data-fragment="protein-anywidget.islands.html"></div>
```

`packages/lektor-marimo-islands` is a local Lektor plugin (installed editable into Lektor's package cache, the same way the PyPI plugins in `personal-site.lektorproject` are installed). It registers a Jinja filter, `inject_marimo_islands`, used by `render_blog_post`. The filter replaces the placeholder with the fragment and records the fragment as a build dependency. Markdown never parses the custom elements or the JSON payload. One island app per page. The placeholder name has to stay inside the post directory.

Generation is not part of `pixi run build`. CI publishes the committed fragment. Regenerate the fragment when the notebook changes, and commit both files.

## Assets, theme, and page weight

`render_head()` for 0.25.2 requests:

- `https://cdn.jsdelivr.net/npm/@marimo-team/islands@0.25.2/dist/main.js` (about 1.0 MB)
- `.../dist/style.css` (about 0.46 MB)
- Google fonts (Fira Mono, Lora, PT Sans) and KaTeX CSS, which the islands stylesheet expects

Those are not copied into the repo. The browser also downloads Pyodide and the wheels named in the notebook (`marimo`, `anywidget`, and anywidget's dependency `traitlets`), then 3Dmol from cdnjs and `1UBQ.pdb` from rcsb.org. The page has to be served over HTTP. A `file://` open will not boot the island. Same constraint as the old export.

The islands stylesheet is Tailwind scoped under `.marimo`, so it does not restyle the blog. It is a preferred stylesheet (`title="marimo-islands"`). Persistent site CSS still applies. terminal.css styles `input` and `select`. It does not style `button` or `canvas`, which is why this viewer is mostly unaffected. A future island that shows a marimo slider or dropdown will pick up terminal.css control styles, and will need a host-level reset.

Dark mode sets `color: #fff` on `body`. The host is pinned to a light panel in `custom.css` so the viewer matches the old light iframe and the status text stays readable. `overflow-anchor: none` is set on the host for the same scroll-anchoring reason the site already sets it on iframes.

Compared with the old embed: the HTML we ship is a similar size (the fragment is about 10 KB, the old file was 29 KB), and the reader still pays for Pyodide. The islands JavaScript is one file instead of the full marimo editor and its modulepreload graph, and the page does not get the editor chrome or a nested scrollport. The cost is that marimo's CSS and a Pyodide runtime now live in the parent page, and a post pays that cost even for readers who never scroll to the viewer, because the module tag is in the article HTML rather than a lazily loaded iframe. The old iframe used `loading="lazy"`. This one starts when the article is parsed.

## Tradeoffs

| | WASM iframe (before) | Islands (this change) |
| --- | --- | --- |
| Reader runtime | Pyodide in an iframe | Pyodide in the page |
| Frontend | Full `@marimo-team/frontend` editor shell | `@marimo-team/islands` |
| Layout | Fixed 540 px frame, own scroll | In the article flow |
| Theme isolation | Strong (separate document) | Host panel plus scoped `.marimo` CSS |
| Notebook source | Embedded in the HTML export | Still `protein-anywidget.py`, plus a generated fragment |
| Build | Commit the export | Commit the fragment; Lektor inlines it |
| Failure mode | Iframe stays on "Loading marimo..." | Status line stays if no canvas appears |

anywidget works. The islands bundle contains marimo's anywidget runtime, and this notebook's viewer cell produced a 3Dmol canvas. Islands are the wrong tool if a notebook needs the full editor, local files, or packages without a WASM wheel. For those, keep a WASM export or link to molab.

## Checks that were run

- `MarimoIslandGenerator.from_file(...).build()` executes this notebook under CPython and emits an anywidget HTML preview. That preview was not committed, for the random-id reason above.
- A standalone islands page, served over HTTP, produced a canvas in headless Chrome with software WebGL. Dragging the canvas changed the orientation of the cartoon. Without WebGL, the same page ran the cells and showed 3Dmol's "error creating viewer" message, which is the widget failing, not the island failing to boot.
- The same Chrome profile rendered the old `protein-anywidget.html` locally as the full marimo app ("made with marimo" badge, overflow menu, same cartoon).
- The production iframe at `https://ericmjl.github.io/blog/2026/10/14/agent-assisted-data-science-with-marimo/` was still on "Loading marimo..." after three minutes in that same session. The identical HTML served locally rendered in about ten seconds once Pyodide was cached, so this is not evidence that the live embed is broken. It is evidence that the full frontend is a heavy cold start.

## Adding another island

1. Put a marimo `.py` notebook in the post directory, with PEP 723 dependencies.
2. Run `uv run scripts/build_marimo_islands.py` on it.
3. Add the placeholder div to the body. One fragment per page. The app id is `main`.
4. Commit the notebook and the `.islands.html` file.

If the notebook imports something Pyodide cannot install, the island will show a loader and then a cell error. `uvx marimo check <notebook>.py` is the local syntax check. It does not prove the WASM install.
