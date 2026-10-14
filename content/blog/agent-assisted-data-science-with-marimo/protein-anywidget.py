# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "anywidget",
# ]
# ///
"""A 3D protein viewer built as a true anywidget, running in marimo."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import anywidget
    import marimo as mo
    import traitlets

    return anywidget, mo, traitlets


@app.cell
def _(anywidget, traitlets):
    class ProteinViewer(anywidget.AnyWidget):
        """Render any PDB entry from the RCSB as a 3D cartoon."""

        _esm = """
        function load3Dmol() {
          return new Promise((resolve, reject) => {
            if (window.$3Dmol) { resolve(window.$3Dmol); return; }
            const script = document.createElement("script");
            script.src = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.0/3Dmol-min.js";
            script.onload = () => resolve(window.$3Dmol);
            script.onerror = () => reject(new Error("could not load 3Dmol.js"));
            document.head.appendChild(script);
          });
        }

        async function render({ model, el }) {
          el.style.position = "relative";
          const status = document.createElement("div");
          status.style.cssText =
            "font:12px ui-monospace,monospace;color:#666;padding:6px 2px;";
          el.appendChild(status);

          async function show(pdbId) {
            status.textContent = "loading " + pdbId + " from rcsb.org...";
            const $ = await load3Dmol();
            el.querySelectorAll(".pv-container").forEach((n) => n.remove());
            const container = document.createElement("div");
            container.className = "pv-container";
            container.style.height = model.get("height") + "px";
            el.appendChild(container);
            const viewer = $.createViewer(container, { backgroundColor: "white" });
            const response = await fetch(
              "https://files.rcsb.org/download/" + pdbId + ".pdb"
            );
            if (!response.ok) {
              status.textContent =
                "rcsb.org returned " + response.status + " for " + pdbId;
              return;
            }
            viewer.addModel(await response.text(), "pdb");
            viewer.setStyle({}, { cartoon: { color: "spectrum" } });
            viewer.zoomTo();
            viewer.render();
            status.textContent =
              pdbId + " \u00b7 drag to rotate, scroll to zoom";
          }

          await show(model.get("pdb_id"));
          model.on("change:pdb_id", () => show(model.get("pdb_id")));
        }

        export default { render };
        """

        pdb_id = traitlets.Unicode("1UBQ").tag(sync=True)
        height = traitlets.Int(440).tag(sync=True)

    return (ProteinViewer,)


@app.cell
def _(ProteinViewer):
    protein_viewer = ProteinViewer(pdb_id="1UBQ", height=440)
    protein_viewer
    return (protein_viewer,)


if __name__ == "__main__":
    app.run()
