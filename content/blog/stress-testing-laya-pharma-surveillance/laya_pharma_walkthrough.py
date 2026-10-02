# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
"""Companion notebook for the blog post on stress-testing Laya for pharma
document triage. All numbers are inlined from the actual experiment runs
(finetune_experiment_results.json and the deployment benchmarks), so this
notebook runs standalone with no model downloads.

Run it with: uvx marimo edit --sandbox <notebook-url>
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Stress-testing Laya for pharma document triage

        This notebook walks through the numbers behind the blog post: we took
        [Laya](https://huggingface.co/convaiinnovations/laya), an open-source
        non-autoregressive "System 1" decision model, and ran it through a
        pharmacovigilance-style document triage workload, then fine-tuned it
        with the RLCD recipe and compared the probabilities before and after.

        Every number below is inlined from the actual experiment runs, so this
        notebook needs no model downloads.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## The corpus: 265 real pharma documents")
    return


@app.cell
def _(mo, plt):
    counts = {
        "scientific_publication": 40,
        "regulatory_guidance": 37,
        "corporate_report": 37,
        "press_coverage": 30,
        "spam_or_irrelevant": 30,
        "safety_communication": 28,
        "inspection_finding": 22,
        "commercial": 15,
        "legal": 13,
        "public_comment": 13,
    }
    sources = {
        "scientific_publication": "PubMed E-utilities",
        "regulatory_guidance": "Federal Register API",
        "corporate_report": "SEC EDGAR full-text search",
        "press_coverage": "FiercePharma RSS",
        "spam_or_irrelevant": "synthetic (marked as such)",
        "safety_communication": "fda.gov Drug Safety Communications",
        "inspection_finding": "fda.gov warning letters",
        "commercial": "vendor press releases",
        "legal": "DOJ enforcement pages + settlement PDFs",
        "public_comment": "curated docket-comment letters",
    }
    cats = sorted(counts, key=counts.get)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(cats, [counts[c] for c in cats], color="#4a7ebb")
    ax.set_xlabel("documents")
    ax.set_title("10 categories x public sources (script-side fetching)")
    fig.tight_layout()
    fig
    return (counts, sources)


@app.cell
def _(mo):
    mo.md(
        """
        ## The task: five typed questions per document, one forward pass

        Laya never generates text. Each document gets a schema of typed
        questions: two `choice` questions (primary + secondary category),
        one `choice` for the owning team, one `noul` (calibrated yes/no) for
        actionability, and one `choice` for urgency. Labels come from the
        corpus construction rules. Split: 212 train / 53 test documents,
        stratified per category.
        """
    )
    return


@app.cell
def _(mo, plt, np):
    questions = ["doc_type", "secondary_type", "route_to", "action_required", "urgency"]
    acc_before = [0.585, 0.038, 0.264, 0.434, 0.358]
    acc_after = [0.943, 0.962, 0.887, 0.906, 0.887]

    x = np.arange(len(questions))
    w = 0.38
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.bar(x - w / 2, acc_before, w, label="before fine-tuning", color="#a8b2bd")
    ax2.bar(x + w / 2, acc_after, w, label="after RLCD fine-tuning", color="#4ade80")
    ax2.axhline(1 / 10, color="grey", linestyle=":", linewidth=1)
    ax2.text(
        len(questions) - 0.5, 1 / 10 + 0.01, "10-way random", fontsize=8, color="grey"
    )
    ax2.set_xticks(x, questions, rotation=12)
    ax2.set_ylabel("held-out accuracy (53 test docs)")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    fig2.tight_layout()
    fig2
    return (acc_after, acc_before)


@app.cell
def _(mo):
    mo.md(
        """
        Fine-tuning improved every held-out metric, not just argmax accuracy.
        Log score (higher is better) and Brier score (lower is better) measure
        the *quality of the whole probability distribution*, and both moved by
        2-6x across the board.
        """
    )
    return


@app.cell
def _(mo, plt, np):
    qs = ["doc_type", "secondary", "route", "action", "urgency"]
    ls_before = [-1.351, -3.960, -2.399, -2.183, -1.222]
    ls_after = [-0.270, -0.655, -0.600, -0.466, -0.137]
    br_before = [0.585, 1.425, 0.963, 0.955, 0.730]
    br_after = [0.165, 0.303, 0.265, 0.284, 0.096]

    fig3, (axl, axb) = plt.subplots(1, 2, figsize=(11, 4))
    x3 = np.arange(len(qs))
    axl.bar(x3 - 0.19, ls_before, 0.38, label="before", color="#a8b2bd")
    axl.bar(x3 + 0.19, ls_after, 0.38, label="after", color="#4ade80")
    axl.set_title("mean log score (higher better)")
    axl.set_xticks(x3, qs, rotation=12)
    axl.legend()
    axb.bar(x3 - 0.19, br_before, 0.38, label="before", color="#a8b2bd")
    axb.bar(x3 + 0.19, br_after, 0.38, label="after", color="#4ade80")
    axb.set_title("Brier score (lower better)")
    axb.set_xticks(x3, qs, rotation=12)
    axb.legend()
    fig3.tight_layout()
    fig3
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## The calibration story: fitted temperatures

        Temperature scaling divides a model's logits by one scalar per
        (question type, option count) bucket, fitted on held-out data by
        minimizing negative log likelihood. The fitted value diagnoses the
        model: **T > 1 means over-confident** (probabilities need flattening);
        **T < 1 means under-confident**.

        On this domain the base model's noul bucket wanted T = 100 (it pegged
        the search ceiling): grossly over-confident. After RLCD fine-tuning -
        whose reward is a strictly proper scoring rule, so honest probabilities
        are the only way to win - the choice buckets landed at T = 0.11-0.73.
        Fine-tuning did the calibration work as a side effect of the objective.
        """
    )
    return


@app.cell
def _(mo, plt, np):
    buckets = ["choice:3-5", "choice:6-10", "choice:11+", "noul:2"]
    T_before = [2.086, 4.833, 13.591, 100.0]
    T_after = [0.732, 0.536, 0.108, 0.152]

    x4 = np.arange(len(buckets))
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.bar(x4 - 0.19, T_before, 0.38, label="before", color="#a8b2bd")
    ax4.bar(x4 + 0.19, T_after, 0.38, label="after", color="#4ade80")
    ax4.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ax4.text(
        len(buckets) - 0.5, 1.6, "T = 1: perfectly scaled", fontsize=8, color="red"
    )
    ax4.set_yscale("log")
    ax4.set_ylabel("fitted temperature (log scale)")
    ax4.set_xticks(x4, buckets, rotation=12)
    ax4.set_title(
        "the base model needed T=100; fine-tuning pulled temperatures toward T=1"
    )
    ax4.legend()
    fig4.tight_layout()
    fig4
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### ECE on the actionability question

        Expected Calibration Error for the noul head, on held-out docs:
        raw ECE fell from 0.463 to 0.261 with fine-tuning alone, and the
        post-hoc temperature refit landed both models near 0.07. The
        difference: the base model reaches 0.07 only through extreme softening
        (T = 100 destroys most of the signal), while the fine-tuned model
        arrives with usable probabilities natively.

        Calibration is separate from discrimination: temperature scaling never
        changes which answer ranks first, only how much you should trust it.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Deployment: the cheapest version still wins on cost")
    return


@app.cell
def _(mo):
    mo.md(
        """
        Deployed on Modal, GitHub-CI-only, scale-to-zero after 60 seconds idle:

        | profile | cold start | p50 | thruput (8c) | rate | $ per 1M requests |
        |---|---|---|---|---|---|
        | CPU (1 core, 3 GiB) | 3.5 s | 3.31 s | 1.2 req/s | \\$0.071/h | \\$17-68 |
        | T4 GPU (2c, 8 GiB) | 44.3 s | 0.48 s | 8.0 req/s | \\$0.749/h | \\$26-115 |

        Server-side inference is 49 ms on the T4 (matching the model card) and
        about 2.9 s on CPU. The CPU profile wins on cost at either concurrency;
        the T4 wins on latency by 7x. All of it runs behind a GitHub Actions
        deploy pipeline with smoke tests on both profiles.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Takeaways

        1. An open 421M-parameter decision model handles well-scoped triage
           schemas zero-shot and becomes genuinely usable (0.89-0.96 held-out
           accuracy) after a ~20-minute fine-tune on 226 documents.
        2. Question phrasing is part of the model: a negatively-phrased
           instruction inverted the yes/no head on both checkpoints. Keep
           decision questions short and positively phrased.
        3. Calibration is a two-step story: fine-tuning on proper scoring
           rules does the heavy lifting in-domain, and a per-bucket
           temperature refit on your own labels finishes the job.
        4. The cheap deployment is the CPU one, and for batch triage it is
           plenty. Reach for the GPU when latency matters.

        Model: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
        (Apache 2.0). Notebook and experiment code ship alongside the post.
        """
    )
    return


if __name__ == "__main__":
    app.run()
