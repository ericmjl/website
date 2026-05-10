# /// script
# dependencies = [
#     "ipython==9.13.0",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.4",
#     "pandas==3.0.2",
#     "plotly==6.7.0",
#     "polars==1.40.1",
#     "scikit-learn==1.8.0",
#     "sentence-transformers==5.4.1",
#     "umap-learn==0.5.12",
#     "wordcloud==1.9.6",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def title_and_overview(mo):
    mo.md(r"""
    # ODSC East 2026: Exploring the Conference Zeitgeist

    This notebook accompanies the blog post analyzing the zeitgeist of ODSC East 2026.

    The experiment was simple: rather than trying to attend every talk, what
    if we could understand the conference's center of gravity by analyzing all
    talk abstracts? Using embeddings, clustering, and a multi-agent
    categorization pipeline, we identified five major thematic zones across 193
    substantive sessions.

    This interactive visualization lets you explore both:
    - **LLM-based zones**: Five categories identified through multi-agent consensus
    - **Embedding clusters**: Unsupervised HDBSCAN clusters based on semantic similarity

    The differences between these views reveal interesting patterns in how the
    conference community organizes itself.
    """)
    return


@app.cell(hide_code=True)
def data_loading_intro(mo):
    mo.md(r"""
    ## Data Loading

    We start by loading the conference schedule data scraped from the ODSC East
    2026 schedule backend. This JSON file contains 237 sessions with metadata
    including titles, abstracts, speakers, tracks, and scheduling information.
    """)
    return


@app.cell(hide_code=True)
def load_schedule_data():
    import json

    with open("odsc-east-2026-schedule-sessions.json") as f:
        data = json.load(f)
    sessions = data["sessions"]
    len(sessions)
    return (sessions,)


@app.cell(hide_code=True)
def data_preparation_intro(mo):
    mo.md(r"""
    ## Data Preparation

    From the 237 sessions, we extract:
    - **200 sessions** with non-empty abstracts
    - **193 substantive sessions** after filtering out non-content items
      (badge pickup, networking, etc.)

    We'll extract abstracts, titles, speakers, tracks, and company affiliations
    for analysis.
    """)
    return


@app.cell(hide_code=True)
def extract_session_metadata(sessions):
    abstracts = [s["session_abstract"] for s in sessions if s.get("session_abstract")]
    titles = [s["session_title"] for s in sessions if s.get("session_abstract")]
    speakers = [s["full_name"] for s in sessions if s.get("session_abstract")]
    tracks = [s["track"] for s in sessions if s.get("session_abstract")]
    len(abstracts)
    return abstracts, speakers, titles, tracks


@app.cell(hide_code=True)
def embeddings_intro(mo):
    mo.md(r"""
    ## Semantic Embeddings

    To capture the semantic content of each abstract, we use the
    `all-MiniLM-L6-v2` sentence transformer model. This creates a
    384-dimensional vector representation for each abstract, where abstracts
    with similar content will have similar vectors.
    """)
    return


@app.cell(hide_code=True)
def create_embeddings(abstracts):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(abstracts, show_progress_bar=False)
    embeddings.shape
    return (embeddings,)


@app.cell(hide_code=True)
def clustering_intro(mo):
    mo.md(r"""
    ## Unsupervised Clustering with HDBSCAN

    HDBSCAN (Hierarchical Density-Based Spatial Clustering) identifies natural
    groupings in the embedding space. Unlike k-means, it doesn't require
    specifying the number of clusters upfront and can identify "noise" points
    that don't belong to any cluster.

    This gives us an unsupervised view of how talks naturally group based on
    semantic similarity.
    """)
    return


@app.cell(hide_code=True)
def hdbscan_clustering(embeddings):
    from sklearn.cluster import HDBSCAN

    clusterer = HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    f"Found {n_clusters} clusters, {list(labels).count(-1)} noise points"
    return (labels,)


@app.cell(hide_code=True)
def umap_intro(mo):
    mo.md(r"""
    ## Dimensionality Reduction with UMAP

    To visualize the 384-dimensional embeddings in 2D, we use UMAP (Uniform
    Manifold Approximation and Projection). UMAP preserves both local and
    global structure better than alternatives like t-SNE, meaning nearby points
    in 2D are also similar in the original high-dimensional space.
    """)
    return


@app.cell(hide_code=True)
def umap_projection(embeddings):
    import umap

    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)
    coords.shape
    return (coords,)


@app.cell(hide_code=True)
def visualization_intro(mo):
    mo.md(r"""
    ## Interactive Visualization

    The scatter plot below shows all 193 substantive sessions positioned based
    on semantic similarity. You can toggle between two views:

    1. **LLM-based zones**: Five categories identified through a multi-agent
       consensus process
    2. **Embedding clusters**: Unsupervised HDBSCAN clusters

    Hover over any point to see the full talk details. Use box select or lasso
    to explore groups of talks.
    """)
    return


@app.cell(hide_code=True)
def llm_zones_intro(mo):
    mo.md(r"""
    ## LLM-Based Zone Classification

    Beyond unsupervised clustering, we ran a multi-agent categorization pipeline:

    1. **Four coding agents** each read 50 abstracts and proposed a
       five-category taxonomy
    2. **Three cross-review agents** read all proposals and abstracts,
       identifying agreement/disagreement
    3. **One arbitrator agent** resolved disputes using majority vote with
       direct abstract review for tiebreaks

    This process identified **five thematic zones**:

    - **Zone 1: Agentic AI Systems** (55 talks, 28%) - Agent architectures,
      orchestration, tool use, production deployment
    - **Zone 2: LLM & Foundation Model Engineering** (37 talks, 19%) -
      Training, fine-tuning, inference optimization, RAG
    - **Zone 3: Data Engineering & ML Infrastructure** (28 talks, 15%) -
      Data platforms, pipelines, quality frameworks
    - **Zone 4: Applied AI, Domain Solutions & Foundations** (41 talks, 21%)
      - Domain applications and core ML training
    - **Zone 5: AI Strategy, Governance & Workforce** (32 talks, 17%) -
      Enterprise strategy, governance, trust engineering

    The zone assignments below map session indices to their categories.
    """)
    return


@app.cell(hide_code=True)
def zone_assignments(titles):
    agentic = {
        6,
        7,
        22,
        26,
        29,
        30,
        36,
        37,
        41,
        45,
        50,
        54,
        55,
        57,
        58,
        60,
        69,
        75,
        76,
        78,
        80,
        82,
        84,
        93,
        106,
        107,
        111,
        114,
        115,
        116,
        117,
        119,
        123,
        124,
        128,
        129,
        130,
        131,
        132,
        135,
        137,
        141,
        145,
        147,
        150,
        152,
        153,
        157,
        160,
        165,
        168,
        169,
        183,
        194,
        195,
    }
    llm_eng = {
        0,
        4,
        5,
        10,
        11,
        12,
        13,
        14,
        15,
        17,
        19,
        35,
        46,
        47,
        51,
        63,
        65,
        77,
        88,
        89,
        100,
        105,
        108,
        109,
        122,
        136,
        142,
        155,
        159,
        162,
        163,
        166,
        176,
        181,
        182,
        196,
        197,
    }
    data_infra = {
        25,
        34,
        38,
        42,
        49,
        52,
        62,
        68,
        73,
        74,
        79,
        83,
        92,
        95,
        98,
        101,
        103,
        113,
        125,
        138,
        139,
        143,
        148,
        154,
        161,
        167,
        178,
        187,
    }
    applied = {
        1,
        2,
        3,
        8,
        9,
        16,
        18,
        20,
        21,
        23,
        24,
        27,
        31,
        32,
        33,
        44,
        48,
        70,
        71,
        81,
        85,
        86,
        87,
        94,
        96,
        99,
        104,
        118,
        121,
        127,
        133,
        134,
        140,
        156,
        158,
        170,
        171,
        174,
        175,
        177,
        199,
    }
    strategy = {
        28,
        39,
        40,
        43,
        53,
        56,
        59,
        61,
        64,
        67,
        72,
        90,
        91,
        97,
        102,
        110,
        112,
        120,
        126,
        144,
        146,
        149,
        151,
        164,
        172,
        179,
        184,
        185,
        186,
        192,
        193,
        198,
    }
    _excluded = {66, 173, 180, 188, 189, 190, 191}

    cat_map = {}
    for i in agentic:
        cat_map[i] = "Agentic AI Systems"
    for i in llm_eng:
        cat_map[i] = "LLM & Foundation Model Engineering"
    for i in data_infra:
        cat_map[i] = "Data Engineering & ML Infrastructure"
    for i in applied:
        cat_map[i] = "Applied AI, Domain Solutions & Foundations"
    for i in strategy:
        cat_map[i] = "AI Strategy, Governance & Workforce"

    agent_labels = [cat_map.get(i, "Excluded") for i in range(len(titles))]
    # len(agent_labels), {c: agent_labels.count(c) for c in set(agent_labels)}
    return agent_labels, agentic, applied, data_infra, llm_eng, strategy


@app.cell(hide_code=True)
def zone_distribution_intro(mo):
    mo.md(r"""
    ### Distribution Across Zones

    The bar chart below shows how the 193 substantive talks are distributed
    across the five thematic zones. Agentic AI Systems emerges as the dominant
    theme, accounting for 28% of all substantive sessions.
    """)
    return


@app.cell(hide_code=True)
def zone_distribution_chart(
    agentic,
    applied,
    data_infra,
    go,
    llm_eng,
    mo,
    strategy,
):
    # Count talks in each zone
    zone_counts = {
        "Agentic AI Systems": len(agentic),
        "LLM & Foundation Model Engineering": len(llm_eng),
        "Applied AI, Domain Solutions & Foundations": len(applied),
        "Data Engineering & ML Infrastructure": len(data_infra),
        "AI Strategy, Governance & Workforce": len(strategy),
    }

    # Create bar chart
    zone_fig = go.Figure(
        data=[
            go.Bar(
                x=list(zone_counts.keys()),
                y=list(zone_counts.values()),
                text=list(zone_counts.values()),
                textposition="auto",
                marker=dict(
                    color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
                ),
            )
        ]
    )

    zone_fig.update_layout(
        title="Distribution of Talks Across Five Thematic Zones",
        xaxis_title="Zone",
        yaxis_title="Number of Talks",
        xaxis=dict(tickangle=-45),
        height=500,
        showlegend=False,
    )

    zone_chart = mo.ui.plotly(zone_fig)
    zone_chart
    return


@app.cell(hide_code=True)
def wordcloud_intro(mo):
    mo.md(r"""
    ### Zone-Specific Word Clouds

    Explore the key themes and vocabulary within each zone through word clouds
    generated from talk abstracts. Select a zone below to see which terms
    appear most frequently in that category.
    """)
    return


@app.cell(hide_code=True)
def precompute_wordclouds(
    abstracts,
    agentic,
    applied,
    data_infra,
    llm_eng,
    strategy,
):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    # Function to generate word cloud from abstracts
    def generate_wordcloud_for_zone(zone_indices, abstracts_list):
        zone_abstracts = [
            abstracts_list[i] for i in zone_indices if i < len(abstracts_list)
        ]
        combined_text = " ".join(zone_abstracts)

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10,
        ).generate(combined_text)

        return wc

    # Pre-compute word clouds for all zones
    wordclouds = {
        "Agentic AI Systems": generate_wordcloud_for_zone(agentic, abstracts),
        "LLM & Foundation Model Engineering": generate_wordcloud_for_zone(
            llm_eng, abstracts
        ),
        "Applied AI, Domain Solutions & Foundations": generate_wordcloud_for_zone(
            applied, abstracts
        ),
        "Data Engineering & ML Infrastructure": generate_wordcloud_for_zone(
            data_infra, abstracts
        ),
        "AI Strategy, Governance & Workforce": generate_wordcloud_for_zone(
            strategy, abstracts
        ),
    }

    len(wordclouds)
    return plt, wordclouds


@app.cell(hide_code=True)
def wordcloud_zone_selector(mo, wordclouds):
    zone_selector = mo.ui.dropdown(
        options=list(wordclouds.keys()),
        value="Agentic AI Systems",
        label="Select Zone:",
    )
    zone_selector
    return (zone_selector,)


@app.cell(hide_code=True)
def display_wordcloud(plt, wordclouds, zone_selector):
    # Convert word cloud to image for display
    wc_to_display = wordclouds[zone_selector.value]

    fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
    ax_wc.imshow(wc_to_display, interpolation="bilinear")
    ax_wc.axis("off")
    ax_wc.set_title(f"Word Cloud: {zone_selector.value}", fontsize=16, pad=20)
    plt.tight_layout()
    fig_wc
    return


@app.cell(hide_code=True)
def data_importance_note(mo):
    mo.md(r"""
    **Observation:** Notice how the word "data" appears prominently in the word
    cloud regardless of which zone you select. This underscores a fundamental
    truth: data has never gone away as an important underlying requirement.
    Even as we build agentic systems and deploy foundation models, quality data
    remains the substrate that determines success or failure.
    """)
    return


@app.cell(hide_code=True)
def controls_intro(mo):
    mo.md(r"""
    ### Interactive Controls

    Use the radio button below to toggle between the two classification views.
    The plot will update to show either the five LLM-identified zones or the
    HDBSCAN clusters.
    """)
    return


@app.cell(hide_code=True)
def view_selector():
    import marimo as mo

    view = mo.ui.radio(
        options=["LLM-based zones", "Embedding clusters (HDBSCAN)"],
        value="LLM-based zones",
    )
    view
    return mo, view


@app.cell(hide_code=True)
def extract_companies(sessions):
    companies = [s["company"] or "" for s in sessions if s.get("session_abstract")]
    len(companies)
    return (companies,)


@app.cell(hide_code=True)
def create_interactive_plot(
    abstracts,
    agent_labels,
    coords,
    labels,
    mo,
    speakers,
    titles,
    tracks,
    view,
):
    import html
    import textwrap

    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    def wrap_text(text, width=80):
        return "<br>".join(textwrap.wrap(text, width))

    df_plot = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "hdbscan": [
                f"Cluster {label}" if label != -1 else "Noise" for label in labels
            ],
            "agent_zone": agent_labels,
            "title": titles,
            "speaker": speakers,
            "track": tracks,
            "abstract": abstracts,
        }
    )

    df_plot["hover"] = df_plot.apply(
        lambda r: (
            f"<b>{html.escape(r['title'])}</b><br>"
            f"<b>Speaker:</b> {html.escape(r['speaker'])}<br>"
            f"<b>Track:</b> {html.escape(r['track'])}<br><br>"
            f"{wrap_text(html.escape(r['abstract']), 80)}"
        ),
        axis=1,
    )

    if view.value == "LLM-based zones":
        color_col = "agent_zone"
        title_text = "ODSC East 2026 - LLM-based Zone Classification (UMAP projection)"
    else:
        color_col = "hdbscan"
        title_text = (
            "ODSC East 2026 - Embedding-based HDBSCAN Clusters (UMAP projection)"
        )

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color=color_col,
        hover_name="title",
        hover_data={
            "hover": True,
            "x": False,
            "y": False,
            "title": False,
            "speaker": False,
            "track": False,
            "abstract": False,
            "hdbscan": False,
            "agent_zone": False,
        },
        title=title_text,
        width=1100,
        height=750,
    )
    fig.update_traces(
        marker=dict(size=8, opacity=0.7),
        hovertemplate="%{customdata[0]}<extra></extra>",
    )
    fig.update_layout(legend_title_text=color_col.replace("_", " ").title())
    scatter = mo.ui.plotly(fig)
    scatter
    return df_plot, go, scatter


@app.cell(hide_code=True)
def interactive_selection(
    agent_labels,
    companies,
    df_plot,
    mo,
    scatter,
    speakers,
    titles,
    view,
):
    import polars as pl

    if scatter.value:
        pts = scatter.value
        active_col = "agent_zone" if view.value == "LLM-based zones" else "hdbscan"

        # Get trace names in the order Plotly created them (matches curveNumber)
        trace_names = [trace.name for trace in scatter._figure.data]
        curve_map = {i: name for i, name in enumerate(trace_names)}

        def get_global_idx(cnum, pidx):
            gname = curve_map[cnum]
            group = df_plot[df_plot[active_col] == gname].reset_index()
            return int(group.iloc[pidx]["index"])

        idxs = [
            get_global_idx(p["curveNumber"], p["pointIndex"])
            for p in pts
            if "pointIndex" in p
        ]
        if idxs:
            result = pl.DataFrame(
                {
                    "title": [titles[i] for i in idxs],
                    "speaker": [speakers[i] for i in idxs],
                    "affiliation": [companies[i] for i in idxs],
                    "zone": [agent_labels[i] for i in idxs],
                }
            )
        else:
            result = mo.md(
                "Use the box select or lasso tool on the scatter plot above "
                "to select talks."
            )
    else:
        result = mo.md(
            "Use the box select or lasso tool on the scatter plot above "
            "to select talks."
        )
    result
    return


@app.cell(hide_code=True)
def key_insights(mo):
    mo.md(r"""
    ## What the Visualization Reveals

    **ODSC East 2026 feels like the year AI builder culture became systems culture.**

    The numbers tell the story:

    ### The Agentic Shift

    Agentic AI Systems is the largest zone at 28% of substantive sessions. This
    reflects how agent architectures have moved from demos to production
    engineering. Teams are wrestling with core systems questions: what
    architecture keeps agents reliable when they run for long periods, call many
    tools, and fail mid-flight? The recurring design problem is orchestration
    and control, from runtimes to control planes to workflow verification.

    ### Systems Over Models

    Agent systems (28%) and foundation model engineering (19%) together account
    for nearly half the conference. The practical energy sits in the joints
    between components: agent architecture, data infrastructure, evaluation,
    deployment, and decision-making structures inside organizations.

    ### Technical and Leadership Coupling

    Technical implementation zones (Agentic AI, LLM Engineering, Data
    Infrastructure) make up 60% of sessions, while organizational adoption
    zones (Strategy/Governance, Applied AI) account for 36%. This 1.6:1 ratio
    shows healthy coupling. That pairing usually appears when teams are moving
    from experimentation budgets to accountability budgets.

    ### Embedding Clusters vs. LLM Zones

    Toggle between the two views above to see an instructive pattern:

    - **Where clusters split a zone**: Sub-communities with distinct vocabulary
      exist within the same theme (e.g., "agent architecture" vs "agent ops"
      within Zone 1)
    - **Where clusters merge zones**: The abstracts share enough language that
      unsupervised methods cannot distinguish them, revealing cross-cutting
      concerns

    HDBSCAN identifies only 3 clusters from the embeddings, while the
    multi-agent categorization process identified 6 meaningful categories. This
    suggests the conference vocabulary has significant overlap, but the semantic
    differences matter for understanding where the field is heading.

    ### Likely Macro Trends

    - Agentic AI becomes an engineering discipline with explicit quality loops
    - RAG and context strategy remain central, but with more scrutiny on evaluation
    - MLOps and LLMOps converge into one reliability stack for mixed AI systems
    - Governance work sits closer to platform design and product strategy
    - The value of attending shifts from "learn a model" to "learn a repeatable system"
    """)
    return


if __name__ == "__main__":
    app.run()
