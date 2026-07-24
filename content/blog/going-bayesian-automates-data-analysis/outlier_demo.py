import marimo  # ruff: noqa: E402

# ruff: noqa: E501, E741

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Normal vs Student-t likelihood: the outlier effect

        This notebook fits the same 4PL dose-response data with two PyMC models:

        1. **Normal likelihood** (thin tails) — one outlier drags the curve
        2. **Student-t likelihood** (heavy tails, adjustable nu) — the outlier gets down-weighted

        Drag the **nu** slider and watch how the Student-t curve ignores the outlier at low nu, then starts chasing it as nu approaches Normal-like behavior at high values.
        """
    )
    return


@app.cell
def _():
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pymc as pm

    return az, np, pm, plt


@app.cell
def _(np):
    def f4pl(x, bottom, top, log_ec50, hill):
        """4-parameter logistic curve."""
        return bottom + (top - bottom) / (1 + 10 ** ((log_ec50 - x) * hill))

    # Clean data: 8 dilutions along a 4PL curve
    x_clean = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    true_params = dict(bottom=5, top=95, log_ec50=4.5, hill=1.5)
    rng = np.random.default_rng(42)
    y_clean = f4pl(x_clean, **true_params) + rng.normal(0, 1.5, len(x_clean))

    # Outlier point
    x_outlier = np.array([4.0])
    y_outlier = np.array([80.0])

    x_all = np.concatenate([x_clean, x_outlier])
    y_all = np.concatenate([y_clean, y_outlier])
    return f4pl, x_all, x_clean, x_outlier, y_all, y_clean, y_outlier


@app.cell
def _(mo):
    nu_slider = mo.ui.slider(
        start=1,
        stop=30,
        step=0.5,
        value=3,
        label="Student-t degrees of freedom (nu)",
        show_value=True,
    )
    nu_slider
    return (nu_slider,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Data

        Blue points follow a clean 4PL dose-response curve. The red point is the outlier.
        """
    )
    return


@app.cell
def _(plt, x_all, x_clean, x_outlier, y_all, y_clean, y_outlier):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(x_clean, y_clean, c="#2196F3", s=60, zorder=5, label="Clean data")
    ax.scatter(x_outlier, y_outlier, c="#e85d3a", s=80, zorder=5, label="Outlier")
    ax.set_xlabel("log10(dilution)")
    ax.set_ylabel("% viability")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper left")
    ax.set_title("Dose-response data with outlier")
    plt.tight_layout()
    plt
    return


@app.cell
def _(mo):
    mo.md("""## Normal likelihood model (thin tails)""")
    return


@app.cell
def _(pm, x_all, y_all):
    with pm.Model() as normal_model:
        bottom_p = pm.Normal("bottom", mu=0, sigma=20)
        top_p = pm.Normal("top", mu=100, sigma=20)
        log_ec50_p = pm.Normal("log_ec50", mu=4.5, sigma=3)
        hill_p = pm.HalfNormal("hill", sigma=3)

        mu = bottom_p + (top_p - bottom_p) / (1 + 10 ** ((log_ec50_p - x_all) * hill_p))

        sigma_n = pm.HalfNormal("sigma", sigma=10)
        pm.Normal("y_obs", mu=mu, sigma=sigma_n, observed=y_all)

        normal_trace = pm.sample(1000, tune=1000, chains=2, progressbar=False)
    return normal_model, normal_trace


@app.cell
def _(mo):
    mo.md("""## Student-t likelihood model (heavy tails, adjustable nu)""")
    return


@app.cell
def _(nu_slider, pm, x_all, y_all):
    _nu_val = float(nu_slider.value)

    with pm.Model() as student_t_model:
        bottom_s = pm.Normal("bottom", mu=0, sigma=20)
        top_s = pm.Normal("top", mu=100, sigma=20)
        log_ec50_s = pm.Normal("log_ec50", mu=4.5, sigma=3)
        hill_s = pm.HalfNormal("hill", sigma=3)

        mu_s = bottom_s + (top_s - bottom_s) / (
            1 + 10 ** ((log_ec50_s - x_all) * hill_s)
        )

        sigma_s = pm.HalfNormal("sigma", sigma=10)
        pm.StudentT("y_obs", nu=_nu_val, mu=mu_s, sigma=sigma_s, observed=y_all)

        student_t_trace = pm.sample(1000, tune=1000, chains=2, progressbar=False)
    return student_t_model, student_t_trace


@app.cell
def _(mo):
    mo.md("""## Comparison: both fitted curves with uncertainty""")
    return


@app.cell
def _(
    az,
    f4pl,
    normal_trace,
    np,
    nu_slider,
    plt,
    student_t_trace,
    x_all,
    x_clean,
    x_outlier,
    y_all,
    y_clean,
    y_outlier,
):
    # Fine x grid for smooth curves
    x_grid = np.linspace(0.5, 8.5, 200)

    def curve_band(trace, x_grid):
        """Compute median curve and 94% HDI band from posterior."""
        posts = trace.posterior
        bottom = posts["bottom"].values.flatten()
        top = posts["top"].values.flatten()
        lec50 = posts["log_ec50"].values.flatten()
        hill = posts["hill"].values.flatten()

        curves = np.array(
            [f4pl(x_grid, b, t, l, h) for b, t, l, h in zip(bottom, top, lec50, hill)]
        )

        median = np.median(curves, axis=0)
        hdi_low = np.percentile(curves, 3, axis=0)
        hdi_high = np.percentile(curves, 97, axis=0)
        return median, hdi_low, hdi_high

    n_med, n_lo, n_hi = curve_band(normal_trace, x_grid)
    s_med, s_lo, s_hi = curve_band(student_t_trace, x_grid)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Normal: blue band + line
    ax.fill_between(x_grid, n_lo, n_hi, alpha=0.15, color="#2196F3")
    ax.plot(x_grid, n_med, color="#2196F3", linewidth=2, label="Normal likelihood")

    # Student-t: orange band + dashed line
    ax.fill_between(x_grid, s_lo, s_hi, alpha=0.15, color="#e85d3a")
    ax.plot(
        x_grid,
        s_med,
        color="#e85d3a",
        linewidth=2,
        linestyle="--",
        label="Student-t likelihood",
    )

    # Data
    ax.scatter(
        x_clean,
        y_clean,
        c="#2196F3",
        s=60,
        zorder=5,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.scatter(
        x_outlier,
        y_outlier,
        c="#e85d3a",
        s=80,
        zorder=5,
        edgecolors="white",
        linewidths=0.5,
        label="Outlier",
    )

    ax.set_xlabel("log10(dilution)")
    ax.set_ylabel("% viability")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper left")
    ax.set_title(f"Normal vs Student-t (nu={float(nu_slider.value)}) fitted curves")
    plt.tight_layout()
    plt
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Notice what happens as you change **nu**:

        - **Low nu (1-3)**: The Student-t curve barely moves toward the outlier. The heavy tails say "that point is rare but possible," so the fit ignores it.
        - **High nu (15-30)**: The Student-t curve starts behaving like the Normal curve, getting pulled toward the outlier. The tails have thinned out.

        This is the entire argument: with the right likelihood, the model handles outliers automatically. No manual exclusion needed.
        """
    )
    return


if __name__ == "__main__":
    app.run()
