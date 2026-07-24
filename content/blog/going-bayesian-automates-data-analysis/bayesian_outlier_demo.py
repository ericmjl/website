import marimo

# ruff: noqa: E501, E741, F821, F841

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=2.0",
#     "scipy>=1.14",
#     "matplotlib>=3.9",
#     "emcee>=3.1",
#     "pymc>=5.0; sys_platform != 'emscripten'",
#     "arviz>=0.20; sys_platform != 'emscripten'",
# ]
# ///

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    mo.md(
        r"""
        # Bayesian outlier handling: Normal vs Student-t likelihood

        Two 4PL dose-response models are fit to the same data with MCMC:

        1. **Normal likelihood** (thin tails): one outlier drags the curve
        2. **Student-t likelihood** (heavy tails, adjustable nu): the outlier is down-weighted

        Drag the **nu** slider. At low nu the Student-t band ignores the
        outlier. At high nu it behaves like the Normal band.

        Shaded regions are 94% posterior credible intervals.
        Locally this uses PyMC/NUTS; in the browser (WASM) it uses emcee.
        """
    )
    return


@app.cell
def _():
    import sys

    import numpy as np
    from scipy.special import gammaln

    IS_WASM = sys.platform == "emscripten"

    try:
        import arviz as az
        import pymc as pm

        HAS_PYMC = True
    except ImportError:
        HAS_PYMC = False

    import matplotlib.pyplot as plt

    return HAS_PYMC, IS_WASM, az, gammaln, np, plt, pm


@app.cell
def _(IS_WASM, mo):
    if IS_WASM:
        import micropip

        await micropip.install("emcee")
    import emcee

    return (emcee,)


@app.cell
def _(np):
    def f4pl(x, bottom, top, log_ec50, hill):
        """4-parameter logistic dose-response curve."""
        return bottom + (top - bottom) / (1 + 10 ** ((log_ec50 - np.asarray(x)) * hill))

    return (f4pl,)


@app.cell
def _(gammaln, np):
    def log_prior(params):
        """Weakly informative priors for 4PL + log(sigma)."""
        bottom, top, log_ec50, hill, log_sigma = params
        if hill < 0 or top < bottom:
            return -np.inf
        sigma = np.exp(log_sigma)
        if sigma < 0.01 or sigma > 200:
            return -np.inf
        lp = -0.5 * (bottom / 20) ** 2
        lp += -0.5 * ((top - 100) / 20) ** 2
        lp += -0.5 * ((log_ec50 - 4.5) / 3) ** 2
        lp += -0.5 * (hill / 3) ** 2
        lp += -0.5 * (sigma / 10) ** 2 - log_sigma
        return lp

    def log_lik_normal(params, x, y):
        b, t, e, h, ls = params
        s = np.exp(ls)
        mu = b + (t - b) / (1 + 10 ** ((e - x) * h))
        return -0.5 * np.sum(((y - mu) / s) ** 2 + np.log(2 * np.pi * s**2))

    def log_lik_studentt(params, x, y, nu):
        b, t, e, h, ls = params
        s = np.exp(ls)
        mu = b + (t - b) / (1 + 10 ** ((e - x) * h))
        z = (y - mu) / s
        per_obs = (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi)
            - np.log(s)
            - (nu + 1) / 2 * np.log1p(z**2 / nu)
        )
        return np.sum(per_obs)

    return log_lik_normal, log_lik_studentt, log_prior


@app.cell
def _(emcee, np):
    def sample_emcee(log_post_fn, init, n_walkers=32, n_steps=2000, burn=500, seed=42):
        """Affine-invariant ensemble sampler (emcee)."""
        ndim = len(init)
        rng = np.random.default_rng(seed)
        p0 = np.array(init) + rng.normal(
            0, [0.5, 0.5, 0.05, 0.05, 0.05], (n_walkers, ndim)
        )
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_post_fn)
        sampler.run_mcmc(p0, n_steps, progress=False)
        return sampler.get_chain(discard=burn, flat=True)

    return (sample_emcee,)


@app.cell
def _(f4pl, np):
    x_clean = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    rng = np.random.default_rng(42)
    y_clean = f4pl(x_clean, 5, 95, 4.5, 1.5) + rng.normal(0, 1.5, len(x_clean))
    x_outlier = np.array([4.0])
    y_outlier = np.array([80.0])
    x_all = np.concatenate([x_clean, x_outlier])
    y_all = np.concatenate([y_clean, y_outlier])
    return x_all, x_clean, x_outlier, y_all, y_clean, y_outlier


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
    mo.md("## Fit both models")
    return


@app.cell
def _(
    HAS_PYMC,
    IS_WASM,
    log_lik_normal,
    log_lik_studentt,
    log_prior,
    np,
    nu_slider,
    pm,
    sample_emcee,
    x_all,
    y_all,
):
    _nu_val = float(nu_slider.value)
    _init = np.array([5, 95, 4.5, 1.5, np.log(5)])

    # --- Normal likelihood ---
    if HAS_PYMC and not IS_WASM:
        with pm.Model() as normal_model:
            b = pm.Normal("bottom", 0, 20)
            t = pm.Normal("top", 100, 20)
            ec = pm.Normal("log_ec50", 4.5, 3)
            h = pm.HalfNormal("hill", 3)
            mu_n = b + (t - b) / (1 + 10 ** ((ec - x_all) * h))
            s = pm.HalfNormal("sigma", 10)
            pm.Normal("y", mu=mu_n, sigma=s, observed=y_all)
            normal_trace = pm.sample(1000, tune=1000, chains=2, progressbar=False)
        normal_samples = np.column_stack(
            [
                normal_trace.posterior[k].values.flatten()
                for k in ["bottom", "top", "log_ec50", "hill", "sigma"]
            ]
        )
        normal_samples[:, 4] = np.log(normal_samples[:, 4])
    else:
        normal_samples = sample_emcee(
            lambda p: log_prior(p) + log_lik_normal(p, x_all, y_all), _init
        )

    # --- Student-t likelihood ---
    if HAS_PYMC and not IS_WASM:
        with pm.Model() as student_t_model:
            b2 = pm.Normal("bottom", 0, 20)
            t2 = pm.Normal("top", 100, 20)
            ec2 = pm.Normal("log_ec50", 4.5, 3)
            h2 = pm.HalfNormal("hill", 3)
            mu_s = b2 + (t2 - b2) / (1 + 10 ** ((ec2 - x_all) * h2))
            s2 = pm.HalfNormal("sigma", 10)
            pm.StudentT("y", nu=_nu_val, mu=mu_s, sigma=s2, observed=y_all)
            student_t_trace = pm.sample(1000, tune=1000, chains=2, progressbar=False)
        student_t_samples = np.column_stack(
            [
                student_t_trace.posterior[k].values.flatten()
                for k in ["bottom", "top", "log_ec50", "hill", "sigma"]
            ]
        )
        student_t_samples[:, 4] = np.log(student_t_samples[:, 4])
    else:
        student_t_samples = sample_emcee(
            lambda p: log_prior(p) + log_lik_studentt(p, x_all, y_all, _nu_val),
            _init,
        )
    return normal_samples, student_t_samples


@app.cell
def _(mo):
    mo.md("## Comparison with posterior credible bands")
    return


@app.cell
def _(
    f4pl,
    normal_samples,
    np,
    nu_slider,
    plt,
    student_t_samples,
    x_clean,
    x_outlier,
    y_clean,
    y_outlier,
):
    x_grid = np.linspace(0.5, 8.5, 200)

    def curve_band(samples, x_grid):
        curves = np.array([f4pl(x_grid, s[0], s[1], s[2], s[3]) for s in samples[::5]])
        return (
            np.median(curves, axis=0),
            np.percentile(curves, 3, axis=0),
            np.percentile(curves, 97, axis=0),
        )

    n_med, n_lo, n_hi = curve_band(normal_samples, x_grid)
    s_med, s_lo, s_hi = curve_band(student_t_samples, x_grid)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(x_grid, n_lo, n_hi, alpha=0.15, color="#2196F3")
    ax.plot(x_grid, n_med, color="#2196F3", lw=2, label="Normal likelihood")

    ax.fill_between(x_grid, s_lo, s_hi, alpha=0.15, color="#e85d3a")
    ax.plot(
        x_grid,
        s_med,
        color="#e85d3a",
        lw=2,
        ls="--",
        label=f"Student-t (nu={float(nu_slider.value)})",
    )

    ax.scatter(
        x_clean, y_clean, c="#2196F3", s=60, zorder=5, edgecolors="white", lw=0.5
    )
    ax.scatter(
        x_outlier,
        y_outlier,
        c="#e85d3a",
        s=80,
        zorder=5,
        edgecolors="white",
        lw=0.5,
        label="Outlier",
    )

    ax.set_xlabel("log10(dilution)")
    ax.set_ylabel("% neutralization")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Bayesian 4PL: Normal vs Student-t posterior")
    plt.tight_layout()
    plt
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Low nu (1-3):** The Student-t band barely shifts toward the
        outlier. Heavy tails say "rare but possible," so the fit ignores it.

        **High nu (15-30):** The Student-t band converges toward the Normal
        band. The tails have thinned, and the outlier pulls the curve.

        This is the whole argument: with the right likelihood, the model
        handles outliers automatically. No manual exclusion needed.
        """
    )
    return


if __name__ == "__main__":
    app.run()
