# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic==0.49.0",
#     "arviz==0.21.0",
#     "joypy==0.2.6",
#     "marimo",
#     "matplotlib==3.10.1",
#     "numba==0.61.0",
#     "numpy==2.1.0",
#     "nutpie==0.14.3",
#     "pandas==2.2.3",
#     "pymc==5.21.1",
#     "seaborn==0.13.2",
#     "tqdm==4.67.1",
# ]
# ///

import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Bayesian Superiority Estimation with R2D2 Priors: A Practical Guide for Protein Screening

        When screening hundreds of molecules, proteins, or interventions,
        two critical questions often arise:

        1. Is our experimental setup actually measuring the effect we care about
           (vs. experimental noise)?
        2. Which candidates are truly superior to others?

        In this tutorial, we'll tackle both challenges using a practical example:
        a protein screening experiment with fluorescence readouts.
        We'll show how **R2D2 priors** help interpret variance decomposition,
        and how **Bayesian superiority calculation**
        enables robust ranking of candidates.
        Both techniques generalize to drug discovery, material science,
        or any domain requiring rigorous comparison of multiple alternatives.
        """  # noqa: E501
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The Protein Screening Example

        Let's consider a dataset with fluorescence measurements for over 100 proteins
        across multiple experiments and replicates.

        Our experimental design includes:

        - A control protein present in all experiments and replicates
        - "Crossover" proteins measured across all experiments
        - Unique test proteins in each experiment

        This design is common in high-throughput screening scenarios
        where measuring all proteins in all conditions is impractical.
        For simplicity, I am leaving out factors such as plate and well position,
        but know that in a real life situation,
        these factors would be considered as part of the experimental design.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Importing PyMC for Bayesian Modeling

        We'll use PyMC to implement our Bayesian hierarchical model with R2D2 priors.
        PyMC is a powerful probabilistic programming framework
        that allows us to define and sample from complex statistical models.
        """
    )
    return


@app.cell
def _():
    import pymc as pm

    return (pm,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Protein Fluorescence Dataset Description

        This dataset contains fluorescence measurements
        from a series of protein experiments.
        The experimental design is as follows:

        ### Structure

        - Total proteins measured: >100 unique proteins
        - Number of experiments: 3
        - Replicates per experiment: 2
        - Total measurements: ~300-400

        ### Key Components

        1. Control protein: Present in all experiments and replicates
        2. Crossover proteins: 3-5 proteins measured across all experiments
        3. Test proteins: Unique to each experiment

        ### Experimental Variation

        - Inter-experiment variation: Higher systematic shifts
        - Intra-experiment variation (between replicates): Lower systematic shifts

        ### Measurement Details

        - Readout: Fluorescent units
        - Each protein appears in either:
          * Multiple measurements (control and crossover proteins)
          * Single measurement (test proteins)

        ### Purpose

        This design allows for:

        - Quality control through control protein measurements
        - Cross-experiment normalization using crossover proteins
        - High-throughput screening of many unique proteins
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Generating Synthetic Protein Fluorescence Data

        To demonstrate our approach,
        we'll generate synthetic data
        that mimics a realistic protein screening experiment.
        Accordingly, we will:

        1. Define the experimental structure (experiments, replicates, proteins)
        2. Create protein identifiers for controls, crossovers, and test proteins
        3. Simulate "true" underlying protein activities
        4. Add systematic experiment and replicate effects
        5. Incorporate measurement noise

        The code below creates a dataset
        that captures key features of real protein screening experiments,
        including batch effects between experiments and replicates.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    # Set random seed for reproducibility
    np.random.seed(42)

    # Define parameters
    n_experiments = 3
    n_replicates = 2
    n_proteins_per_exp = 40
    n_crossover = 4

    # Create protein names
    control = ["Control"]
    crossover_proteins = [f"Crossover_{i}" for i in range(n_crossover)]
    other_proteins = [f"Protein_{i}" for i in range(100)]

    # Base fluorescence values
    base_values = {}
    base_values["Control"] = 1000
    for p in crossover_proteins:
        base_values[p] = np.random.normal(1000, 200)
    for p in other_proteins:
        base_values[p] = np.random.normal(1000, 200)

    # Create experiment effects
    exp_effects = np.random.normal(1, 0.3, n_experiments)
    rep_effects = np.random.normal(1, 0.1, (n_experiments, n_replicates))

    # Generate data
    data = []
    for exp in range(n_experiments):
        # Select proteins for this experiment
        exp_proteins = (
            control + crossover_proteins + other_proteins[exp * 30 : (exp + 1) * 30]
        )

        for rep in range(n_replicates):
            for protein in exp_proteins:
                # Add noise and effects
                value = (
                    base_values[protein]
                    * exp_effects[exp]
                    * rep_effects[exp, rep]
                    * np.random.normal(1, 0.05)
                )

                data.append(
                    {
                        "Experiment": f"Exp_{exp+1}",
                        "Replicate": f"Rep_{rep+1}",
                        "Protein": protein,
                        "Fluorescence": value,
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df
    return (
        base_values,
        control,
        crossover_proteins,
        data,
        df,
        exp,
        exp_effects,
        exp_proteins,
        n_crossover,
        n_experiments,
        n_proteins_per_exp,
        n_replicates,
        np,
        other_proteins,
        p,
        pd,
        protein,
        rep,
        rep_effects,
        value,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        In the synthetic dataset we've created, we simulated:

        - 3 experiments with 2 replicates each
        - A control protein and 4 crossover proteins present in all experiments
        - 100 other proteins distributed across experiments
        - Multiplicative experiment effects (mean=1, sd=0.3)
        - Multiplicative replicate effects (mean=1, sd=0.1)
        - Multiplicative measurement noise (mean=1, sd=0.05)

        This structure simulates a typical screening setup
        where batch effects between experiments are stronger than replicate variation,
        and both contribute significantly to the observed fluorescence values.
        The setting mirrors real experimental challenges
        where we need to separate biological signal from technical noise.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Examining the Raw Data

        Before modeling,
        let's visualize the raw data for control and crossover proteins
        to understand the experimental variation:
        """
    )
    return


@app.cell
def _(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Filter for Control and Crossover samples
    mask = df["Protein"].str.contains("Control|Crossover")
    filtered_df = df[mask]

    # Create the swarm plot
    sns.swarmplot(
        data=filtered_df, x="Experiment", y="Fluorescence", hue="Protein", size=8
    )

    plt.title("Activity for Controls and Crossover Samples")
    plt.ylabel("Fluorescence")
    plt.gca()
    return filtered_df, mask, plt, sns


@app.cell
def _(mo):
    mo.md(
        r"""
        Notice the dramatic shift in fluorescence values across experiments.
        Experiment 3 shows substantially higher fluorescence readings
        (around 1500-2000 units)
        compared to Experiments 1 and 2 (mostly below 1300 units).
        This systematic shift affects all proteins,
        including our control and crossover samples.

        This pattern illustrates a common challenge in high-throughput screening:
        significant batch effects between experiments
        that can mask the true biological signal we're interested in.
        Without accounting for these experimental factors,
        we might incorrectly attribute higher activity to proteins
        simply because they were measured in Experiment 3.

        Additionally, there's variation between replicates within each experiment,
        though this effect is smaller than the between-experiment variation.
        Our modeling approach needs to account for
        both sources of experimental noise
        to accurately compare proteins across the entire dataset.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The R2D2 Prior: Interpretable Variance Decomposition

        To address these experimental batch effects
        and properly separate biological signal from noise,
        we need a modeling approach
        that explicitly accounts for different sources of variation.
        This is where the R2D2 prior becomes valuable.

        The R2D2 prior (R-squared Dirichlet decomposition)
        provides an interpretable framework for variance decomposition
        by placing a prior on the Bayesian coefficient of determination ($R^2$),
        which then induces priors on individual parameters.
        Introduced by [Yanchenko, Bondell, and Reich (2021)](https://arxiv.org/abs/2111.10718),
        the R2D2 prior is especially valuable
        for generalized linear mixed models
        where we want to understand how much of the total variance
        is explained by different model components.

        The core idea of R2D2 is to:

        1. Place a $\text{Beta}(a,b)$ prior on the coefficient of determination ($R^2$). An uninformative prior, such as a $\text{Beta}(1,1)$ is appropriate.
        2. This induces a prior on the global variance parameter (`global_var` in our model, which represents the total variance of the linear predictor)
        3. The global variance is then decomposed into component-specific variances via a Dirichlet-distributed vector, each mapping to a particular variance component.

        This approach offers a significant advantage over traditional hierarchical models where we would specify separate, unrelated priors for each variance component (e.g., $\sigma^2_{experiment}$, $\sigma^2_{replicate}$, $\sigma^2_{protein}$). Those traditional approaches often rely on arbitrary "magic numbers" for each variance parameter, making it difficult to interpret how much each component contributes to the overall model fit. With R2D2, we instead control the total explained variance through a single interpretable parameter ($R^2$) and then partition that variance meaningfully through the Dirichlet distribution, creating a coherent framework for understanding variance decomposition. And to top it off, the hierarchical nature of variance specification here ensures regularization of variance estimates away from unreasonable values.

        Here's how we implement the model:
        """  # noqa: E501
    )
    return


@app.cell
def _(df, np, pd, pm):
    def _():
        # Create categorical indices
        exp_idx = pd.Categorical(df["Experiment"]).codes
        rep_idx = pd.Categorical(df["Replicate"]).codes
        prot_idx = pd.Categorical(df["Protein"]).codes

        # Define coordinates for dimensions
        coords = {
            "experiment": df["Experiment"].unique(),
            "replicate": df["Replicate"].unique(),
            "protein": df["Protein"].unique(),
        }

        with pm.Model(coords=coords) as model:
            # Explicitly define R² prior (core of R2D2)
            r_squared = pm.Beta("r_squared", alpha=1, beta=1)

            # Global parameters
            global_mean = pm.Normal(
                "global_mean", mu=7, sigma=1
            )  # log scale for fluorescence

            # Residual variance (unexplained)
            sigma_squared = pm.HalfNormal("sigma_squared", sigma=1)

            # Global variance derived from R² and residual variance
            # For normal models: W = sigma² * r²/(1-r²)
            global_var = pm.Deterministic(
                "global_var", sigma_squared * r_squared / (1 - r_squared)
            )
            global_sd = pm.Deterministic("global_sd", pm.math.sqrt(global_var))  # noqa: F841

            # R2D2 decomposition parameters
            # 4 components: experiment, replicate (nested in experiment), protein,
            # and unexplained
            props = pm.Dirichlet("props", a=np.ones(4))

            # Component variances (for interpretability)
            exp_var = pm.Deterministic("exp_var", props[0] * global_var)
            rep_var = pm.Deterministic("rep_var", props[1] * global_var)
            prot_var = pm.Deterministic("prot_var", props[2] * global_var)
            unexplained_var = pm.Deterministic("unexplained_var", props[3] * global_var)

            # Component standard deviations
            exp_sd = pm.Deterministic("exp_sd", pm.math.sqrt(exp_var))
            rep_sd = pm.Deterministic("rep_sd", pm.math.sqrt(rep_var))
            prot_sd = pm.Deterministic("prot_sd", pm.math.sqrt(prot_var))
            unexplained_sd = pm.Deterministic(
                "unexplained_sd", pm.math.sqrt(unexplained_var)
            )

            # Component effects
            exp_effect = pm.Normal("exp_effect", mu=0, sigma=exp_sd, dims="experiment")
            rep_effect = pm.Normal(
                "rep_effect", mu=0, sigma=rep_sd, dims=("experiment", "replicate")
            )
            prot_effect = pm.Normal("prot_effect", mu=0, sigma=prot_sd, dims="protein")

            # Protein activity (what we're ultimately interested in)
            prot_activity = pm.Deterministic(  # noqa: F841
                "prot_activity", global_mean + prot_effect, dims="protein"
            )

            # Expected value
            y_hat = (
                global_mean
                + exp_effect[exp_idx]
                + rep_effect[exp_idx, rep_idx]
                + prot_effect[prot_idx]
            )

            # Calculate model R² directly (for verification)
            model_r2 = pm.Deterministic(  # noqa: F841
                "model_r2",
                (exp_var + rep_var + prot_var)
                / (exp_var + rep_var + prot_var + unexplained_var),
            )

            # Likelihood
            y = pm.Normal(  # noqa: F841
                "y", mu=y_hat, sigma=unexplained_sd, observed=np.log(df["Fluorescence"])
            )

            # Sample
            trace = pm.sample(
                2000, tune=1000, return_inferencedata=True, nuts_sampler="nutpie"
            )
        return model, trace

    model, trace = _()
    return model, trace


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Model Convergence and Posterior Analysis

        Now that we've fitted our Bayesian model with R2D2 priors, we need to assess:

        1. Whether the model converged properly
        2. How the variance is partitioned across different components
        3. What protein activities look like after accounting for experimental effects

        First, let's check for divergent transitions, which can indicate problems with the model fitting process.
        """  # noqa: E501
    )
    return


@app.cell
def _(trace):
    trace.sample_stats.diverging.sum()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Zero divergent transitions indicates good model convergence. This means our posterior samples should provide reliable estimates of all model parameters.

        Next, let's examine the posterior distributions of the variance components. The `props` parameter from our model represents how the total variance is partitioned across experiment, replicate, protein, and unexplained components.
        """  # noqa: E501
    )
    return


@app.cell
def _(trace):
    import arviz as az

    axes_posterior_props = az.plot_posterior(trace, var_names=["props"])
    axes_posterior_props[0].set_title("experiment")
    axes_posterior_props[1].set_title("replicate")
    axes_posterior_props[2].set_title("protein")
    axes_posterior_props[3].set_title("unexplained")
    return axes_posterior_props, az


@app.cell
def _(mo):
    mo.md(
        r"""
        These posterior plots show the distributions of the variance components. Each represents the proportion of total variance attributed to that component. We can see clear differences in the contributions from each source.

        We can also look at the total $R^2$ value, which represents the proportion of variance explained by the model:
        """  # noqa: E501
    )
    return


@app.cell
def _(az, trace):
    az.plot_posterior(trace, var_names=["model_r2"])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Let's examine the posterior distributions of each variance component. In the ridgeline plot below, we see that unexplained variance currently constitutes a tiny fraction of total variation, while experiment and replicate -- two components that really should not contribute much to the output variation, are actually significant contributors to the readout variation. This can serve as a metric on laboratory consistency. Ideally, the protein that we're engineering should contribute the most variation to the output that we see.

        The R2D2 prior answers our first critical question in a way that guides practical action: we should focus on improving experimental execution to make our experiments more consistent. Ideally, the protein effect should be the majority contributor to the readout variation. This analysis suggests that refining our experimental protocol to reduce batch effects between experiments and variability between replicates would substantially improve the signal-to-noise ratio of our screening platform.
        """  # noqa: E501
    )
    return


@app.cell
def _(pd, plt, trace):
    import joypy

    # Extract the posterior samples for props
    props_samples = trace.posterior["props"].values.reshape(-1, 4)
    props_df = pd.DataFrame(
        props_samples, columns=["Experiment", "Replicate", "Protein", "Unexplained"]
    )

    # Create the ridgeline plot
    fig, axes = joypy.joyplot(
        props_df,
        figsize=(10, 4),
        colormap=plt.cm.tab10,
        alpha=0.7,
        title="Posterior Distributions of Variance Components",
    )

    # Add vertical lines for means
    for i, col in enumerate(props_df.columns):
        axes[i].axvline(props_df[col].mean(), color="black", linestyle="--", alpha=0.8)
        axes[i].text(
            props_df[col].mean() + 0.02,
            0.5,
            f"Mean: {props_df[col].mean():.2f}",
            transform=axes[i].get_yaxis_transform(),
        )

    plt.tight_layout()
    plt.gca()
    return axes, col, fig, i, joypy, props_df, props_samples


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Examining Protein Activity Estimates

        Now that we've decomposed the variance and accounted for experimental effects, we can examine the actual protein activity estimates. These represent the "true" biological signal after removing the experimental noise:
        """  # noqa: E501
    )
    return


@app.cell
def _(az, trace):
    ax = az.plot_forest(trace.posterior["prot_activity"])[0]
    ax.set_xlabel("log(protein activity)")
    return (ax,)


@app.cell
def _(mo):
    mo.md(
        r"""
        This forest plot displays posterior distributions of protein activity on the log scale, with horizontal lines representing 94% credible intervals. We can identify several proteins (such as Protein_12, Protein_38, and Protein_66) that appear to have higher activity than others.

        However, a critical challenge emerges: despite most proteins having similar uncertainty in their estimates (shown by consistent credible interval widths), this uncertainty creates significant ambiguity when comparing proteins with similar point estimates. Simply ranking proteins by their posterior mean activity could lead us to prioritize proteins with slightly higher point estimates when overlapping uncertainty makes it difficult to confidently determine their true superiority.

        This illustrates a fundamental limitation of ranking by point estimates alone: it fails to properly account for uncertainty across different proteins. A protein with a slightly lower mean but narrower credible intervals might actually be a better candidate than one with a higher mean but wider uncertainty bounds.

        While the forest plot offers a comprehensive overview of all protein activities, making definitive comparisons remains difficult due to overlapping credible intervals. This is why we need the Bayesian superiority calculation in the next section - to properly quantify the probability that one protein truly outperforms another while fully accounting for uncertainty in our estimates.
        """  # noqa: E501
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Why not just use effect sizes?

        A natural question arises: if we want to compare proteins, why not simply calculate effect sizes based on posterior activity estimates?

        Effect sizes (like Cohen's d or standardized mean differences) are often used in traditional statistical analyses to quantify the magnitude of differences between groups. In our Bayesian context, we could calculate effect sizes between each pair of proteins using our posterior activity estimates.

        Let's illustrate this approach by calculating standardized effect sizes between a reference protein (the control) and all other proteins:
        """  # noqa: E501
    )
    return


@app.cell
def _(np, pd, plt, sns, trace):
    def plot_effect_sizes(proteins_to_plot: list[str] = None, num_to_plot: int = None):
        # Get posterior samples of protein activities
        prot_activity = trace.posterior["prot_activity"].values

        # Reshape to (samples, proteins)
        n_samples = prot_activity.shape[0] * prot_activity.shape[1]
        n_proteins = prot_activity.shape[2]
        prot_activity_flat = prot_activity.reshape(-1, n_proteins)

        # Get protein names
        protein_names = trace.posterior["protein"].values

        # Find index of control protein
        control_idx = np.where(protein_names == "Control")[0][0]

        # Calculate effect sizes for each posterior sample
        # Effect size = (protein_activity - control_activity) / pooled_std
        effect_sizes = np.zeros((n_samples, n_proteins))

        for i in range(n_samples):
            control_value = prot_activity_flat[i, control_idx]
            for j in range(n_proteins):
                if j != control_idx:
                    # Calculate effect size for this sample
                    effect = prot_activity_flat[i, j] - control_value
                    # Using a simplified effect size calculation (mean difference)
                    # In practice, you might use a pooled standard deviation
                    effect_sizes[i, j] = effect

        # Create a DataFrame with effect size statistics
        effect_size_stats = pd.DataFrame(
            {
                "Protein": protein_names,
                "Mean_Effect": np.mean(effect_sizes, axis=0),
                "Lower_CI": np.percentile(effect_sizes, 2.5, axis=0),
                "Upper_CI": np.percentile(effect_sizes, 97.5, axis=0),
            }
        )

        # Filter proteins if specified
        if proteins_to_plot is not None:
            effect_size_stats = effect_size_stats[
                effect_size_stats["Protein"].isin(proteins_to_plot)
            ]

        # Sort by mean effect
        top_proteins = effect_size_stats.sort_values("Mean_Effect", ascending=False)

        if num_to_plot:
            top_proteins = top_proteins.head(num_to_plot)

        plt.figure(figsize=(10, 0.5 * len(top_proteins)))

        # Create scatter plot with error bars
        plt.errorbar(
            x=top_proteins["Mean_Effect"],
            y=range(len(top_proteins)),
            xerr=np.vstack(
                [
                    top_proteins["Mean_Effect"] - top_proteins["Lower_CI"],
                    top_proteins["Upper_CI"] - top_proteins["Mean_Effect"],
                ]
            ),
            fmt="o",
            capsize=5,
        )

        plt.yticks(range(len(top_proteins)), top_proteins["Protein"])
        plt.axvline(x=0, color="gray", linestyle="--")
        plt.xlabel("Effect Size (difference from Control)")
        plt.title("Posterior Effect Sizes of Top Proteins vs Control")
        sns.despine()
        return plt.gca(), effect_sizes

    axes_effect_sizes, effect_sizes = plot_effect_sizes(num_to_plot=20)
    axes_effect_sizes
    return axes_effect_sizes, effect_sizes, plot_effect_sizes


@app.cell
def _(mo):
    mo.md(
        r"""
        This plot shows the posterior distribution of effect sizes for the top proteins compared to the control. The horizontal lines represent 95% credible intervals.

        What's particularly interesting is Crossover 3, Protein 51, and Protein 66. Let's focus on them.

        - Between Crossover 3 and Protein 51, is Protein 51 genuinely worse than Crossover 3? It's hard to tell.
        - Between Crossover 3 and Protein 66, is Protein 66 genuinely better than Crossover 3? It's also hard to tell.
        """  # noqa: E501
    )
    return


@app.cell
def _(plot_effect_sizes):
    axes_effect_sizes_filtered, _ = plot_effect_sizes(
        proteins_to_plot=["Crossover_3", "Protein_51", "Protein_66"]
    )
    axes_effect_sizes_filtered
    return (axes_effect_sizes_filtered,)


@app.cell
def _(mo):
    mo.md(
        r"""
        While effect sizes are useful for quantifying the magnitude of differences, there are several important limitations to this approach:

        1. **Still a posterior distribution**: Notice that effect sizes themselves have a posterior distribution, not just a single value. The effect size calculation doesn't resolve our uncertainty - it merely transforms it.

        2. **Arbitrary reference**: Any effect size calculation requires choosing a reference protein (here, we used the control). The results would differ if we chose a different reference.

        3. **Scale dependence**: The interpretation of what constitutes a "large" effect size varies by context and can be subjective.

        4. **Doesn't directly answer "which is better"**: Effect sizes tell us about the magnitude of differences, but not directly about the probability that one protein is superior to another.
        """  # noqa: E501
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Probability of superiority

        This is where the probability of superiority calculation shines: it integrates over the entire posterior distribution to produce a single, interpretable number that directly answers our key question: "What is the probability that protein A is better than protein B?"

        The probability of superiority is calculated by:

        1. For each posterior sample, compare protein A's activity to protein B's
        2. Count the proportion of samples where A > B
        3. This gives us P(A > B) - a single number from 0 to 1

        This approach:

        1. **Integrates over uncertainty**: Uses the full posterior distribution, not just point estimates
        2. **Directly answers our question**: Tells us the probability of superiority, not just a difference magnitude
        3. **Avoids arbitrary references**: Can compare any two proteins directly
        4. **Produces an interpretable metric**: A probability from 0 to 1 is intuitive for decision-making

        Let's illustrate how the superiority probability integrates over the posterior by looking at the comparison between two specific proteins:
        """  # noqa: E501
    )
    return


@app.cell
def _(np, plt, sns, trace):
    def _():
        # Get posterior samples
        prot_activity = trace.posterior["prot_activity"].values
        prot_activity_flat = prot_activity.reshape(-1, prot_activity.shape[2])

        # Get protein names
        protein_names = trace.posterior["protein"].values

        # Choose two proteins to compare
        protein1 = "Protein_12"  # A high performer
        protein2 = "Protein_66"  # Another high performer

        idx1 = np.where(protein_names == protein1)[0][0]
        idx2 = np.where(protein_names == protein2)[0][0]

        # Extract their posterior samples
        samples1 = prot_activity_flat[:, idx1]
        samples2 = prot_activity_flat[:, idx2]

        # Calculate differences for each posterior sample
        differences = samples1 - samples2

        # Calculate superiority probability
        prob_superiority = np.mean(differences > 0)

        # Plot the posterior of differences
        plt.figure(figsize=(10, 6))

        # Histogram of differences
        sns.histplot(differences, bins=30, alpha=0.6)

        # Add vertical line at zero
        plt.axvline(x=0, color="r", linestyle="--")

        # Shade the area where protein1 > protein2
        positive_mask = differences > 0
        plt.fill_between(
            np.sort(differences[positive_mask]),
            0,
            plt.gca().get_ylim()[1] / 2,  # Half height for visibility
            alpha=0.3,
            color="green",
            label=f"P({protein1} > {protein2}) = {prob_superiority:.3f}",
        )

        plt.xlabel(f"Activity Difference ({protein1} - {protein2})")
        plt.ylabel("Frequency")
        plt.title(
            "Posterior Distribution of Activity Difference "
            f"Between {protein1} and {protein2}"
        )
        return plt.legend()

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        This visualization demonstrates the core concept behind the probability of superiority calculation. The green shaded area represents the proportion of posterior samples where the first protein outperforms the second. This proportion (in this case, approximately 0.6) is the probability of superiority.

        Rather than reducing our rich posterior distributions to point estimates or even to effect size distributions that still require interpretation, the superiority probability directly integrates over all our uncertainty to answer the precise question we care about: "How likely is it that this protein is better than that one?"

        This approach is particularly valuable when:

        - Uncertainty differs between proteins
        - Point estimates are similar but distributions differ in shape
        - You need a clear decision metric for ranking or selection

        Next, we'll see how to calculate this for all pairs of proteins to create a comprehensive superiority matrix.
        """  # noqa: E501
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Examining Replicate Effects

        Before moving to protein superiority, let's quickly look at the replicate effects across experiments. This helps us understand the magnitude of technical variation within each experiment.
        """  # noqa: E501
    )
    return


@app.cell
def _(az, trace):
    az.plot_forest(trace.posterior["rep_effect"], rope=[-0.05, 0.05])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The forest plot shows the replicate effects across different experiments. The "ROPE" (Region of Practical Equivalence) from -0.05 to 0.05 helps identify which effects are practically significant. Replicate effects that include the ROPE in their credible intervals might be considered negligible for practical purposes.

            Now that we've examined both replicate effects and explored why effect sizes are insufficient for robust protein comparison, we're ready to implement the Bayesian superiority calculation we just described - a method that integrates over the full posterior distribution to determine which proteins truly outperform others.
        """  # noqa: E501
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Bayesian Superiority Calculation

        Now that we've properly modeled the sources of variation and examined the protein activities, we can tackle our second question: which proteins are truly superior?

        The traditional approach would be to compare point estimates (means) and judge significance using p-values. But the Bayesian approach offers something more powerful: the probability that one protein is superior to another, calculated directly from posterior samples.

        Here's how we calculate a "superiority matrix":
        """  # noqa: E501
    )
    return


@app.cell
def _(np, trace):
    from tqdm.auto import tqdm

    def _():
        n_proteins = trace.posterior["prot_activity"].shape[-1]
        prot_activity = trace.posterior["prot_activity"].values.reshape(-1, n_proteins)

        superiority_matrix = np.zeros((n_proteins, n_proteins))

        for i in tqdm(range(n_proteins)):
            for j in range(n_proteins):
                if i != j:
                    superiority_matrix[i, j] = np.mean(
                        prot_activity[:, i] > prot_activity[:, j]
                    )
        return superiority_matrix

    superiority_matrix = _()
    return superiority_matrix, tqdm


@app.cell
def _(mo):
    mo.md(
        r"""
        This superiority matrix gives us, for each pair of proteins (i, j), the probability that protein i has higher activity than protein j, estimated from our posterior samples. This calculation incorporates all the uncertainty in our model.

        The calculation is straightforward but powerful:

        1. For each pair of proteins (i,j)
        2. For each posterior sample, check if protein i's activity exceeds protein j's
        3. Calculate the proportion of samples where i > j

        This gives us a probability interpretation: "There's an 85% chance that protein A is superior to protein B" rather than the frequentist approach of "protein A is significantly better than protein B with p<0.05."

        Let's visualize this matrix:
        """  # noqa: E501
    )
    return


@app.cell
def _(plt, sns, superiority_matrix):
    # Create heatmap
    sns.heatmap(superiority_matrix, annot=True, cmap="YlOrRd", fmt=".2f")
    plt.title("Superiority Matrix")
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The heatmap shows the pairwise superiority probabilities between all proteins. Brighter colors indicate higher probabilities that the row protein is superior to the column protein.

        We can also look at the column-wise mean of the superiority matrix, which gives us a different perspective on protein performance:
        """  # noqa: E501
    )
    return


@app.cell
def _(sns, superiority_matrix):
    sns.heatmap(superiority_matrix.mean(axis=0).reshape(-1, 1))
    return


@app.cell
def _(mo):
    mo.md(
        r"""We can rank proteins by their average probability of superiority. While this may seem disingenuous, given that I just ranted against point estimates, it gives us at least one meaningful basis of comparison. Let's plot both the average probability of superiority and the full underlying pairwise comparisons."""  # noqa: E501
    )
    return


@app.cell
def _(df, np, pd, plt, sns, superiority_matrix):
    def _():
        # Calculate average probability of superiority and sort proteins
        avg_superiority = superiority_matrix.mean(axis=1)
        protein_names = df["Protein"].unique()
        superiority_df = pd.DataFrame(
            {"Protein": protein_names, "Avg_Superiority": avg_superiority}
        )
        sorted_superiority = superiority_df.sort_values(
            "Avg_Superiority", ascending=False
        ).head(20)

        # Create plot
        plt.figure(figsize=(12, 6))

        # For each protein, plot individual points and mean line
        for i, protein in enumerate(sorted_superiority["Protein"]):
            protein_idx = np.where(protein_names == protein)[0][0]
            protein_probs = superiority_matrix[protein_idx]
            plt.scatter(
                [i] * len(protein_probs), protein_probs, alpha=0.5, color="blue"
            )
            plt.hlines(avg_superiority[protein_idx], i - 0.25, i + 0.25, color="red")

        plt.xticks(
            range(len(sorted_superiority)), sorted_superiority["Protein"], rotation=90
        )
        plt.ylabel("Probability of Superiority")
        plt.title("Distribution of Superiority Probabilities by Protein")
        plt.tight_layout()
        sns.despine()
        return plt.gca()

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""Looking at our results, some proteins emerge as the clear top performers, followed by others with moderate superiority. This ranking is particularly valuable because it differs from what we might conclude by simply examining forest plots of the posterior distributions. In forest plots, we would only see the estimated activity levels and their uncertainty intervals, which might lead us to favor proteins with high mean activity but also high uncertainty. The superiority metric, in contrast, directly quantifies the probability that one protein outperforms others, properly accounting for the full posterior distribution and the uncertainty in each comparison."""  # noqa: E501
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Relationship Between Protein Activity and Superiority

        To better understand how protein activity relates to superiority probability, let's create a scatter plot that compares:

        1. The posterior mean protein activity (x-axis)
        2. Two measures of superiority probability (y-axis):
           - Mean probability of superiority
           - Minimum probability of superiority (a conservative estimate)

        This will help identify proteins that are consistently superior versus those that have high mean activity but with high uncertainty.
        """  # noqa: E501
    )
    return


@app.cell
def _(np, plt, sns, superiority_matrix, trace):
    def _():
        # Calculate probability of superiority distribution for each protein
        superiority_dist = [
            np.concatenate([superiority_matrix[i, :j], superiority_matrix[i, j + 1 :]])
            for i, j in enumerate(range(len(superiority_matrix)))
        ]

        # Get protein activity statistics from trace
        protein_activity_mean = (
            trace.posterior["prot_activity"].mean(dim=["chain", "draw"]).values
        )

        # Create scatter plot
        plt.figure(figsize=(10, 6))

        # Plot points
        plt.scatter(
            protein_activity_mean,
            [np.mean(dist) for dist in superiority_dist],
            alpha=0.6,
            label="mean p(superior)",
        )
        plt.scatter(
            protein_activity_mean,
            [np.percentile(dist, q=0) for dist in superiority_dist],
            alpha=0.6,
            label="minimum p(superior)",
        )

        plt.xlabel("Posterior Mean Protein Activity (log scale)")
        plt.ylabel("Probability of Superiority")

        plt.legend()
        plt.title("Protein Activity vs Probability of Superiority")
        sns.despine()
        return plt.gca()

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        This plot reveals important insights:

        1. The relationship between activity and superiority is non-linear
        2. Proteins with similar activities can have different probabilities of superiority depending on certainty
        3. 3. The minimum probability of superiority (0th percentile) provides a conservative measure for decision making -- a protein with a high minimum probability of superiority is more likely to be the superior candidate than a protein with a low minimum probability of superiority.

        """  # noqa: E501
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Practical Applications Beyond Protein Screening

        While we've focused on protein screening, the techniques demonstrated here apply broadly:

        1. **Drug Discovery**: Compare efficacy of different compounds while accounting for batch effects
        2. **Materials Science**: Evaluate materials' properties with appropriate uncertainty quantification
        3. **A/B Testing**: Assess the probability that one variant truly outperforms another
        4. **Clinical Trials**: Calculate the probability that a treatment is superior to alternatives

        In each case, the R2D2 prior helps answer "Is our experiment measuring what we care about?" while Bayesian superiority calculation addresses "Which option is truly better?"
        """  # noqa: E501
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Conclusion

        In high-throughput screening, two questions are critical: "Are we measuring what matters?" and "Which candidates are truly superior?" Traditional approaches using point estimates and p-values fail to adequately address these questions, especially when dealing with experimental noise and multiple comparisons.

        The Bayesian framework we've demonstrated offers a powerful alternative:

        1. **R2D2 priors** decompose variance into interpretable components, revealing how much of our signal comes from the biological effect versus experimental artifacts. This guides concrete improvements to experimental protocols.
        2. **Bayesian superiority calculation** directly quantifies the probability that one candidate outperforms others, properly accounting for uncertainty and avoiding the pitfalls of simple rank ordering.

        Together, these techniques transform screening data into actionable insights. While we've focused on protein screening, the same approach applies to any domain requiring robust comparison of multiple candidates under noisy conditions.

        Bayesian methods allow us to move beyond simplistic "winners and losers" to a nuanced understanding of which candidates are most likely to succeed, with what degree of certainty, and how we can improve our measurement process itself. And more meta-level: none of this needs fancy names. It's just logic and math, applied to data. Wasn't that what statistics was supposed to be?
        """  # noqa: E501
    )
    return


if __name__ == "__main__":
    app.run()
