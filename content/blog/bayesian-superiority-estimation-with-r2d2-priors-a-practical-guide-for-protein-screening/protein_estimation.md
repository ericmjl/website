```
import marimo as mo

```

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


```
import pymc as pm
```

## The Protein Screening Example

We'll use a dataset with fluorescence measurements for over 100 proteins
across multiple experiments and replicates, with an experimental design that includes:

- A control protein present in all experiments and replicates
- "Crossover" proteins measured across all experiments
- Unique test proteins in each experiment

This design is common in high-throughput screening where measuring all proteins in all conditions is impractical.
We'll implement our analysis using PyMC, a powerful probabilistic programming framework for Bayesian modeling.

## Generating Synthetic Data

To demonstrate our approach, we'll generate synthetic data that mimics a realistic protein screening experiment with:

- 3 experiments with 2 replicates each
- A control protein and crossover proteins present in all experiments
- Test proteins unique to each experiment
- Systematic experiment effects (batch effects)
- Replicate-level variation
- Measurement noise


```
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
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Experiment</th>
      <th>Replicate</th>
      <th>Protein</th>
      <th>Fluorescence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Exp_1</td>
      <td>Rep_1</td>
      <td>Control</td>
      <td>1087.476281</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Exp_1</td>
      <td>Rep_1</td>
      <td>Crossover_0</td>
      <td>1054.176224</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Exp_1</td>
      <td>Rep_1</td>
      <td>Crossover_1</td>
      <td>955.647739</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Exp_1</td>
      <td>Rep_1</td>
      <td>Crossover_2</td>
      <td>1091.751188</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Exp_1</td>
      <td>Rep_1</td>
      <td>Crossover_3</td>
      <td>1189.344109</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>205</th>
      <td>Exp_3</td>
      <td>Rep_2</td>
      <td>Protein_85</td>
      <td>1765.149428</td>
    </tr>
    <tr>
      <th>206</th>
      <td>Exp_3</td>
      <td>Rep_2</td>
      <td>Protein_86</td>
      <td>1595.422298</td>
    </tr>
    <tr>
      <th>207</th>
      <td>Exp_3</td>
      <td>Rep_2</td>
      <td>Protein_87</td>
      <td>1889.585595</td>
    </tr>
    <tr>
      <th>208</th>
      <td>Exp_3</td>
      <td>Rep_2</td>
      <td>Protein_88</td>
      <td>1394.395041</td>
    </tr>
    <tr>
      <th>209</th>
      <td>Exp_3</td>
      <td>Rep_2</td>
      <td>Protein_89</td>
      <td>1411.831297</td>
    </tr>
  </tbody>
</table>
<p>210 rows × 4 columns</p>
</div>


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

## Examining the Raw Data

Before modeling, let's visualize the control and crossover proteins to understand experimental variation:


```
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
```



![png](protein_estimation_files/protein_estimation_8_0.png)



Notice the dramatic shift in fluorescence values across experiments.
Experiment 3 shows substantially higher fluorescence readings (around 1500-2000 units)
compared to Experiments 1 and 2 (mostly below 1300 units).

This pattern illustrates a common challenge in high-throughput screening:
significant batch effects between experiments that can mask the true biological signal.
Without accounting for these experimental factors, we might incorrectly attribute
higher activity to proteins simply because they were measured in Experiment 3.

Our modeling approach needs to account for both between-experiment and within-experiment
sources of variation to accurately compare proteins across the entire dataset.

## The R2D2 Prior for Variance Decomposition

The [R2D2 prior (R-squared Dirichlet decomposition)](https://arxiv.org/abs/2111.10718) provides an interpretable framework
for variance decomposition by placing a prior on the Bayesian coefficient of determination ($R^2$),
which then induces priors on individual parameters.

The core idea is simple but powerful:

1. Place a $\text{Beta}(a,b)$ prior on $R^2$ (the proportion of variance explained by the model)
2. Induce a prior on the global variance parameter representing total variance
3. Decompose the global variance into components via a Dirichlet distribution

This approach has key advantages over traditional hierarchical models:

- Instead of specifying arbitrary priors for each variance component separately, we control
  total explained variance through a single interpretable parameter ($R^2$)
- The variance partition becomes meaningful through the Dirichlet distribution
- The hierarchical nature ensures regularization of variance estimates

Here's our implementation of the model:


```
def _():
    # create categorical indices
    exp_idx = pd.Categorical(df["Experiment"]).codes
    rep_idx = pd.Categorical(df["Replicate"]).codes
    prot_idx = pd.Categorical(df["Protein"]).codes

    # define coords for dimensions
    coords = {
        "experiment": df["Experiment"].unique(),
        "replicate": df["Replicate"].unique(),
        "protein": df["Protein"].unique(),
        "component": ["experiment", "replicate", "protein"],
    }

    with pm.Model(coords=coords) as model:
        # 1. residual variance (σ²) stands alone
        sigma_squared = pm.HalfNormal("sigma_squared", sigma=1)
        residual_sd   = pm.Deterministic("residual_sd", pm.math.sqrt(sigma_squared))

        # 2. R² prior and induced total signal variance W
        r_squared = pm.Beta("r_squared", alpha=1, beta=1)
        W         = pm.Deterministic(
            "W", sigma_squared * r_squared / (1 - r_squared)
        )

        # 3. split W among the three signal components
        props = pm.Dirichlet("props", a=np.ones(3), dims="component")
        exp_var, rep_var, prot_var = props * W

        # component standard deviations
        exp_sd  = pm.Deterministic("exp_sd", pm.math.sqrt(exp_var))
        rep_sd  = pm.Deterministic("rep_sd", pm.math.sqrt(rep_var))
        prot_sd = pm.Deterministic("prot_sd", pm.math.sqrt(prot_var))

        # 4. group effects
        exp_effect  = pm.Normal("exp_effect", mu=0, sigma=exp_sd, dims="experiment")
        rep_effect  = pm.Normal(
            "rep_effect", mu=0, sigma=rep_sd, dims=("experiment", "replicate")
        )
        prot_effect = pm.Normal("prot_effect", mu=0, sigma=prot_sd, dims="protein")

        # global mean on log scale
        global_mean = pm.Normal("global_mean", mu=7, sigma=1)

        # expected value for each observation
        y_hat = (
            global_mean
            + exp_effect[exp_idx]
            + rep_effect[exp_idx, rep_idx]
            + prot_effect[prot_idx]
        )

        # optional: compute implied model R² for checking
        model_r2 = pm.Deterministic(
            "model_r2",
            (exp_var + rep_var + prot_var)
            / (exp_var + rep_var + prot_var + sigma_squared),
        )

        # 5. likelihood uses only the residual_sd
        y = pm.Normal(
            "y",
            mu=y_hat,
            sigma=residual_sd,
            observed=np.log(df["Fluorescence"]),
        )

        # sample
        trace = pm.sample(2000, tune=1000, return_inferencedata=True, nuts_sampler="nutpie")

    return model, trace

model, trace = _()
```

## Model Convergence and Variance Analysis

After fitting our model, zero divergent transitions confirm good convergence.

Let's examine how variance is partitioned across components:


```
trace.sample_stats.diverging.sum()
```


<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1f1f1f;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: inline-block;
  opacity: 0;
  height: 0;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:focus + label {
  border: 2px solid var(--xr-font-color0);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: "▼";
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: "(";
}

.xr-dim-list:after {
  content: ")";
}

.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;diverging&#x27; ()&gt; Size: 8B
array(0)</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'diverging'</div></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-c99a31a0-17a2-456b-bcbf-974a2ca5805f' class='xr-array-in' type='checkbox' checked><label for='section-c99a31a0-17a2-456b-bcbf-974a2ca5805f' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0</span></div><div class='xr-array-data'><pre>array(0)</pre></div></div></li><li class='xr-section-item'><input id='section-b8c16d3c-86d4-4e87-8354-9c81177af604' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-b8c16d3c-86d4-4e87-8354-9c81177af604' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-1c2fb95c-816e-4ef2-a1d5-b5b59328304b' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-1c2fb95c-816e-4ef2-a1d5-b5b59328304b' class='xr-section-summary'  title='Expand/collapse section'>Indexes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-0dd3a477-2091-44af-9871-42810eeac5dd' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-0dd3a477-2091-44af-9871-42810eeac5dd' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>


In the ridgeline plot below, we see that unexplained variance contributes minimally to total variation,
while experiment and replicate effects—which ideally should contribute little—are actually
significant contributors to the readout variation. This serves as a metric of laboratory consistency.

Ideally, the protein variation (our biological signal of interest) should be the dominant source
of variation. This analysis suggests that improving experimental protocols to reduce batch effects
would substantially improve our signal-to-noise ratio.


```
import arviz as az

axes_posterior_props = az.plot_posterior(trace, var_names=["props"], grid=(2, 2))
axes_posterior_props.flatten()[0].set_title("experiment")
axes_posterior_props.flatten()[2].set_title("replicate")
axes_posterior_props.flatten()[1].set_title("protein")
axes_posterior_props.flatten()[3].set_title("unexplained")
```



![png](protein_estimation_files/protein_estimation_15_0.png)



These posterior plots show the distributions of the variance components. Each represents the proportion of total variance attributed to that component. We can see clear differences in the contributions from each source.

We can also look at the total $R^2$ value, which represents the proportion of variance explained by the model:


```
az.plot_posterior(trace, var_names=["model_r2", "r_squared"])
```


<pre>Axes(0.125,0.110;0.352x0.770)
Axes(0.548,0.110;0.352x0.770)</pre><br><img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABIgAAAHxCAYAAAAGOYdtAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjEsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvc2/+5QAAAAlwSFlzAAAPYQAAD2EBqD+naQAAkiZJREFUeJzt3Qd4m9XVwPHjvfd2hh0ncfbeZJOEPcLeq7RAW1ootIxCoS2lQD+gFFpGyyylBUrZM0D23nvvxInjvbf0fs+9khw7thM7Hu8r6f97HqPXmlcRko7PvfccH8MwDAEAAAAAAIDX8jV7AAAAAAAAADAXCSIAAAAAAAAvR4IIAAAAAADAy5EgAgAAAAAA8HIkiAAAAAAAALwcCSIAAAAAAAAvR4IIAAAAAADAy5EgAgAAAAAA8HIkiAAAAAAAALwcCSIATdx8883i4+Mjb775ZofcX3p6ur6//fv3S2c6ePCgvPLKK3LppZdKWlqaBAUFSXh4uAwbNkx+/etfS25ubqc+PgAAADrHtGnTdDw5f/58s4cCeCx/swcAAB3l2muvlSVLloi/v7+MGDFCJkyYIAUFBbJixQp54okn5NVXX5U5c+bI8OHDzR4qAAAAAFgKCSIAHqNbt27y5z//WW644QaJi4urP1+tHLryyiv1jJM63bZtm/j5+Zk6VgAAAACwEhJEADzGe++91+z5CQkJ8vbbb0uPHj1k165dsmzZMpk0aVKXjw8AAAAArIoaREAXUXum1Y/yr3/9S8aOHavr46jkxTXXXKPr5yiGYchf//pXvQ0qLCxM4uPjdU2gnJycFu/7m2++kQsuuEASExMlMDBQUlNT5aqrrpLVq1e3eBu19eruu++ur9XTs2dPufPOO/X5p/L999/rOj8pKSn68dTjXnLJJTrx0hX7zhctWiQXXnih/rfz9fVtVa2k7t27639L5dChQ502TgAA4P6x2htvvKG3qkdFRZ12HcXi4mJ5+OGHZciQITqmU/GWitEmTpwojzzyiNTW1ja5jYqlzj33XImOjtZx4ujRo+X1119vMj4XNS51nqr32NZakFu3bpVHH31Uj0etwlYxnVqBPXPmTHn//febvS8Vh6n7UnFZRUWFfh4DBgyQ0NDQJmNYs2aNXHfddTrGVM89NjZWzj77bPnyyy9bHKuK0X7wgx/oGDM4OFj69u0rDz30kFRWVrZ4GwAdhwQR0MUefPBBueWWWyQiIkIHAOoL9d1339UrWgoLC+Xqq6+WX/3qV/qLUX2Jqq1Qb731lsyaNUtqamqa3N9vfvMbOeecc/SXbWZmplx++eWSlJSkv9jHjx9fH1Q0dOzYMX3ZX/7yFyktLdXJpVGjRsk777yjE1dqHC355S9/qQOHTz75RH/hz549WzIyMvTvkydP1gFVZ/rvf/+rg5K9e/fqcah/FxV0nEpeXl7981L/tgAAAM352c9+Jj/84Q91TcPzzz9fxo0b1yQxcyoqeaJiu8cff1zHXTNmzNCTa/369dMxzGOPPSbl5eVNYhwVS3399dd61fNFF10kISEheiz33ntvBz9LkWeffVZ+//vf68lBlcRyjW/evHl6ovGee+5p8bZVVVU6HlP30atXLz1WlcxxUTGmiin//e9/66STunzQoEE6waT+TdXjnmj79u06IaZiSfXvrW6jYltVPkD9+zUXBwPoYAaALqHebuonLi7OWL9+ff35FRUVxqRJk/RlQ4YMMXr37m3s37+//vLc3FyjT58++vJ//etfje7zq6++0ucHBwcbc+bMaXTZq6++qi8LCAgwNm/e3Oiyyy+/XF82efJko6ioqP78/Px8Y9y4cfVjfeONNxrd7u9//7s+X41nw4YNjS5bsGCBERERYQQGBho7d+5sdFlaWpq+3b59+4zTNXXq1Ppx/e1vf2vz7e+//35925SUFKO6uvq0xwEAADyTK86IjIw0li1b1q77euutt/R9nXvuuUZNTU2jy2w2mzF//vxG8cjRo0d1HKVu8+yzzza6/nfffadjPdf4GlKxlTpPxVotaSkOU2PYs2dPk+tv377d6N69u77NihUrGl02b968+nEMHTpUj/tEX3/9teHj42PEx8fr+LChjRs31t+3evyGxowZo8+/8sorjcrKyvrzDxw4oONj1+OqMQDoHKwgArqYmjFRbddd1MyQa4Zm06ZN8vzzz+ttXy5qW9SPf/zj+q1dDT399NP69Cc/+YleSdPQrbfeqlcGqeXLahan4dLdDz/8UM/MvPzyy3rptIta+qvOa47dbpff/va3+liteBo6dGijy6dMmaJXM6nZHdVqvrOceeaZ+vm2xXfffVf/b/XMM8/oJdQAAAAtrZZWK63bQ60aUlR8FhAQ0OgytT1+6tSpjeKR1157Ta/qVo/7i1/8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SZ/OJdAAAABe5EB+hRwurJQAPx8Zn2H9ib2GcZua2FNb44BTIUEEACY5Pgtl7jLlLVu2yBVXXCEJCQkSEhIiQ4YMkeeee07sdnuT687o75iJWrgzT2rqml5us9nkhRdekFGjRklYWJhERUXJlClT5MMPP2z2sd988009w3Wqn3/+85+d8MwBAADatr1sVFqMhAX5u0XcNjo9RiKC/aWgvEbWO4trN0TchhOZ9382AHixOptdljhXEE3JNG+Z8rJly2TGjBlSWVkpY8eOlfT0dFm4cKH84he/kKVLl8p7773XqHj2kG5RkhARJLml1bJyX4FMarDEWgUZs2fPls8//1zCw8Nl0qRJOlhR93PZZZfJo48+Kr/97W8bPX6fPn3kpptuanZsxcXF8vHHH+tjdV8AAABmUZNjZtcfamvcFuDnK9P6JcpnG47I99uO6eSWC3EbmmUAALrc6v0FRtr9nxtDf/uNUWezmzKGmpoao1evXmq9sfHss8/Wn19aWmpMmDBBn//GG280ud19/92gx/7bTzc3Ov/pp5/Wt0lPTzd2795df/62bduM1NRUfdnSpUtbPb4XX3xR32bixImn/RwBAADaq7rWZgz8zVc6/tl0uMit4raP1h7W4z77zwsanU/chuawxQwATLDIuUx5Up94XaTaDB999JHs27dPhg0bpmeeXNQs0l//+ld9/MwzzzS53ZmugofbcnTBRpeXXnpJnz7++OPSu3fv+vP79+9fPwP1pz/9qdXj+9e//qVPb7jhhtN4dgAAAB1j3cFCKa+xSVxYoAxMiXSruG1qZoKoUHN7dqkcLqyoP5+4Dc0hQQTAY+zfv18vq502bZqUl5fLPffcIz169ND7s0eOHCmfffZZ/XX/+9//yrhx4/R+66SkJPn5z3+ul+ueqKKiQp544gkZMWKE/gJWP+PHj5e33nqr2TEsWrRI7rzzThk6dKjExMTox1ZftA888IAUFRU1qT8UV7JLj/nmm2+WgoIC+fGPfywpKSkSFBQkgwcPltdff106yxdffKFPL7/88iaXqX+vjIwM2bx5s/53bUgltQL9fOVgQYXsyS2rX1a8Z88efaz+/U80ffp0ffrNN99IdXX1KcemAiC1xDkwMFCuvPLK03yGAADAqtwqbnNO7KXX7hc/P1+3ittiwgJldFqsPp63PUefErehJSSIAHicmpoa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ntGmjoXYzfNjt2muBNHOXNrddyISRAAsoc5mly83ObeXDTN3e9mHH36ov9T79esnjz76aP35aubnj3/8o/4yVV+uGzZsaPE+1MzS7t275eWXX5aAgIBmr7N//34pKiqSq6++Ws9wKddcc40EBwfroKOh++67T9/PH/7wB7Ea9e+iZulWrFihly0vWLBAXnrpJbE6V6CxcFcu7e4BAGijzzYcLwtg1uoYF2I3z4/dxvSKkdBAP8ktrZYtR6y/4sldkSACYAkr9hVIXlmNRIcGyKQ+jg5TZlmzZo0+nTJlivj6Nv6YVF/0EydO1MeffPJJs7fftGmT/N///Z/84Ac/0LM2LSksLNSnMTEx9eepx1NLll2XKUuXLpW3335bnnjiCYmNtW4ByMTERPn973+vj59//nmxuhE9oiUy2F+KKmpl/SHa3QMA0Frl1XXy/fZjluleRuzm+bGbo92942+EBZQH6DQkiABYwucbHdvLzh2crNuQm6m8vLzJl39DcXFx+rS5WShVuPC2227Te9/VEuST6dmzpz7duXNn/XkquFDLlF2XqftTxRVHjRp1yqXRVnDllVfqQGnPnj2t2utvJn/d7t6xzYxuZgAAtN53245JVa1d0uNCZXA3R00/MxG7eUfsNs3ZZIQ6RJ2HBBEA09Xa7PLVZmeRQwvMQiUkOL58Dhw40Ozl+/bta/FyVYRw+fLl8vTTT59yxig5OVlGjhwpb7zxhu6SoQKMe+65RwcW559/vr6OWuasliyr+z1xRsyKIiIiJCMjQx+roozu0jZ1HnWIAABoteNdZ1NN316mELt5R+zmShCtOVAoxRXWrpnkrqz/fywAj7d4d57e5hMfHiTjMhwzPGZSy5OVL774oklb1KysLPn222/18YndKtSsy0MPPaRbh954442teqxnnnlGz3pNnjxZByWqgOB5552n94Xn5+fLb37zG73ceezYsY2KKKpA5HSd2PK04c8tt9wi7RUf71j+23CptVW52t1vyirWe9oBAMDJFVfWygLnxIoVupcpxG7eEbt1jwmVvonhokpHLtrN5F5n8O+UewWA0+hedv6QZPHzNX8W6qyzztKzQ2vXrpVzzz1XzwANHDhQ709XnSjq6ur09U6cFfrpT38q1dXVbSrypwIS9Thqn7oqejhu3Di54YYb9GUPPvig7tLw5JNP6t+///57+fnP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'/>


Taken together, we can interpret that the model fits the data very well (`model_r2` close to 1), but it is concerning to me that protein only explains 19% of the variation in readout, while experiment and replicate explains more than 70% of the output variation, which signals to me that the measurements are not particularly tight, and a lot could be done to control experiment-to-experiment variation. Additionally, the fact that our `model_r2` and `r_squared` parameters are self-consistent is reassuring as well.

## Protein Activity Estimates

Now that we've decomposed the variance and accounted for experimental effects,
let's examine the protein activity estimates—the "true" biological signal after
removing experimental noise:


```
ax = az.plot_forest(trace.posterior["prot_effect"])[0]
ax.set_xlabel("log(protein activity)")
```



![png](protein_estimation_files/protein_estimation_20_0.png)



The forest plot displays posterior distributions of protein activity (log scale),
with horizontal lines representing 94% credible intervals.

A key challenge emerges: despite similar uncertainty across proteins,
overlapping credible intervals make it difficult to determine which proteins
are truly superior. Simply ranking by posterior means could lead us to prioritize
proteins with slightly higher point estimates when uncertainty makes their actual
superiority ambiguous.

This is a fundamental limitation of ranking by point estimates alone:
it fails to properly account for uncertainty. A protein with a slightly lower mean
but narrower credible intervals might be a better candidate than one with a higher
mean but wider uncertainty bounds.

## Why not calculate the effect sizes?

While effect sizes quantify difference magnitudes, they have important limitations:

1. They still have posterior distributions with uncertainty
2. They require an arbitrary reference protein
3. Scale interpretation is subjective
4. They don't directly answer "which is better?"

This is where the probability of superiority calculation shines: it integrates over
the entire posterior distribution to directly answer our key question:
"What is the probability that protein A is better than protein B?"

The probability of superiority:

1. For each posterior sample, compares protein A's activity to protein B's
2. Counts the proportion of samples where A > B
3. Results in P(A > B) - a single number from 0 to 1

This approach integrates over uncertainty, directly answers our question,
avoids arbitrary references, and produces an intuitive metric for decision-making.

Let's illustrate this with a comparison between two specific proteins:


```
def _():
    # Get posterior samples
    prot_activity = trace.posterior["prot_effect"].values
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
```



![png](protein_estimation_files/protein_estimation_23_0.png)



This visualization demonstrates the core concept. The green shaded area represents
the proportion of posterior samples where the first protein outperforms the second.
This proportion is the probability of superiority.

Rather than reducing our rich posterior distributions to point estimates or effect sizes
that still require interpretation, the superiority probability directly integrates over
all uncertainty to answer our precise question: "How likely is it that this protein is
better than that one?"

Now let's calculate this for all pairs of proteins to create a comprehensive superiority matrix:


```
from tqdm.auto import tqdm

def _():
    n_proteins = trace.posterior["prot_effect"].shape[-1]
    prot_activity = trace.posterior["prot_effect"].values.reshape(-1, n_proteins)

    superiority_matrix = np.zeros((n_proteins, n_proteins))

    for i in tqdm(range(n_proteins)):
        for j in range(n_proteins):
            if i != j:
                superiority_matrix[i, j] = np.mean(
                    prot_activity[:, i] > prot_activity[:, j]
                )
    return superiority_matrix

superiority_matrix = _()
```


  0%|                                                                                                   | 0/95 [00:00<?, ?it/s]


 40%|███████████████████████████████████▌                                                     | 38/95 [00:00<00:00, 372.42it/s]


 80%|███████████████████████████████████████████████████████████████████████▏                 | 76/95 [00:00<00:00, 373.04it/s]


100%|█████████████████████████████████████████████████████████████████████████████████████████| 95/95 [00:00<00:00, 372.76it/s]


This superiority matrix gives us, for each pair of proteins (i, j), the probability
that protein i has higher activity than protein j, incorporating all model uncertainty.

The calculation yields a probability interpretation: "There's an 85% chance that
protein A is superior to protein B" rather than the frequentist "protein A is
significantly better than protein B with p<0.05."

Let's visualize this matrix:


```
# Create heatmap
sns.heatmap(superiority_matrix, annot=False, cmap="YlOrRd", fmt=".2f")
plt.title("Superiority Matrix")
plt.gca()
```



![png](protein_estimation_files/protein_estimation_27_0.png)



We can rank proteins by their average probability of superiority across all comparisons:


```
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
        plt.scatter([i] * len(protein_probs), protein_probs, alpha=0.5)
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
```



![png](protein_estimation_files/protein_estimation_29_0.png)



This ranking differs from what we might conclude by examining forest plots alone.
The superiority metric directly quantifies the probability that one protein
outperforms others, properly accounting for the full posterior distribution
and uncertainty in each comparison.

To better understand how protein activity relates to superiority probability,
let's compare their posterior mean activity with two superiority measures:


```
def _():
    # Calculate probability of superiority distribution for each protein
    superiority_dist = [
        np.concatenate([superiority_matrix[i, :j], superiority_matrix[i, j + 1 :]])
        for i, j in enumerate(range(len(superiority_matrix)))
    ]

    # Get protein activity statistics from trace
    protein_activity_mean = (
        trace.posterior["prot_effect"].mean(dim=["chain", "draw"]).values
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
```



![png](protein_estimation_files/protein_estimation_31_0.png)



This plot reveals that:

1. The relationship between activity and superiority is non-linear
2. Proteins with similar activities can have different superiority probabilities
   depending on the certainty of their estimates
3. The minimum probability of superiority provides a conservative decision-making
   measure—a protein with high minimum superiority is more likely to be the superior
   candidate

## Conclusion

When screening candidates, two questions are critical: "Are we measuring what matters?"
and "Which candidates are truly superior?" Traditional approaches using point estimates
and p-values inadequately address these questions when dealing with experimental noise
and multiple comparisons.

A Bayesian model with explicitly modeled terms offers a powerful alternative:

1. **R2D2 priors** decompose variance into interpretable components, revealing how
   much signal comes from the biological effect versus experimental artifacts.
   This guides concrete improvements to experimental protocols.

2. **Bayesian superiority calculation** directly quantifies the probability that
   one candidate outperforms others, properly accounting for uncertainty and
   avoiding the pitfalls of simple rank ordering.

These techniques transform screening data into actionable insights and apply to
any domain requiring robust comparison of multiple candidates under noisy conditions:
drug discovery, materials science, A/B testing, clinical trials, and more.

Bayesian methods move us beyond simplistic "winners and losers" to a nuanced
understanding of which candidates are most likely to succeed, with what degree
of certainty, and how we can improve our measurement process itself.
