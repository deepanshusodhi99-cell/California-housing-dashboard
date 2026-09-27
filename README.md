# California Housing Price Prediction Dashboard

A regression-model dashboard predicting median house values from the
**California Housing Prices** dataset (1990 census) — real public data, real
trained model, real metrics.

## Why California Housing, not Boston Housing

The classic "Boston Housing" dataset was **removed from scikit-learn in
version 1.2 (2022)** because one of its features encoded neighborhood racial
composition as a predictor of home value, which the ML community flagged as
an ethically compromised design choice. California Housing is scikit-learn's
standard modern replacement for this exact type of regression-modeling demo,
with no such issue.

## What's real here

- **Data**: 20,433 real property records (after dropping rows with missing
  values) from the 1990 California census — longitude/latitude, housing age,
  rooms, population, income, proximity to ocean, and median house value.
- **Model**: a scikit-learn `RandomForestRegressor` in a `Pipeline` with
  one-hot encoding for the categorical `ocean_proximity` feature.
- **Metrics**: R² and Mean Absolute Error shown on the dashboard are the
  model's actual performance on a held-out 20% test set — not illustrative
  numbers.

## Tech stack

- **Python** — pandas (data prep), scikit-learn (model training/evaluation)
- **JavaScript + Chart.js** — scatter plot, feature importance, residual
  histogram
- **HTML/CSS** — static, no framework

## Structure

```
.
├── housing.csv          # real dataset
├── generate_model.py     # trains the model, writes model_results.json
├── model_results.json    # model output consumed by the dashboard
├── index.html             # the dashboard
└── README.md
```

## Running locally

```bash
python generate_model.py    # re-trains the model, regenerates model_results.json
python -m http.server 8000  # serve locally (fetch() needs http, not file://)
# open http://localhost:8000
```

## Live demo

Enable GitHub Pages (Settings → Pages → deploy from `main`, root):
`https://deepanshusodhi99-cell.github.io/california-housing-dashboard/`
