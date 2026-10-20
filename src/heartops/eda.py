"""Descriptive EDA figures. EDA describes all patients; it never drives model selection."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data import CATEGORICAL, FEATURES, NUMERIC

LABELS = {0: "absent", 1: "present"}
PALETTE = {"absent": "#4C72B0", "present": "#DD8452"}


def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close("all")


def summary_table(data):
    """Per-feature dtype, missing count, unique values and range."""
    return pd.DataFrame({"missing": data[FEATURES].isna().sum(),
                         "unique": data[FEATURES].nunique(),
                         "min": data[FEATURES].min(), "median": data[FEATURES].median(),
                         "max": data[FEATURES].max()})


def make_figures(data, figures=Path("reports/figures")):
    figures = Path(figures)
    figures.mkdir(parents=True, exist_ok=True)
    labelled = data.assign(diagnosis=data.target.map(LABELS))

    data[FEATURES].hist(figsize=(14, 10), bins=15, color="#4C72B0", edgecolor="white")
    plt.suptitle("Feature histograms (all 303 patients)")
    _save(figures / "histograms.png")

    plt.figure(figsize=(11, 9))
    sns.heatmap(data.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, square=True)
    plt.title("Pearson correlation (categorical codes are descriptive only)")
    _save(figures / "correlations.png")

    counts = labelled.diagnosis.value_counts().reindex(["absent", "present"])
    axis = counts.plot.bar(color=[PALETTE[k] for k in counts.index], rot=0)
    axis.bar_label(axis.containers[0],
                   labels=[f"{n} ({n / len(data):.0%})" for n in counts])
    axis.set(title="Class balance", xlabel="Heart disease", ylabel="Patients")
    _save(figures / "class_balance.png")

    _, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, column in zip(axes.flat, NUMERIC):
        sns.kdeplot(data=labelled, x=column, hue="diagnosis", fill=True, common_norm=False,
                    palette=PALETTE, ax=axis, warn_singular=False)
    plt.suptitle("Numeric feature density by diagnosis")
    _save(figures / "numeric_by_target.png")

    _, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, column in zip(axes.flat, CATEGORICAL):
        rate = data.groupby(column).target.agg(["mean", "size"])
        axis.bar(rate.index.astype(str), rate["mean"], color="#DD8452")
        axis.bar_label(axis.containers[0], labels=[f"n={n}" for n in rate["size"]], fontsize=8)
        axis.set(title=column, ylim=(0, 1.1), ylabel="Disease rate")
    plt.suptitle("Observed disease rate per category code")
    _save(figures / "categorical_rates.png")

    _, axes = plt.subplots(1, len(NUMERIC), figsize=(16, 4))
    for axis, column in zip(axes, NUMERIC):
        sns.boxplot(data=labelled, x="diagnosis", y=column, hue="diagnosis",
                    palette=PALETTE, ax=axis, legend=False)
    plt.suptitle("Outliers and spread by diagnosis")
    _save(figures / "boxplots.png")

    missing = data[FEATURES].isna().sum()
    axis = missing.plot.barh(color="#C44E52", figsize=(7, 5))
    axis.set(title="Missing values per feature (raw '?' markers)", xlabel="Count")
    _save(figures / "missing_values.png")
    return sorted(p.name for p in figures.glob("*.png"))
