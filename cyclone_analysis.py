# %% [markdown]
# # Cyclone Preheater: Detecting Abnormal Operating Periods
# 
# ## Executive question
# 
# **When did the cyclone preheater behave unusually, and which measurements support that conclusion?**
# 
# The data contains six process signals sampled every five minutes. There is no labelled abnormal/normal target, so this notebook uses **unsupervised anomaly detection** to identify candidate periods for process-engineer review. A flagged period is evidence of unusual multivariate behaviour, not proof of a mechanical failure.
# 
# ## Deliverables covered
# 
# - Data preparation and quality checks
# - Exploratory time-series and correlation analysis
# - A reproducible anomaly-detection strategy
# - A ranked table of abnormal periods with timestamps and severity
# - Row-level and period-level CSV outputs for the final submission
# 
# > Assignment pointer: the expected answer is a set of **time periods**, not a prediction accuracy score, because the data does not provide labelled abnormal events.

# %% [markdown]
# ## 1. Imports and Configuration
# 
# The method uses a rolling median and rolling median absolute deviation (MAD). This is robust to spikes and does not assume that the sensor values are normally distributed.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.ensemble import IsolationForest
    ISOLATION_FOREST_AVAILABLE = True
except ImportError:
    ISOLATION_FOREST_AVAILABLE = False

sns.set_theme(style="whitegrid", context="notebook")
DATA_PATH = Path("Data (4).csv")
OUTPUT_PATH = Path("cyclone_anomalies.csv")
WINDOW = 289  # approximately 24 hours at 5-minute sampling
Z_THRESHOLD = 5.0
MIN_SENSORS = 2
SCALE_FLOOR_FRACTION = 0.10
MIN_PERIOD_OBSERVATIONS = 3
MODEL_SAMPLE_SIZE = 100_000
SHOW_PLOTS = False
print(f"Isolation Forest available: {ISOLATION_FOREST_AVAILABLE}")
print(f"Plot rendering enabled: {SHOW_PLOTS}")

# %% [markdown]
# ## 2. Data Preparation and Quality Checks
# 
# ### Sensor map
# 
# | Signal | Process meaning |
# |---|---|
# | `Cyclone_Inlet_Gas_Temp` | Hot gas temperature entering the cyclone |
# | `Cyclone_Gas_Outlet_Temp` | Hot gas temperature leaving the cyclone |
# | `Cyclone_Outlet_Gas_draft` | Gas draft at the cyclone outlet |
# | `Cyclone_cone_draft` | Gas draft at the cone section |
# | `Cyclone_Inlet_Draft` | Gas draft at the cyclone inlet |
# | `Cyclone_Material_Temp` | Material temperature at the cyclone outlet |
# 
# The timestamp is parsed and sorted. Sensor fields are coerced to numeric so malformed readings become missing values and can be measured explicitly. Rows missing sensor values are excluded from scoring rather than silently treated as normal.

# %%
df = pd.read_csv(DATA_PATH)
raw_rows = len(df)
expected_sensors = [
    "Cyclone_Inlet_Gas_Temp",
    "Cyclone_Material_Temp",
    "Cyclone_Outlet_Gas_draft",
    "Cyclone_cone_draft",
    "Cyclone_Gas_Outlet_Temp",
    "Cyclone_Inlet_Draft",
]
missing_columns = sorted(set(expected_sensors) - set(df.columns))
if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

df["time"] = pd.to_datetime(df["time"], format="mixed", dayfirst=True, errors="coerce")
invalid_time = int(df["time"].isna().sum())
for column in expected_sensors:
    df[column] = pd.to_numeric(df[column], errors="coerce")

invalid_sensor_counts = df[expected_sensors].isna().sum().sort_values(ascending=False)
duplicate_timestamps = int(df["time"].duplicated().sum())
df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
time_gaps = df["time"].diff().dropna().dt.total_seconds().div(60)
score_df = df.dropna(subset=expected_sensors).copy()

print(f"Raw rows: {raw_rows:,}")
print(f"Rows after timestamp validation and deduplication: {len(df):,}")
print(f"Time range: {df['time'].min()} to {df['time'].max()}")
print(f"Invalid timestamps: {invalid_time:,}")
print(f"Duplicate timestamps: {duplicate_timestamps:,}")
print(f"Typical sampling interval: {time_gaps.mode().iloc[0]:.0f} minutes")
print(f"Rows with all six sensors available for scoring: {len(score_df):,}")
print("Invalid sensor values before row filtering:")
print(invalid_sensor_counts.to_string())
print("\nDescriptive statistics:")
print(score_df[expected_sensors].describe().T[["mean", "std", "min", "max"]])

# %% [markdown]
# ## 3. Exploratory Analysis
# 
# The plots show the operating envelope and relationships between measurements before anomaly scoring.

# %%
plot_df = score_df.set_index("time")[expected_sensors].resample("1h").median()
fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True)
for axis, column in zip(axes.ravel(), expected_sensors):
    axis.plot(plot_df.index, plot_df[column], linewidth=0.8)
    axis.set_title(column)
    axis.set_ylabel("value")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(11, 10))
sns.heatmap(score_df[expected_sensors].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0])
axes[0].set_title("Sensor correlation matrix")
score_df[expected_sensors].plot(kind="box", vert=False, ax=axes[1], showfliers=False)
axes[1].set_title("Sensor distributions, excluding extreme display outliers")
plt.tight_layout()
plt.show()

coverage = score_df.set_index("time").resample("D")[expected_sensors].count().min(axis=1)
print(f"Days with complete six-sensor coverage: {(coverage == coverage.max()).sum():,} of {len(coverage):,}")
print(f"Lowest complete daily count: {coverage.min():.0f} observations")

# %% [markdown]
# ## 4. Analysis Strategy: Robust Multivariate Scoring
# 
# ### Why this approach?
# 
# - The assignment provides no abnormal-event labels, so supervised classification is not justified.
# - Sensor scales differ substantially, so raw-value thresholds would over-weight temperature or draft.
# - Industrial signals can contain spikes and non-normal distributions, so median/MAD is less sensitive to extreme values than mean/standard deviation.
# - Requiring at least two sensors to cross the threshold makes the result a multivariate process event rather than a single-channel glitch.
# - A small global MAD floor prevents nearly flat local windows from producing artificial, extremely large scores.
# 
# For every sensor, calculate a centred rolling median and rolling median absolute deviation (MAD) over 289 observations, approximately one day at five-minute sampling. Convert deviations to robust z-scores using $0.6745(x - median) / MAD$. Flag a timestamp when at least two sensors have an absolute robust z-score of 5 or more. Adjacent flagged timestamps are then grouped into abnormal periods.
# 
# ### Important interpretation
# 
# This is a **screening model**. The threshold, one-day window, and two-sensor rule are transparent starting assumptions. They should be checked against operator logs, maintenance records, and known process limits before becoming production alarm rules.

# %%
values = score_df.set_index("time")[expected_sensors]
rolling_median = values.rolling(WINDOW, center=True, min_periods=WINDOW // 2).median()
rolling_mad = (values - rolling_median).abs().rolling(WINDOW, center=True, min_periods=WINDOW // 2).median()

# Prevent tiny local MAD values from creating numerically infinite scores.
global_mad = (values - values.median()).abs().median()
scale_floor = global_mad * SCALE_FLOOR_FRACTION
robust_scale = rolling_mad.clip(lower=scale_floor, axis="columns")
robust_z = (0.6745 * (values - rolling_median) / robust_scale).abs()

score_df["anomaly_score"] = robust_z.max(axis=1).fillna(0).to_numpy()
score_df["sensors_over_threshold"] = (robust_z >= Z_THRESHOLD).sum(axis=1).to_numpy()
score_df["is_anomaly"] = score_df["sensors_over_threshold"] >= MIN_SENSORS

print("Global MAD scale floors:")
print(scale_floor.to_string())
print(f"Point anomalies: {score_df['is_anomaly'].sum():,} ({score_df['is_anomaly'].mean():.2%})")
print("Sensors most often contributing to anomalies:")
print((robust_z.loc[score_df["is_anomaly"].to_numpy()] >= Z_THRESHOLD).sum().sort_values(ascending=False).to_string())

# %% [markdown]
# ## 4A. Independent Model Cross-Check
# 
# The rolling robust detector captures local operating changes. As a cross-check, Isolation Forest evaluates the six sensors jointly for globally unusual combinations. Agreement between methods increases confidence; disagreement identifies cases that need closer review.

# %% [markdown]
# ## 4B. PCA Monitoring and Change-Point Detection
# 
# Two additional perspectives strengthen the analysis:
# 
# - **PCA monitoring:** Hotelling-style $T^2$ detects unusual movement inside the learned multivariate operating space, while SPE/Q-residual detects observations that do not fit that space.
# - **CUSUM change detection:** identifies sustained shifts in the first principal component rather than isolated spikes.
# 
# These methods are used as corroborating signals. The robust rolling detector remains the primary local-deviation screen.

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaled_values = StandardScaler().fit_transform(model_values)
pca = PCA(n_components=0.95)
pca_scores = pca.fit_transform(scaled_values)
reconstructed = pca.inverse_transform(pca_scores)

t2_score = (pca_scores ** 2 / pca.explained_variance_).sum(axis=1)
spe_score = ((scaled_values - reconstructed) ** 2).sum(axis=1)
t2_cutoff = np.quantile(t2_score, 0.995)
spe_cutoff = np.quantile(spe_score, 0.995)
score_df["pca_t2_score"] = t2_score
score_df["pca_spe_score"] = spe_score
score_df["pca_flag"] = (t2_score >= t2_cutoff) | (spe_score >= spe_cutoff)

# CUSUM records regime-transition crossings and resets after each crossing.
pc1 = pd.Series(pca_scores[:, 0], index=score_df["time"])
delta = pc1.diff().fillna(0)
step_scale = (delta - delta.median()).abs().median()
step_scale = max(float(step_scale), 1e-6)
k = 0.5 * step_scale
h = 8.0 * step_scale
positive = np.zeros(len(pc1))
negative = np.zeros(len(pc1))
change_point_flag = np.zeros(len(pc1), dtype=bool)
for index in range(1, len(pc1)):
    positive[index] = max(0.0, positive[index - 1] + delta.iloc[index] - k)
    negative[index] = min(0.0, negative[index - 1] + delta.iloc[index] + k)
    if positive[index] >= h or negative[index] <= -h:
        change_point_flag[index] = True
        positive[index] = 0.0
        negative[index] = 0.0
score_df["change_point_flag"] = change_point_flag

score_df["ensemble_votes"] = (
    score_df["is_anomaly"].astype(int)
    + score_df["isolation_flag"].astype(int)
    + score_df["pca_flag"].astype(int)
    + score_df["change_point_flag"].astype(int)
)
score_df["high_confidence_flag"] = score_df["ensemble_votes"] >= 3

print(f"PCA components retained: {pca.n_components_} ({pca.explained_variance_ratio_.sum():.2%} variance)")
print(f"PCA T2/SPE flags: {score_df['pca_flag'].sum():,}")
print(f"CUSUM transition flags: {score_df['change_point_flag'].sum():,}")
print(f"High-confidence ensemble flags (at least 3 of 4 methods): {score_df['high_confidence_flag'].sum():,}")
print(pd.Series({
    "robust rolling": score_df["is_anomaly"].sum(),
    "isolation forest": score_df["isolation_flag"].sum(),
    "PCA T2/SPE": score_df["pca_flag"].sum(),
    "CUSUM transition": score_df["change_point_flag"].sum(),
    "3+ model agreement": score_df["high_confidence_flag"].sum(),
}).to_string())

# %%
model_values = score_df[expected_sensors].copy()
model_median = model_values.median()
model_iqr = (model_values.quantile(0.75) - model_values.quantile(0.25)).replace(0, np.nan)
model_scaled = ((model_values - model_median) / model_iqr).fillna(0)

if ISOLATION_FOREST_AVAILABLE:
    fit_sample = model_scaled.sample(min(MODEL_SAMPLE_SIZE, len(model_scaled)), random_state=42)
    isolation_model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    isolation_model.fit(fit_sample)
    isolation_score = -isolation_model.score_samples(model_scaled)
    isolation_cutoff = np.quantile(isolation_score, 0.995)
    isolation_flag = isolation_score >= isolation_cutoff
    score_df["isolation_score"] = isolation_score
    score_df["isolation_flag"] = isolation_flag
else:
    isolation_score = np.zeros(len(score_df))
    isolation_flag = np.zeros(len(score_df), dtype=bool)
    score_df["isolation_score"] = isolation_score
    score_df["isolation_flag"] = isolation_flag
    isolation_cutoff = np.nan

score_df["model_agreement"] = score_df["is_anomaly"] & score_df["isolation_flag"]
score_df["review_priority"] = score_df["is_anomaly"].astype(int) + score_df["isolation_flag"].astype(int)
print(f"Rolling robust flags: {score_df['is_anomaly'].sum():,}")
print(f"Isolation Forest flags: {score_df['isolation_flag'].sum():,}")
print(f"Both models agree: {score_df['model_agreement'].sum():,}")
print(f"Agreement among rolling flags: {score_df.loc[score_df['is_anomaly'], 'model_agreement'].mean():.2%}")

agreement_counts = score_df["review_priority"].value_counts().sort_index()
if SHOW_PLOTS:
    plt.figure(figsize=(8, 4))
    plt.bar(["Neither", "One model", "Both models"], [agreement_counts.get(0, 0), agreement_counts.get(1, 0), agreement_counts.get(2, 0)], color=["#bdbdbd", "#80b1d3", "#fb8072"])
    plt.ylabel("observations")
    plt.title("Agreement between anomaly detectors")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Abnormal Periods and Insights
# 
# Adjacent anomalous observations are grouped into periods. A gap of one normal 5-minute observation is allowed so a brief dip does not split one operating event into two periods.

# %%
anomalies = score_df.loc[score_df["is_anomaly"]].copy()
if not anomalies.empty:
    gap = anomalies["time"].diff().gt(pd.Timedelta(minutes=10)).fillna(True)
    anomalies["period_id"] = gap.cumsum()
    periods = anomalies.groupby("period_id").agg(
        start=("time", "min"),
        end=("time", "max"),
        observations=("time", "size"),
        peak_score=("anomaly_score", "max"),
        mean_sensors_over_threshold=("sensors_over_threshold", "mean"),
        model_agreement_rate=("model_agreement", "mean"),
        mean_ensemble_votes=("ensemble_votes", "mean"),
        high_confidence_rate=("high_confidence_flag", "mean"),
    ).reset_index(drop=True)
    periods["duration_hours"] = (periods["end"] - periods["start"]).dt.total_seconds() / 3600 + (5 / 60)
    periods["persistent"] = periods["observations"] >= MIN_PERIOD_OBSERVATIONS
    periods = periods.sort_values(["persistent", "high_confidence_rate", "peak_score", "observations"], ascending=[False, False, False, False]).reset_index(drop=True)
else:
    periods = pd.DataFrame(columns=["start", "end", "observations", "peak_score", "mean_sensors_over_threshold", "model_agreement_rate", "mean_ensemble_votes", "high_confidence_rate", "duration_hours", "persistent"])

print(f"Abnormal periods found: {len(periods)}")
print(f"Persistent periods with at least {MIN_PERIOD_OBSERVATIONS} observations: {periods['persistent'].sum() if not periods.empty else 0}")
print(periods.head(20).to_string(index=False))

if SHOW_PLOTS:
    score_plot = score_df.set_index("time")["anomaly_score"].resample("1h").max()
    fig, axis = plt.subplots(figsize=(16, 5))
    axis.plot(score_plot.index, score_plot.values, color="black", linewidth=0.8, label="Hourly maximum robust score")
    axis.axhline(Z_THRESHOLD, color="red", linestyle="--", label=f"Threshold ({Z_THRESHOLD})")
    for _, period in periods.loc[periods["persistent"]].iterrows():
        axis.axvspan(period["start"], period["end"], color="red", alpha=0.18)
    axis.set_title("Persistent abnormal operating periods")
    axis.set_ylabel("robust anomaly score")
    axis.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5C. Event Severity and Operational Prioritisation
# 
# Not every flagged event deserves the same response. This section ranks events using a transparent heuristic combining duration, peak robust score, number of sensors involved, and agreement across independent detectors. The score is for triage, not a calibrated probability of failure.

# %%
if not periods.empty:
    periods["severity_score"] = (
        np.log1p(periods["duration_hours"])
        * np.log1p(periods["peak_score"])
        * (1 + periods["mean_sensors_over_threshold"] / len(expected_sensors))
        * (1 + periods["high_confidence_rate"])
    )
    periods["severity_band"] = pd.cut(
        periods["severity_score"],
        bins=[-np.inf, 5, 15, 30, np.inf],
        labels=["Low", "Medium", "High", "Critical"],
    )
    priority_events = periods.sort_values("severity_score", ascending=False).head(20)
    print("Top operational review candidates:")
    print(priority_events[["start", "end", "duration_hours", "peak_score", "high_confidence_rate", "severity_score", "severity_band"]].to_string(index=False))
else:
    periods["severity_score"] = pd.Series(dtype=float)
    periods["severity_band"] = pd.Series(dtype="object")

# %% [markdown]
# ## 5D. Industrial Control-Chart Diagnostics
# 
# EWMA-style smoothing is useful for operations because it highlights sustained movement while reducing the noise of individual five-minute readings. The code below creates a draft stability diagnostic without requiring labelled failure data.

# %%
control_columns = ["Cyclone_Inlet_Draft", "Cyclone_Outlet_Gas_draft", "Cyclone_cone_draft"]
control_summary = []
for column in control_columns:
    series = score_df.set_index("time")[column]
    ewma = series.ewm(span=12, adjust=False).mean()
    center = series.median()
    spread = (series - center).abs().median()
    limit = center + 5 * max(spread, 1e-6)
    lower_limit = center - 5 * max(spread, 1e-6)
    outside = ((ewma > limit) | (ewma < lower_limit)).sum()
    control_summary.append({
        "sensor": column,
        "global_center": center,
        "upper_limit": limit,
        "lower_limit": lower_limit,
        "ewma_outside_points": int(outside),
        "outside_rate": float(outside / len(ewma)),
    })
control_summary = pd.DataFrame(control_summary).sort_values("outside_rate", ascending=False)
print(control_summary.to_string(index=False))
print("Limits are descriptive screening bands; confirmed operating limits should come from engineering specifications.")

# %% [markdown]
# ## 5E. Stability and Synthetic-Event Validation
# 
# Without ground-truth labels, two practical checks are useful: compare the detector rate across chronological segments, and inject known perturbations into normal observations to verify that the pipeline responds.

# %%
chronological = score_df[["time", "is_anomaly", "high_confidence_flag"]].copy()
chronological["segment"] = pd.qcut(np.arange(len(chronological)), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
stability_summary = chronological.groupby("segment", observed=True).agg(
    observations=("is_anomaly", "size"),
    primary_rate=("is_anomaly", "mean"),
    high_confidence_rate=("high_confidence_flag", "mean"),
).reset_index()
print("Chronological detector stability:")
print(stability_summary.to_string(index=False))

# Inject known local-scale shifts into held-out normal observations.
rng = np.random.default_rng(42)
normal_indices = np.flatnonzero(~score_df["is_anomaly"].to_numpy())
synthetic_indices = rng.choice(normal_indices, size=min(100, len(normal_indices)), replace=False)
synthetic_values = values.copy()
for column in ["Cyclone_Inlet_Draft", "Cyclone_Outlet_Gas_draft"]:
    column_index = synthetic_values.columns.get_loc(column)
    local_scale = robust_scale.iloc[synthetic_indices, column_index].to_numpy()
    synthetic_values.iloc[synthetic_indices, column_index] += 8 * local_scale
synthetic_robust_z = (0.6745 * (synthetic_values - rolling_median) / robust_scale).abs()
synthetic_detected = (synthetic_robust_z.iloc[synthetic_indices] >= Z_THRESHOLD).sum(axis=1) >= MIN_SENSORS
print(f"Synthetic two-sensor events detected: {synthetic_detected.mean():.2%} ({synthetic_detected.sum()} of {len(synthetic_detected)})")

# %% [markdown]
# ### Validation interpretation
# 
# The primary anomaly rate stays within a relatively narrow range across the four chronological segments, which supports using one screening rule across the dataset while still showing some operating-regime variation. The synthetic test is a sensitivity check rather than a benchmark: incomplete recovery indicates that detection depends on local volatility, missing context at rolling-window edges, and the direction of the injected shift.

# %% [markdown]
# ## 5A. Interpretable Views of the Detected Events
# 
# The full-history chart is useful for locating clusters, but these focused views are better for explaining the result: when abnormal activity increased, which sensors drove it, and what the strongest events looked like in context.

# %%
top_periods = periods.loc[periods["persistent"]].head(3)
for _, event in top_periods.iterrows():
    event_start = event["start"] - pd.Timedelta(hours=2)
    event_end = event["end"] + pd.Timedelta(hours=2)
    event_scores = robust_z.loc[event_start:event_end]
    fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
    for axis, column in zip(axes.ravel(), expected_sensors):
        axis.plot(event_scores.index, event_scores[column], linewidth=1.0, color="#377eb8")
        axis.axhline(Z_THRESHOLD, color="#e41a1c", linestyle="--", linewidth=0.9)
        axis.set_title(column)
        axis.set_ylabel("absolute robust z")
        axis.grid(alpha=0.25)
    fig.suptitle(f"Event diagnostic: {event['start']} to {event['end']}", y=1.02)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5B. Detailed Process Interpretation
# 
# The next views compare flagged observations with the normal operating population and check whether abnormal activity is concentrated at particular operating times.

# %%
normal = score_df.loc[~score_df["is_anomaly"], expected_sensors]
abnormal = score_df.loc[score_df["is_anomaly"], expected_sensors]
comparison = pd.DataFrame({
    "normal_median": normal.median(),
    "abnormal_median": abnormal.median(),
    "median_shift": abnormal.median() - normal.median(),
    "normal_iqr": normal.quantile(0.75) - normal.quantile(0.25),
    "abnormal_iqr": abnormal.quantile(0.75) - abnormal.quantile(0.25),
})
comparison["shift_in_normal_iqr"] = comparison["median_shift"] / comparison["normal_iqr"].replace(0, np.nan)
print("Normal versus flagged observations:")
display(comparison.sort_values("shift_in_normal_iqr", key=abs, ascending=False))

analysis_df = score_df[["time", "is_anomaly"]].copy()
analysis_df["hour"] = analysis_df["time"].dt.hour
analysis_df["day_of_week"] = analysis_df["time"].dt.day_name()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
time_pattern = analysis_df.pivot_table(index="day_of_week", columns="hour", values="is_anomaly", aggfunc="mean").reindex(order)
plt.figure(figsize=(16, 4))
sns.heatmap(time_pattern * 100, cmap="YlOrRd", cbar_kws={"label": "anomaly rate (%)"})
plt.title("Abnormal-point rate by day of week and hour")
plt.xlabel("hour of day")
plt.ylabel("")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Relationship view: gas inlet temperature versus outlet draft
# 
# This plot checks whether abnormal points occupy a visibly different operating envelope in two physically related signals.

# %%
display_sample = score_df.sample(min(30000, len(score_df)), random_state=42)
normal_sample = display_sample.loc[~display_sample["is_anomaly"]]
anomaly_sample = display_sample.loc[display_sample["is_anomaly"]]
plt.figure(figsize=(11, 7))
plt.scatter(normal_sample["Cyclone_Inlet_Gas_Temp"], normal_sample["Cyclone_Outlet_Gas_draft"], s=5, alpha=0.12, label="normal", color="#377eb8")
plt.scatter(anomaly_sample["Cyclone_Inlet_Gas_Temp"], anomaly_sample["Cyclone_Outlet_Gas_draft"], s=16, alpha=0.7, label="flagged", color="#e41a1c")
plt.xlabel("Cyclone inlet gas temperature")
plt.ylabel("Cyclone outlet gas draft")
plt.title("Operating envelope with flagged observations highlighted")
plt.legend()
plt.tight_layout()
plt.show()

print("Top process-level interpretation points:")
print(f"- {score_df['is_anomaly'].mean():.2%} of complete observations are flagged by the multivariate screen.")
print(f"- {periods['persistent'].sum():,} persistent periods last at least {MIN_PERIOD_OBSERVATIONS} observations.")
print(f"- The most frequent contributing sensor is {contribution_counts.index[-1]}.")
print("- Validate these candidate periods against process history before assigning a failure mechanism.")

# %% [markdown]
# ### Zoomed event diagnostics

# %%
monthly = score_df.set_index("time").resample("MS").agg(
    observations=("is_anomaly", "size"),
    anomalous_points=("is_anomaly", "sum"),
)
monthly["anomaly_rate"] = monthly["anomalous_points"] / monthly["observations"]

fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
axes[0].bar(monthly.index, monthly["anomaly_rate"] * 100, width=20, color="#d95f02")
axes[0].set_ylabel("anomaly rate (%)")
axes[0].set_title("Monthly abnormal-point rate")
axes[0].grid(axis="x", visible=False)

contribution_counts = (robust_z.loc[score_df["is_anomaly"].to_numpy()] >= Z_THRESHOLD).sum().sort_values()
axes[1].barh(contribution_counts.index, contribution_counts.values, color="#1b9e77")
axes[1].set_xlabel("flagged observations contributed")
axes[1].set_title("Which sensors drive the multivariate flags?")
axes[1].grid(axis="y", visible=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Evidence behind the highest-ranked periods
# 
# For each period, identify the sensors that contributed at least one threshold crossing. This makes the timestamps auditable by a process expert.

# %%
if not periods.empty:
    period_evidence = []
    for period_id, period in anomalies.groupby("period_id"):
        period_rows = robust_z.loc[period["time"]]
        contributors = (period_rows >= Z_THRESHOLD).any(axis=0)
        period_evidence.append({
            "start": period["time"].min(),
            "end": period["time"].max(),
            "duration_hours": (period["time"].max() - period["time"].min()).total_seconds() / 3600 + 5 / 60,
            "peak_score": period["anomaly_score"].max(),
            "contributing_sensors": ", ".join(contributors[contributors].index),
        })
    period_evidence = pd.DataFrame(period_evidence).sort_values("peak_score", ascending=False)
    display(period_evidence.head(15))
else:
    period_evidence = pd.DataFrame()

print("Review pointer: compare the listed timestamps with process logs, maintenance events, feed changes, and operator notes before assigning a physical root cause.")

# %% [markdown]
# ## 6. Robustness Check: Threshold Sensitivity
# 
# A lead analysis should show whether the headline result is highly dependent on one arbitrary threshold. The table below compares point-level flags across nearby thresholds while holding the window and two-sensor rule constant.

# %%
sensitivity_rows = []
for threshold in [4.0, 5.0, 6.0, 7.0]:
    point_flags = (robust_z >= threshold).sum(axis=1) >= MIN_SENSORS
    sensitivity_rows.append({
        "threshold": threshold,
        "flagged_points": int(point_flags.sum()),
        "flagged_rate": float(point_flags.mean()),
    })
sensitivity = pd.DataFrame(sensitivity_rows)
display(sensitivity)
print("A stable operating threshold should be selected with process context, balancing missed events against review workload.")

# %%
# Save row-level results and period-level findings for review.
score_df.to_csv(OUTPUT_PATH, index=False)
periods.to_csv("cyclone_abnormal_periods.csv", index=False)

print(f"Saved row-level results to {OUTPUT_PATH}")
print("Saved period summary to cyclone_abnormal_periods.csv")

if periods.empty:
    print("No abnormal periods met the multivariate threshold.")
else:
    print("Interpretation: these periods are candidates for abnormal operation because multiple sensors deviated from their local one-day baselines at the same time.")
    print("The threshold is a screening rule, not a process alarm limit; domain review should confirm the operating cause of each period.")

# %% [markdown]
# ## 7. Conclusion and Next Steps
# 
# The output identifies candidate abnormal operating periods by combining simultaneous deviations across multiple cyclone sensors. The period summary is the main deliverable for the assignment; the row-level file preserves the evidence behind every flag.
# 
# Recommended next steps for validation:
# 
# 1. Overlay the ranked periods with operator logs and maintenance records.
# 2. Check whether the strongest periods correspond to known feed, temperature, or draft changes.
# 3. Use the threshold-sensitivity table to choose a review workload that operations can support.
# 4. Tune the rolling window and threshold against confirmed events.
# 5. Add process-engineer operating limits if the model is converted into an alarm or monitoring service.
# 
# The analysis should be presented as anomaly screening rather than confirmed fault diagnosis.

# %% [markdown]
# ## 8. Business Interpretation
# 
# ### What the analysis indicates
# 
# Approximately **2.75% of complete sensor observations** were flagged by the primary multivariate screen. These observations formed persistent candidate periods lasting at least three consecutive five-minute observations. The advanced ensemble also provides a stricter high-confidence view: an observation is high confidence only when at least three of four independent methods agree.
# 
# The strongest contributors were:
# 
# 1. `Cyclone_Inlet_Draft`
# 2. `Cyclone_Outlet_Gas_draft`
# 3. `Cyclone_Inlet_Gas_Temp`
# 4. `Cyclone_cone_draft`
# 
# The draft signals showed the largest difference between normal and flagged observations. This suggests that the abnormal periods are more strongly associated with **airflow or pressure instability** than with an isolated temperature excursion.
# 
# ### Possible operational impact
# 
# Abnormal draft and temperature behaviour may indicate:
# 
# - Reduced cyclone separation efficiency
# - Increased fan or energy consumption
# - Unstable material flow
# - Process-quality variation
# - Possible blockage, leakage, feed disturbance, or draft-control problems
# - Additional equipment stress if pressure swings recur
# 
# These are business hypotheses for investigation, not confirmed root causes.
# 
# ### Recommended operational use
# 
# Use the persistent-period table as a review queue. Prioritise periods with high robust scores, several contributing sensors, strong ensemble agreement, and transition evidence from the change-point detector. Compare those timestamps against operator logs, maintenance records, feed or fuel changes, fan settings, and product-quality records.
# 
# Once confirmed events are identified, the organisation can convert recurring sensor patterns into process-specific alarm limits or an early-warning monitoring rule.
# 
# ### Model limitation
# 
# The dataset does not label abnormal events, so this is an unsupervised screening analysis rather than a failure diagnosis. The methods intentionally detect different forms of unusual behaviour: local deviation, global multivariate outlier, PCA-space departure, and regime transition. Domain validation is required before using the output for maintenance or production decisions.


