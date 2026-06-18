"""
hPrime Assessment Data — Exploratory Data Analysis
CSCI323 Modern Artificial Intelligence — Spring 2026
University of Wollongong in Dubai
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE  = "/mnt/user-data/outputs/hprime_clean.csv"
OUT_DIR    = "/mnt/user-data/outputs/eda"
os.makedirs(OUT_DIR, exist_ok=True)

BRAND_BLUE   = "#1F4E79"
BRAND_MID    = "#2E75B6"
BRAND_LIGHT  = "#BDD7EE"
FAIL_RED     = "#C00000"
PASS_GREEN   = "#538135"
WARN_ORANGE  = "#E97C30"
BG           = "#F7F9FC"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    "#CCCCCC",
    "axes.labelcolor":   "#222222",
    "axes.labelsize":    11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   BRAND_BLUE,
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "grid.color":        "#E0E0E0",
    "grid.linewidth":    0.7,
    "font.family":       "DejaVu Sans",
    "legend.fontsize":   9,
    "legend.framealpha": 0.8,
})

CBD_SCORE_COLS     = ["score_record_keeping","score_history_examination",
                       "score_management_plan","score_clinical_judgement"]
MINICEX_SCORE_COLS = ["score_history","score_physical_exam","score_communication",
                       "score_clinical_judgement.1","score_professionalism",
                       "score_organisation","score_overall_care"]

CBD_LABELS     = ["Record\nKeeping","History &\nExam","Management\nPlan","Clinical\nJudgement"]
MINICEX_LABELS = ["History","Physical\nExam","Communication","Clinical\nJudgement",
                  "Professional-\nism","Organisation","Overall\nCare"]

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {name}")
    return path

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE, parse_dates=["assessment_date"])
df["month_year"] = df["assessment_date"].dt.to_period("M")

cbd  = df[df["form_type"] == "Case Based Discussion"].copy()
mcex = df[df["form_type"] == "Mini-CEX"].copy()

print(f"Loaded {len(df)} records | CBD: {len(cbd)} | MiniCEX: {len(mcex)}")
print(f"Unique candidates: {df['candidate_id'].nunique()} | Fail rate: {df['meets_standard'].eq(0).mean():.1%}\n")

saved_files = []

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Dataset Overview...")
fig = plt.figure(figsize=(16, 10), facecolor=BG)
fig.suptitle("hPrime Assessment Data — Dataset Overview",
             fontsize=16, fontweight="bold", color=BRAND_BLUE, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# 1a — Assessment type split (donut)
ax = fig.add_subplot(gs[0, 0])
vals   = [len(mcex), len(cbd)]
labels = [f"Mini-CEX\n{len(mcex):,}", f"CBD\n{len(cbd):,}"]
colors = [BRAND_MID, WARN_ORANGE]
wedges, texts, autotexts = ax.pie(
    vals, labels=labels, colors=colors, autopct="%1.1f%%",
    startangle=90, wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
    textprops={"fontsize": 10}
)
for at in autotexts: at.set_fontsize(9)
ax.set_title("Assessment Type Split")

# 1b — Pass / Fail (donut)
ax = fig.add_subplot(gs[0, 1])
p, f = df["meets_standard"].eq(1).sum(), df["meets_standard"].eq(0).sum()
ax.pie([p, f], labels=[f"Pass\n{p:,}", f"Fail\n{f}"],
       colors=[PASS_GREEN, FAIL_RED], autopct="%1.1f%%", startangle=90,
       wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
       textprops={"fontsize": 10})
ax.set_title("Pass / Fail Distribution")

# 1c — Assessments per candidate histogram
ax = fig.add_subplot(gs[0, 2])
counts = df.groupby("candidate_id").size()
ax.hist(counts, bins=15, color=BRAND_MID, edgecolor="white", linewidth=0.8)
ax.axvline(counts.mean(), color=FAIL_RED, lw=1.8, ls="--",
           label=f"Mean = {counts.mean():.1f}")
ax.set_xlabel("Number of Assessments")
ax.set_ylabel("Number of Candidates")
ax.set_title("Assessments per Candidate")
ax.legend(); ax.grid(axis="y")

# 1d — Monthly volume
ax = fig.add_subplot(gs[1, :2])
monthly = df.groupby("month_year").size()
monthly.index = monthly.index.to_timestamp()
ax.bar(monthly.index, monthly.values, color=BRAND_MID,
       edgecolor="white", linewidth=0.5, width=20)
ax.plot(monthly.index, monthly.rolling(3, min_periods=1).mean(),
        color=FAIL_RED, lw=2, label="3-month rolling avg")
ax.set_xlabel("Month"); ax.set_ylabel("Number of Assessments")
ax.set_title("Assessment Volume Over Time (Jun 2024 – May 2026)")
ax.legend(); ax.grid(axis="y")

# 1e — Setting distribution
ax = fig.add_subplot(gs[1, 2])
setting_counts = df["setting"].value_counts().head(4)
bars = ax.barh(setting_counts.index, setting_counts.values,
               color=[BRAND_BLUE, BRAND_MID, BRAND_LIGHT, WARN_ORANGE],
               edgecolor="white")
for bar, v in zip(bars, setting_counts.values):
    ax.text(v + 5, bar.get_y() + bar.get_height()/2,
            str(v), va="center", fontsize=9)
ax.set_xlabel("Count"); ax.set_title("Assessment Setting")
ax.grid(axis="x")

saved_files.append(save(fig, "fig1_dataset_overview.png"))

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Score Distributions
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Score Distributions...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
fig.suptitle("Score Distributions — All Assessments",
             fontsize=16, fontweight="bold", color=BRAND_BLUE)

# 2a — avg_score by form type
ax = axes[0]
for ft, color, lbl in [(cbd, WARN_ORANGE, "CBD"), (mcex, BRAND_MID, "Mini-CEX")]:
    ax.hist(ft["avg_score"], bins=20, alpha=0.65, color=color,
            edgecolor="white", label=lbl)
ax.axvline(df["avg_score"].mean(), color=FAIL_RED, lw=1.8, ls="--",
           label=f"Overall mean = {df['avg_score'].mean():.2f}")
ax.set_xlabel("Average Score (1–4 scale)"); ax.set_ylabel("Frequency")
ax.set_title("Average Score Distribution"); ax.legend(); ax.grid(axis="y")

# 2b — avg_score: pass vs fail
ax = axes[1]
pass_scores = df[df["meets_standard"] == 1]["avg_score"]
fail_scores = df[df["meets_standard"] == 0]["avg_score"]
ax.hist(pass_scores, bins=20, alpha=0.65, color=PASS_GREEN,
        edgecolor="white", label=f"Pass (n={len(pass_scores)})")
ax.hist(fail_scores, bins=12, alpha=0.85, color=FAIL_RED,
        edgecolor="white", label=f"Fail (n={len(fail_scores)})")
ax.set_xlabel("Average Score"); ax.set_ylabel("Frequency")
ax.set_title("Avg Score: Pass vs Fail"); ax.legend(); ax.grid(axis="y")

# 2c — Score std (consistency)
ax = axes[2]
ax.hist(df[df["meets_standard"]==1]["score_std"], bins=20, alpha=0.65,
        color=PASS_GREEN, edgecolor="white", label="Pass")
ax.hist(df[df["meets_standard"]==0]["score_std"], bins=12, alpha=0.85,
        color=FAIL_RED, edgecolor="white", label="Fail")
ax.set_xlabel("Score Std Dev (consistency across dimensions)")
ax.set_ylabel("Frequency")
ax.set_title("Score Consistency: Pass vs Fail"); ax.legend(); ax.grid(axis="y")

plt.tight_layout()
saved_files.append(save(fig, "fig2_score_distributions.png"))

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Per-Dimension Score Profiles
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Per-Dimension Score Profiles...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
fig.suptitle("Per-Dimension Score Profiles",
             fontsize=16, fontweight="bold", color=BRAND_BLUE)

for ax, subset, cols, labels, title, color in [
    (axes[0], cbd,  CBD_SCORE_COLS,     CBD_LABELS,     "CBD (4 Dimensions)", WARN_ORANGE),
    (axes[1], mcex, MINICEX_SCORE_COLS, MINICEX_LABELS, "Mini-CEX (7 Dimensions)", BRAND_MID),
]:
    pass_means = subset[subset["meets_standard"]==1][cols].mean()
    fail_means = subset[subset["meets_standard"]==0][cols].mean()
    x = np.arange(len(cols))
    w = 0.35
    ax.bar(x - w/2, pass_means, w, label="Pass", color=PASS_GREEN,
           edgecolor="white", linewidth=0.8)
    ax.bar(x + w/2, fail_means, w, label="Fail", color=FAIL_RED,
           edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 4.5); ax.set_ylabel("Mean Score (1–4)")
    ax.set_title(title); ax.legend(); ax.grid(axis="y")
    ax.axhline(2.5, color="#999999", lw=1, ls=":")

plt.tight_layout()
saved_files.append(save(fig, "fig3_dimension_profiles.png"))

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Failure Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: Failure Analysis...")
fig = plt.figure(figsize=(16, 10), facecolor=BG)
fig.suptitle("Failure Analysis — Who Fails, When, and Why",
             fontsize=16, fontweight="bold", color=BRAND_BLUE, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# 4a — Failures by form type
ax = fig.add_subplot(gs[0, 0])
fail_by_type = df[df["meets_standard"]==0]["form_type"].value_counts()
bars = ax.bar(fail_by_type.index, fail_by_type.values,
              color=[WARN_ORANGE, BRAND_MID][:len(fail_by_type)],
              edgecolor="white")
for bar, v in zip(bars, fail_by_type.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
            str(v), ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Number of Failures"); ax.set_title("Failures by Form Type")
ax.grid(axis="y")

# 4b — Fail rate by assessor seniority
ax = fig.add_subplot(gs[0, 1])
seniority_fails = df.groupby("assessor_seniority")["meets_standard"].agg(
    lambda x: (x==0).sum() / len(x) * 100
).sort_values(ascending=False)
colors_s = [BRAND_BLUE, BRAND_MID, BRAND_LIGHT]
bars = ax.bar(seniority_fails.index, seniority_fails.values,
              color=colors_s[:len(seniority_fails)], edgecolor="white")
for bar, v in zip(bars, seniority_fails.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
            f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Fail Rate (%)"); ax.set_title("Fail Rate by Assessor Seniority")
ax.grid(axis="y")

# 4c — Which assessment number do failures cluster at?
ax = fig.add_subplot(gs[0, 2])
fail_by_num = df[df["meets_standard"]==0]["assessment_number"].value_counts().sort_index()
ax.bar(fail_by_num.index, fail_by_num.values, color=FAIL_RED, edgecolor="white")
ax.set_xlabel("Assessment Number (in candidate history)")
ax.set_ylabel("Number of Failures")
ax.set_title("When Do Failures Occur?"); ax.grid(axis="y")

# 4d — Candidates ranked by cumulative fail count
ax = fig.add_subplot(gs[1, :2])
cand_fails = df.groupby("candidate_id")["meets_standard"].apply(
    lambda x: (x==0).sum()
).sort_values(ascending=False).head(25)
colors_bar = [FAIL_RED if v >= 3 else WARN_ORANGE if v == 2 else BRAND_MID
              for v in cand_fails.values]
ax.bar(cand_fails.index.astype(str), cand_fails.values,
       color=colors_bar, edgecolor="white")
ax.set_xlabel("Candidate ID"); ax.set_ylabel("Number of Failed Assessments")
ax.set_title("Top 25 Candidates by Failure Count")
ax.axhline(2, color="#999999", lw=1, ls="--", label="2+ fails threshold")
ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(axis="y")

# 4e — rolling_fail_rate_3 at time of failure vs pass
ax = fig.add_subplot(gs[1, 2])
ax.boxplot(
    [df[df["meets_standard"]==1]["rolling_fail_rate_3"].dropna(),
     df[df["meets_standard"]==0]["rolling_fail_rate_3"].dropna()],
    labels=["Pass", "Fail"],
    patch_artist=True,
    boxprops=dict(facecolor=BRAND_LIGHT, color=BRAND_BLUE),
    medianprops=dict(color=FAIL_RED, linewidth=2),
    flierprops=dict(marker="o", markerfacecolor=BRAND_MID, markersize=4)
)
ax.set_ylabel("Rolling 3-Assessment Fail Rate")
ax.set_title("Rolling Fail Rate at Assessment Time"); ax.grid(axis="y")

saved_files.append(save(fig, "fig4_failure_analysis.png"))

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Temporal Trends
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 5: Temporal Trends...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
fig.suptitle("Temporal Trends — Score Progression Over Assessment History",
             fontsize=16, fontweight="bold", color=BRAND_BLUE)

# 5a — avg_score progression by assessment number
ax = axes[0]
avg_by_num = df.groupby("assessment_number")["avg_score"].agg(["mean","sem"])
ax.plot(avg_by_num.index, avg_by_num["mean"], color=BRAND_BLUE, lw=2.2,
        marker="o", markersize=5)
ax.fill_between(avg_by_num.index,
                avg_by_num["mean"] - avg_by_num["sem"],
                avg_by_num["mean"] + avg_by_num["sem"],
                alpha=0.2, color=BRAND_MID)
ax.set_xlabel("Assessment Number"); ax.set_ylabel("Mean Avg Score")
ax.set_title("Score Progression Over Time"); ax.grid()

# 5b — Days between assessments distribution
ax = axes[1]
ax.hist(df["days_since_last"].dropna(), bins=30, color=BRAND_MID,
        edgecolor="white")
ax.axvline(df["days_since_last"].median(), color=FAIL_RED, lw=1.8, ls="--",
           label=f"Median = {df['days_since_last'].median():.0f} days")
ax.set_xlabel("Days Since Last Assessment"); ax.set_ylabel("Frequency")
ax.set_title("Gap Between Assessments"); ax.legend(); ax.grid(axis="y")

# 5c — Improvement: first 3 vs last 3 assessments per candidate
ax = axes[2]
first3 = df.groupby("candidate_id").apply(
    lambda x: x.nsmallest(3,"assessment_number")["avg_score"].mean()
)
last3  = df.groupby("candidate_id").apply(
    lambda x: x.nlargest(3,"assessment_number")["avg_score"].mean()
)
diff = (last3 - first3).dropna()
colors_diff = [PASS_GREEN if v >= 0 else FAIL_RED for v in diff.values]
ax.bar(range(len(diff)), sorted(diff.values), color=sorted(
    colors_diff, key=lambda c: diff.values[colors_diff.index(c)]), edgecolor="white")
ax.axhline(0, color="#333333", lw=1)
ax.set_xlabel("Candidates (sorted by improvement)")
ax.set_ylabel("Avg Score: Last 3 minus First 3")
ax.set_title("Score Improvement Across Training Period"); ax.grid(axis="y")

plt.tight_layout()
saved_files.append(save(fig, "fig5_temporal_trends.png"))

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Feature Correlations
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 6: Feature Correlations...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
fig.suptitle("Feature Correlations with meets_standard",
             fontsize=16, fontweight="bold", color=BRAND_BLUE)

# 6a — Correlation heatmap (numeric features vs meets_standard)
numeric_features = [
    "avg_score","score_std","any_score_1","assessment_number",
    "days_since_last","rolling_avg_3","rolling_fail_rate_3",
    "cumulative_fail_count","patient_age_years","feedback_length",
    "discussion_time_mins","prep_time_mins","num_competencies"
]
corr_cols = [c for c in numeric_features if c in df.columns] + ["meets_standard"]
corr_mat  = df[corr_cols].corr()
mask = np.zeros_like(corr_mat, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(corr_mat, ax=axes[0], annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 7},
            mask=mask, cbar_kws={"shrink": 0.8})
axes[0].set_title("Feature Correlation Matrix")
axes[0].tick_params(axis="x", rotation=45, labelsize=8)
axes[0].tick_params(axis="y", rotation=0, labelsize=8)

# 6b — Top correlations with meets_standard (bar chart)
ax = axes[1]
target_corr = corr_mat["meets_standard"].drop("meets_standard").sort_values()
colors_corr = [FAIL_RED if v < 0 else PASS_GREEN for v in target_corr.values]
ax.barh(target_corr.index, target_corr.values, color=colors_corr, edgecolor="white")
ax.axvline(0, color="#333333", lw=1)
ax.set_xlabel("Pearson Correlation with meets_standard")
ax.set_title("Feature Importance (Correlation)")
ax.grid(axis="x")
for i, (idx, v) in enumerate(target_corr.items()):
    ax.text(v + (0.005 if v >= 0 else -0.005), i,
            f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)

plt.tight_layout()
saved_files.append(save(fig, "fig6_correlations.png"))

# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — At-Risk Candidate Profiles
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 7: At-Risk Candidate Profiles...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
fig.suptitle("At-Risk Candidate Profiles — Tracking High-Risk Trainees",
             fontsize=16, fontweight="bold", color=BRAND_BLUE)

# Identify top 5 most at-risk candidates (most failures)
at_risk_ids = (
    df.groupby("candidate_id")["meets_standard"]
    .apply(lambda x: (x==0).sum())
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)

# 7a — Rolling avg score over time for at-risk vs typical candidates
ax = axes[0, 0]
typical_ids = (
    df.groupby("candidate_id")["meets_standard"]
    .apply(lambda x: (x==0).sum())
    .eq(0)
    .where(lambda x: x)
    .dropna()
    .index[:5]
    .tolist()
)
for cid in at_risk_ids:
    cand = df[df["candidate_id"] == cid].sort_values("assessment_number")
    ax.plot(cand["assessment_number"], cand["rolling_avg_3"],
            color=FAIL_RED, alpha=0.7, lw=1.5)
for cid in typical_ids:
    cand = df[df["candidate_id"] == cid].sort_values("assessment_number")
    ax.plot(cand["assessment_number"], cand["rolling_avg_3"],
            color=PASS_GREEN, alpha=0.5, lw=1.2)
from matplotlib.lines import Line2D
legend_els = [Line2D([0],[0],color=FAIL_RED,lw=2,label="At-risk candidates"),
              Line2D([0],[0],color=PASS_GREEN,lw=2,label="Typical candidates")]
ax.legend(handles=legend_els); ax.grid()
ax.set_xlabel("Assessment Number"); ax.set_ylabel("Rolling Avg Score (3-window)")
ax.set_title("Score Trajectory: At-Risk vs Typical"); ax.axhline(2.5, ls=":", color="#999")

# 7b — Avg score at each assessment for top at-risk candidate
ax = axes[0, 1]
top_cand = df[df["candidate_id"] == at_risk_ids[0]].sort_values("assessment_number")
colors_pts = [FAIL_RED if m==0 else PASS_GREEN for m in top_cand["meets_standard"]]
ax.bar(top_cand["assessment_number"], top_cand["avg_score"],
       color=colors_pts, edgecolor="white")
ax.plot(top_cand["assessment_number"], top_cand["rolling_avg_3"],
        color=BRAND_BLUE, lw=2, ls="--", label="Rolling avg (3)")
ax.axhline(2.5, color="#999", ls=":", lw=1, label="2.5 threshold")
ax.set_xlabel("Assessment Number"); ax.set_ylabel("Avg Score")
ax.set_title(f"Candidate {int(at_risk_ids[0])} — Detailed Timeline")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=FAIL_RED, label="Failed assessment"),
    Patch(color=PASS_GREEN, label="Passed assessment"),
    Line2D([0],[0],color=BRAND_BLUE,ls="--",lw=2,label="Rolling avg"),
])
ax.grid(axis="y")

# 7c — Cumulative fail count by assessment number (at-risk candidates)
ax = axes[1, 0]
for cid in at_risk_ids:
    cand = df[df["candidate_id"] == cid].sort_values("assessment_number")
    ax.plot(cand["assessment_number"], cand["cumulative_fail_count"],
            marker="o", markersize=4, lw=1.8, label=f"ID {int(cid)}")
ax.set_xlabel("Assessment Number"); ax.set_ylabel("Cumulative Fail Count")
ax.set_title("Cumulative Failures — Top 5 At-Risk Candidates")
ax.legend(fontsize=8); ax.grid()

# 7d — Scatter: avg_score vs rolling_fail_rate_3, coloured by outcome
ax = axes[1, 1]
pass_df = df[df["meets_standard"]==1]
fail_df = df[df["meets_standard"]==0]
ax.scatter(pass_df["avg_score"], pass_df["rolling_fail_rate_3"],
           alpha=0.2, s=15, color=PASS_GREEN, label="Pass")
ax.scatter(fail_df["avg_score"], fail_df["rolling_fail_rate_3"],
           alpha=0.8, s=30, color=FAIL_RED, label="Fail", zorder=5)
ax.set_xlabel("Average Score"); ax.set_ylabel("Rolling Fail Rate (last 3 assessments)")
ax.set_title("Score vs Rolling Fail Rate")
ax.legend(); ax.grid()

plt.tight_layout()
saved_files.append(save(fig, "fig7_at_risk_profiles.png"))

# ══════════════════════════════════════════════════════════════════════════════
# Print EDA key findings
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("KEY EDA FINDINGS FOR REPORT")
print("="*60)
print(f"1. Dataset: {len(df)} assessments, {df['candidate_id'].nunique()} candidates, Jun 2024–May 2026")
print(f"2. Class imbalance: {df['meets_standard'].eq(0).mean():.1%} fail rate → SMOTE needed")
print(f"3. At-risk: {df.groupby('candidate_id')['meets_standard'].apply(lambda x:(x==0).sum()).gt(0).sum()} of {df['candidate_id'].nunique()} candidates failed at least once")
print(f"4. Avg score: pass={df[df['meets_standard']==1]['avg_score'].mean():.2f}, fail={df[df['meets_standard']==0]['avg_score'].mean():.2f}")
print(f"5. Strongest predictor (correlation): {df[[c for c in ['avg_score','rolling_avg_3','rolling_fail_rate_3','any_score_1','cumulative_fail_count'] if c in df.columns]+['meets_standard']].corr()['meets_standard'].drop('meets_standard').abs().idxmax()}")
print(f"6. Assessment frequency: median {df['days_since_last'].median():.0f} days between assessments")
print(f"7. CBD fail rate: {cbd['meets_standard'].eq(0).mean():.1%} | MiniCEX fail rate: {mcex['meets_standard'].eq(0).mean():.1%}")
print(f"8. Candidates with 3+ failures: {df.groupby('candidate_id')['meets_standard'].apply(lambda x:(x==0).sum()).ge(3).sum()}")
print("\nAll figures saved to:", OUT_DIR)
print("Files:", [os.path.basename(f) for f in saved_files])
