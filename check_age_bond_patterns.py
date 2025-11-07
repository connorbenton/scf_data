# check_age_bond_patterns.py
import numpy as np
import pandas as pd
from pathlib import Path

CONCAT_PATH = "p22i6_with_concats.parquet"

# ------------------ load ------------------
if not Path(CONCAT_PATH).exists():
    raise FileNotFoundError(f"Missing {CONCAT_PATH}. Run your concatenation script first.")

df = pd.read_parquet(CONCAT_PATH)
df.columns = [c.strip().lower() for c in df.columns]

# ------------------ weights ------------------
if "x42001" not in df.columns:
    raise KeyError("x42001 not found in dataframe (needed to build weights).")
df["x42001"] = pd.to_numeric(df["x42001"], errors="coerce").fillna(0)
df = df[df["x42001"] > 0].copy()
df["wgt"] = df["x42001"] / 5.0

# ------------------ required vars ------------------
bond_like = ["bond", "tfbmutf", "gbmutf", "obmutf"]
for c in bond_like:
    if c not in df.columns:
        raise KeyError(f"Missing required column: {c}")
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Find an age variable (common SCF/household survey names included)
age_candidates = ["age", "age_head", "ageh", "agehh", "respondent_age", "x14"]
age_col = next((a for a in age_candidates if a in df.columns), None)
if age_col is None:
    raise KeyError(f"Could not find an age column. Tried: {age_candidates}")

df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

# Keep finite entries for age and weight
df = df[np.isfinite(df["wgt"]) & (df["wgt"] > 0) & np.isfinite(df[age_col])].copy()

# ------------------ helpers ------------------
def weighted_median(vals, wts):
    vals = np.asarray(vals, dtype=float)
    wts = np.asarray(wts, dtype=float)
    m = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    if not np.any(m):
        return np.nan
    vals, wts = vals[m], wts[m]
    order = np.argsort(vals)
    vals, wts = vals[order], wts[order]
    cw = np.cumsum(wts)
    cutoff = 0.5 * wts.sum()
    return vals[np.searchsorted(cw, cutoff)]

def wmean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return np.average(x[m], weights=w[m])

def wls_slope(y, x, w):
    """
    Return slope from weighted least squares of y ~ a + b*x.
    Uses closed-form formulas (no external deps).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(y) & np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    y, x, w = y[m], x[m], w[m]
    W = w.sum()
    xbar = np.sum(w * x) / W
    ybar = np.sum(w * y) / W
    cov_w = np.sum(w * (x - xbar) * (y - ybar))
    var_w = np.sum(w * (x - xbar) ** 2)
    if var_w <= 0:
        return np.nan
    return cov_w / var_w

# ------------------ construct totals & bins ------------------
df["bond_total"] = df["bond"].fillna(0) + df["tfbmutf"].fillna(0) + df["gbmutf"].fillna(0) + df["obmutf"].fillna(0)

# Age bands typical in portfolio guidance
age_bins = [-np.inf, 34, 44, 54, 64, 74, np.inf]
age_labels = ["<35", "35–44", "45–54", "55–64", "65–74", "75+"]
df["AGE_BIN"] = pd.cut(df[age_col], bins=age_bins, labels=age_labels, right=True)

# ------------------ table by age band ------------------
def summarize_by_age(g):
    out = {}
    w = pd.to_numeric(g["wgt"], errors="coerce")
    for col in bond_like + ["bond_total"]:
        v = pd.to_numeric(g[col], errors="coerce")
        # ownership indicator and rate
        own = (v > 0) & np.isfinite(v)
        out[f"{col}_ownrate"] = wmean(own.astype(float), w)  # weighted % with >0
        # unconditional stats
        out[f"{col}_mean"] = wmean(v, w)
        out[f"{col}_median"] = weighted_median(v, w)
        # owners only stats
        if own.any():
            out[f"{col}_mean_owners"] = wmean(v[own], w[own])
            out[f"{col}_median_owners"] = weighted_median(v[own], w[own])
        else:
            out[f"{col}_mean_owners"] = np.nan
            out[f"{col}_median_owners"] = np.nan
    return pd.Series(out)

age_grp = (
    df.dropna(subset=["AGE_BIN"])
      .groupby("AGE_BIN", observed=True)
      .apply(summarize_by_age)
      .reindex(age_labels)
)

# ------------------ weighted trend checks (continuous age) ------------------
trend_rows = []
for col in bond_like + ["bond_total"]:
    y = np.log1p(pd.to_numeric(df[col], errors="coerce"))  # dampen right tail
    x = pd.to_numeric(df[age_col], errors="coerce")
    w = pd.to_numeric(df["wgt"], errors="coerce")
    slope = wls_slope(y, x, w)  # approx % change in level per 1-year age (on log scale)
    trend_rows.append({"asset": col, "slope_log1p_per_year": slope})
trend_df = pd.DataFrame(trend_rows)

# ------------------ monotonicity across age bands (medians) ------------------
def is_non_decreasing(seq):
    seq = [s for s in seq if np.isfinite(s)]
    return all(seq[i] <= seq[i+1] for i in range(len(seq)-1)) if seq else False

mono_rows = []
for col in bond_like + ["bond_total"]:
    medians = age_grp[f"{col}_median"].to_list()
    mono_rows.append({
        "asset": col,
        "non_decreasing_median_across_age_bins": bool(is_non_decreasing(medians))
    })
mono_df = pd.DataFrame(mono_rows)

# ------------------ print results ------------------
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

print(f"Age variable used: {age_col}")
print("\nWeighted ownership rates and balances by AGE BIN")
print("(ownrate is fraction with balance > 0; *_owners are conditional on ownership)\n")
print(age_grp)

print("\nWeighted trend: slope of log1p(balance) ~ age (per 1 year).")
print("Interpretation: positive slope ⇒ older households tend to hold larger amounts.\n")
print(trend_df.sort_values("asset"))

print("\nMonotonicity check across age bins (using weighted medians):")
print(mono_df.sort_values("asset"))

# ------------------ quick textual summary ------------------
def fmt_slope(s):
    return "increasing" if np.isfinite(s) and s > 0 else ("decreasing" if np.isfinite(s) and s < 0 else "n/a")

print("\nSummary:")
for _, r in trend_df.iterrows():
    asset = r["asset"]
    slope = r["slope_log1p_per_year"]
    mono_flag = bool(mono_df.loc[mono_df["asset"] == asset, "non_decreasing_median_across_age_bins"].iloc[0])
    print(f"  {asset:9s}: trend={fmt_slope(slope)} (slope={slope:,.5f}); "
          f"medians non-decreasing across age bins? {mono_flag}")
