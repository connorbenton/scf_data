import numpy as np
import pandas as pd
from pathlib import Path

CONCAT_PATH = "p22i6_with_concats.parquet"

# ---------- load + weights ----------
if not Path(CONCAT_PATH).exists():
    raise FileNotFoundError(f"Missing {CONCAT_PATH}. Run your concatenation script first.")

df = pd.read_parquet(CONCAT_PATH)
df.columns = [c.strip().lower() for c in df.columns]

# keep only positive weights; WGT = X42001/5
if "x42001" not in df.columns:
    raise KeyError("x42001 not found in dataframe.")
df["x42001"] = pd.to_numeric(df["x42001"], errors="coerce").fillna(0)
df = df[df["x42001"] > 0].copy()
df["wgt"] = df["x42001"] / 5.0

needed = ["networth", "liq", "stocks", "bond", "othma", "wgt"]
for c in needed:
    if c not in df.columns:
        raise KeyError(f"Missing required column: {c}")
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[np.isfinite(df["networth"]) & np.isfinite(df["wgt"]) & (df["wgt"] > 0)].copy()

# ---------- weighted quantile helper ----------
def weighted_quantiles(values, weights, probs):
    sorter = np.argsort(values)
    v = np.array(values)[sorter]
    w = np.array(weights)[sorter]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return np.interp(probs, cw, v)

# ---------- weighted median helpers ----------
def weighted_median(vals, wts):
    vals = np.asarray(vals)
    wts = np.asarray(wts)
    m = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    if not np.any(m):
        return np.nan
    vals = vals[m]
    wts = wts[m]
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    cw = np.cumsum(wts)
    cutoff = 0.5 * wts.sum()
    return vals[np.searchsorted(cw, cutoff)]

def bin_weighted_median(g, col, owners_only=False):
    v = pd.to_numeric(g[col], errors="coerce")
    w = pd.to_numeric(g["wgt"], errors="coerce")
    if owners_only:
        keep = (v > 0) & np.isfinite(v) & np.isfinite(w) & (w > 0)
    else:
        keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return np.nan
    return weighted_median(v[keep].to_numpy(), w[keep].to_numpy())

spec = {
    "liq":       dict(owners_only=False),
    "stocks":    dict(owners_only=True),
    "bond":      dict(owners_only=True),
    "othma":     dict(owners_only=True),
}

# ---------- weighted NW breakpoints ----------
probs_nw = np.array([0, 0.25, 0.50, 0.75, 0.90, 1.00])
bps_nw = weighted_quantiles(df["networth"], df["wgt"], probs_nw)

# ensure strictly increasing for pd.cut
bins_nw = np.maximum.accumulate(bps_nw)
for i in range(1, len(bins_nw)):
    if bins_nw[i] <= bins_nw[i - 1]:
        bins_nw[i] = np.nextafter(bins_nw[i - 1], np.inf)

labels_nw = ["<25", "25–49.9", "50–74.9", "75–89.9", "90–100"]
df["NW_BIN"] = pd.cut(df["networth"], bins=bins_nw, labels=labels_nw, right=False, include_lowest=True)

grp_nw = df.dropna(subset=["NW_BIN"]).groupby("NW_BIN", observed=True)

results_nw = pd.DataFrame({
    col: grp_nw.apply(lambda g, c=col, oo=kw["owners_only"]: bin_weighted_median(g, c, owners_only=oo))
    for col, kw in spec.items()
}).reindex(labels_nw)

# ---------- weighted INCOME breakpoints ----------
if "income" not in df.columns:
    raise KeyError("income not found in dataframe.")
df["income"] = pd.to_numeric(df["income"], errors="coerce")

probs_inc = np.array([0, 0.20, 0.40, 0.60, 0.80, 0.90, 1.00])
bps_inc = weighted_quantiles(df["income"], df["wgt"], probs_inc)

# ensure strictly increasing
bins_inc = np.maximum.accumulate(bps_inc)
for i in range(1, len(bins_inc)):
    if bins_inc[i] <= bins_inc[i - 1]:
        bins_inc[i] = np.nextafter(bins_inc[i - 1], np.inf)

labels_inc = ["<20", "20–39.9", "40–59.9", "60–79.9", "80–89.9", "90–100"]
df["INC_BIN"] = pd.cut(df["income"], bins=bins_inc, labels=labels_inc, right=False, include_lowest=True)

grp_inc = df.dropna(subset=["INC_BIN"]).groupby("INC_BIN", observed=True)

results_inc = pd.DataFrame({
    col: grp_inc.apply(lambda g, c=col, oo=kw["owners_only"]: bin_weighted_median(g, c, owners_only=oo))
    for col, kw in spec.items()
}).reindex(labels_inc)

# ---------- print ----------
print("NW weighted breakpoints (0,25,50,75,90,100):")
for p, v in zip([0,25,50,75,90,100], bins_nw):
    print(f"  {p:>3d}% : {v:,.2f}")

print("\nWeighted medians by NW percentile range:")
for lab, row in results_nw.iterrows():
    print(
        f"{lab:>8}: "
        f"LIQ={row['liq']:,.2f}  "
        f"STOCKS(owners)={row['stocks']:,.2f}  "
        f"BOND={row['bond']:,.2f}  "
        f"OTHMA={row['othma']:,.2f}"
    )

print("\nINCOME weighted breakpoints (0,20,40,60,80,90,100):")
for p, v in zip([0,20,40,60,80,90,100], bins_inc):
    print(f"  {p:>3d}% : {v:,.2f}")

print("\nWeighted medians by INCOME percentile range:")
for lab, row in results_inc.iterrows():
    print(
        f"{lab:>8}: "
        f"LIQ={row['liq']:,.2f}  "
        f"STOCKS(owners)={row['stocks']:,.2f}  "
        f"BOND={row['bond']:,.2f}  "
        f"OTHMA={row['othma']:,.2f}"
    )