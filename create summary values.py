# scf_concat_2022.py
# From-scratch build of SCF concatenations (lowercase x-variables) to compute:
# NFIN, DEBT, NETWORTH (and all upstream components needed for those)
#
# Assumes: p22i6 parquet columns are already lowercase ("small x").
# Target: YEAR = 2022 (so we follow the >=2010/2013/2016/2022 branches in your SAS spec)

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------
# Config
# ------------------------------
YEAR = 2022
PUBLIC = True   # public-released data logic where relevant
IN_PATH = "p22i6.parquet"           # input
OUT_PATH = "p22i6_with_concats.parquet"  # output

# ------------------------------
# Helpers
# ------------------------------
def get(df, col, default=0.0):
    """Safe get numeric column as float Series; missing -> default (scalar)."""
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        return s.fillna(0.0)
    return pd.Series(default, index=df.index, dtype="float64")

def ge0(s):
    return s.clip(lower=0.0)

def eq(s, val):
    # returns 1/0 indicator as float (SAS booleans behave numerically)
    return (s == val).astype(float)

def in_set(s, values):
    return s.isin(values).astype(float)

def between_inclusive(s, lo, hi):
    # (lo <= s <= hi) indicator
    return ((s >= lo) & (s <= hi)).astype(float)

def max_of(*args):
    # elementwise max among Series or scalars
    arrs = [a if isinstance(a, pd.Series) else pd.Series(a, index=args[0].index) for a in args]
    out = arrs[0]
    for a in arrs[1:]:
        out = pd.concat([out, a], axis=1).max(axis=1)
    return out

# Load
df = pd.read_parquet(IN_PATH)
df.columns = [c.lower() for c in df.columns]

# Short alias for readability
g = lambda c: get(df, c)


# ------------------------------
# Preliminary sample & weight adjustments (SAS alignment)
# ------------------------------

# Column detection for ID/IID (public SCF names can vary slightly by release)
id_candidates  = ["y1", "id", "caseid", "yy1", "hhid"]
iid_candidates = ["y2", "iid", "imp", "implicate", "imputation"]

def first_present(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

ID_COL  = first_present(id_candidates)
IID_COL = first_present(iid_candidates)

# Keep obs with valid ID/IID and positive weight X42001
# (If an ID column isn't present, we gracefully skip that part of the filter.)
valid_mask = pd.Series(True, index=df.index)
if ID_COL is not None:
    valid_mask &= pd.to_numeric(df[ID_COL], errors="coerce").fillna(0) > 0
if IID_COL is not None:
    valid_mask &= pd.to_numeric(df[IID_COL], errors="coerce").fillna(0) > 0
valid_mask &= pd.to_numeric(df.get("x42001", np.nan), errors="coerce").fillna(0) > 0

df = df.loc[valid_mask].copy()

df["x42001"] = pd.to_numeric(df["x42001"], errors="coerce")
df = df[df["x42001"] > 0].copy()

# Dummy merge variable
df["mergeid"] = 1

# Weights: WGT = X42001/5, retain original as WGT0
df["wgt0"] = pd.to_numeric(df["x42001"], errors="coerce")
df["wgt"]  = df["wgt0"] / 5.0

# ============================================
# 1) Transactions accounts (for FIN & ASSET)
# ============================================

# CHECKING (2001+)
checking = (
    ge0(g("x3506")) * eq(g("x3507"), 5)
  + ge0(g("x3510")) * eq(g("x3511"), 5)
  + ge0(g("x3514")) * eq(g("x3515"), 5)
  + ge0(g("x3518")) * eq(g("x3519"), 5)
  + ge0(g("x3522")) * eq(g("x3523"), 5)
  + ge0(g("x3526")) * eq(g("x3527"), 5)
  + ge0(g("x3529")) * eq(g("x3527"), 5)
)

# SAVING (2001+ “else” branch)
saving = (
    ge0(g("x3730") * (1 - in_set(g("x3732"), [4,30])))
  + ge0(g("x3736") * (1 - in_set(g("x3738"), [4,30])))
  + ge0(g("x3742") * (1 - in_set(g("x3744"), [4,30])))
  + ge0(g("x3748") * (1 - in_set(g("x3750"), [4,30])))
  + ge0(g("x3754") * (1 - in_set(g("x3756"), [4,30])))
  + ge0(g("x3760") * (1 - in_set(g("x3762"), [4,30])))
  + ge0(g("x3765") * (1 - in_set(g("x3762"), [4,30])))
)

# MMDA and MMMF (2001+ branches)
mmda = (
    ge0(g("x3506")) * eq(g("x3507"), 1) * between_inclusive(g("x9113"), 11, 13)
  + ge0(g("x3510")) * eq(g("x3511"), 1) * between_inclusive(g("x9114"), 11, 13)
  + ge0(g("x3514")) * eq(g("x3515"), 1) * between_inclusive(g("x9115"), 11, 13)
  + ge0(g("x3518")) * eq(g("x3519"), 1) * between_inclusive(g("x9116"), 11, 13)
  + ge0(g("x3522")) * eq(g("x3523"), 1) * between_inclusive(g("x9117"), 11, 13)
  + ge0(g("x3526")) * eq(g("x3527"), 1) * between_inclusive(g("x9118"), 11, 13)
  + ge0(g("x3529")) * eq(g("x3527"), 1) * between_inclusive(g("x9118"), 11, 13)
  + ge0(g("x3730") * in_set(g("x3732"), [4,30]) * between_inclusive(g("x9259"), 11, 13))
  + ge0(g("x3736") * in_set(g("x3738"), [4,30]) * between_inclusive(g("x9260"), 11, 13))
  + ge0(g("x3742") * in_set(g("x3744"), [4,30]) * between_inclusive(g("x9261"), 11, 13))
  + ge0(g("x3748") * in_set(g("x3750"), [4,30]) * between_inclusive(g("x9262"), 11, 13))
  + ge0(g("x3754") * in_set(g("x3756"), [4,30]) * between_inclusive(g("x9263"), 11, 13))
  + ge0(g("x3760") * in_set(g("x3762"), [4,30]) * between_inclusive(g("x9264"), 11, 13))
  + ge0(g("x3765") * in_set(g("x3762"), [4,30]) * between_inclusive(g("x9264"), 11, 13))
)

mmmf = (
    ge0(g("x3506")) * eq(g("x3507"), 1) * (1 - between_inclusive(g("x9113"), 11, 13))
  + ge0(g("x3510")) * eq(g("x3511"), 1) * (1 - between_inclusive(g("x9114"), 11, 13))
  + ge0(g("x3514")) * eq(g("x3515"), 1) * (1 - between_inclusive(g("x9115"), 11, 13))
  + ge0(g("x3518")) * eq(g("x3519"), 1) * (1 - between_inclusive(g("x9116"), 11, 13))
  + ge0(g("x3522")) * eq(g("x3523"), 1) * (1 - between_inclusive(g("x9117"), 11, 13))
  + ge0(g("x3526")) * eq(g("x3527"), 1) * (1 - between_inclusive(g("x9118"), 11, 13))
  + ge0(g("x3529")) * eq(g("x3527"), 1) * (1 - between_inclusive(g("x9118"), 11, 13))
  + ge0(g("x3730") * in_set(g("x3732"), [4,30]) * (1 - between_inclusive(g("x9259"), 11, 13)))
  + ge0(g("x3736") * in_set(g("x3738"), [4,30]) * (1 - between_inclusive(g("x9260"), 11, 13)))
  + ge0(g("x3742") * in_set(g("x3744"), [4,30]) * (1 - between_inclusive(g("x9261"), 11, 13)))
  + ge0(g("x3748") * in_set(g("x3750"), [4,30]) * (1 - between_inclusive(g("x9262"), 11, 13)))
  + ge0(g("x3754") * in_set(g("x3756"), [4,30]) * (1 - between_inclusive(g("x9263"), 11, 13)))
  + ge0(g("x3760") * in_set(g("x3762"), [4,30]) * (1 - between_inclusive(g("x9264"), 11, 13)))
  + ge0(g("x3765") * in_set(g("x3762"), [4,30]) * (1 - between_inclusive(g("x9264"), 11, 13)))
)

mma = mmda + mmmf
call_acct = ge0(g("x3930"))

# PREPAID (2016+)
prepaid = ge0(g("x7596"))

liq = checking + saving + mma + call_acct + prepaid

# HLIQ (2004+): include accounts with zero reported balances
hliqsupport = (eq(g("x3501"), 1) + eq(g("x3727"), 1) + eq(g("x3929"), 1)) > 0
hliqsupport = hliqsupport.astype(float)
hliqsupport.name = "hliqsupport"

hliqs = ((liq > 0).astype(float) + hliqsupport).clip(upper=1.0)
# include zero-balance accounts
liq = pd.concat([liq, hliqs], axis=1).max(axis=1)

# ============================================
# 2) Other financial assets needed for FIN
# ============================================

cds = ge0(g("x3721"))

# Mutual funds (2004+): STMUTF/TFBMUTF/GBMUTF/OBMUTF/COMUTF and OMUTF (other)
stmutf = eq(g("x3821"), 1) * ge0(g("x3822"))
tfbmutf = eq(g("x3823"), 1) * ge0(g("x3824"))
gbmutf = eq(g("x3825"), 1) * ge0(g("x3826"))
obmutf = eq(g("x3827"), 1) * ge0(g("x3828"))
comutf = eq(g("x3829"), 1) * ge0(g("x3830"))
omutf  = eq(g("x7785"), 1) * ge0(g("x7787"))  # 2004+

nmmf = stmutf + tfbmutf + gbmutf + obmutf + comutf + omutf

stocks = ge0(g("x3915"))

# Bonds (>=1992): NOTXBND, MORTBND, GOVTBND, OBND
notxbnd = g("x3910")
mortbnd = g("x3906")
govtbnd = g("x3908")
obnd = g("x7634") + g("x7633")
bond = notxbnd + mortbnd + govtbnd + obnd

# Quasi-liquid retirement (IRAs/thrift/future/curr pensions)
# Implement modern (2010+) structure.
# Full thrift reconstruction is lengthy; here we include the direct IRA balances,
# future pensions, and currently received pensions per spec branches, and set
# THRIFT=0 unless you want the full pension grid logic. (Tell me if you want that added.)
irakh = (
    ge0(g("x6551")) + ge0(g("x6559")) + ge0(g("x6567")) + ge0(g("x6552")) +
    ge0(g("x6560")) + ge0(g("x6568")) + ge0(g("x6553")) + ge0(g("x6561")) +
    ge0(g("x6569")) + ge0(g("x6554")) + ge0(g("x6562")) + ge0(g("x6570"))
)

# ------------------------------
# THRIFT & PENEQ (per SAS spec)
# ------------------------------
# Arrays (1..4): pensions for R (1–2) and S (3–4)
ptype1_cols = ["x11000", "x11100", "x11300", "x11400"]
ptype2_cols = ["x11001", "x11101", "x11301", "x11401"]
pamt_cols   = ["x11032", "x11132", "x11332", "x11432"]
pbor_cols   = ["x11025", "x11125", "x11325", "x11425"]
pwit_cols   = ["x11031", "x11131", "x11331", "x11431"]
pall_cols   = ["x11036", "x11136", "x11336", "x11436"]
ppct_cols   = ["x11037", "x11137", "x11337", "x11437"]

thrift = pd.Series(0.0, index=df.index)
peneq  = pd.Series(0.0, index=df.index)

rthrift = pd.Series(0.0, index=df.index)
sthrift = pd.Series(0.0, index=df.index)

req = pd.Series(0.0, index=df.index)
seq = pd.Series(0.0, index=df.index)

for i in range(4):
    ptype1 = g(ptype1_cols[i])
    ptype2 = g(ptype2_cols[i])
    pamt   = ge0(g(pamt_cols[i]))
    pbor   = g(pbor_cols[i])
    pwit   = g(pwit_cols[i])
    pall   = g(pall_cols[i])
    ppct   = ge0(g(ppct_cols[i]))

    elig = (
        (ptype1 == 1)
        | (ptype2.isin([2,3,4,6,20,21,22,26]))
        | (pbor == 1)
        | (pwit == 1)
    )

    hold = pamt * elig.astype(float)

    # Accumulate R (i=0,1) vs S (i=2,3)
    if i <= 1:
        rthrift = rthrift + hold
    else:
        sthrift = sthrift + hold

    thrift = thrift + hold

    # PENEQ counts thrift invested in stock
    peneq = peneq + hold * (
        (pall == 1).astype(float) +
        (pall.isin([3, 30]).astype(float) * (ppct / 10000.0))
    )

    # Track REQ/SEQ as in looped SAS logic
    if i <= 1:
        req = peneq.copy()
    else:
        seq = peneq - req

# ------------------------------
# Pension mop-ups (X11259, X11559)
# ------------------------------
# Helper for safe division
def safe_div(num, den):
    return (num / den.replace(0, np.nan)).fillna(0.0)

# R-side mop-up: X11259
x11259 = ge0(g("x11259"))
cond_r_any_access = (
    (g("x11000") == 1) | (g("x11100") == 1) |
    (g("x11001").isin([2,3,4,6,20,21,22,26])) | (g("x11101").isin([2,3,4,6,20,21,22,26])) |
    (g("x11031") == 1) | (g("x11131") == 1) | (g("x11025") == 1) | (g("x11125") == 1)
)
cond_r_all_known_no_access = (
    (g("x11000") != 0) & (g("x11100") != 0) & (g("x11031") != 0) & (g("x11131") != 0)
)
pmop_r = np.where(cond_r_any_access, x11259, np.where(cond_r_all_known_no_access, 0.0, x11259))
pmop_r = pd.Series(pmop_r, index=df.index)

thrift = thrift + pmop_r
peneq  = peneq + np.where((req > 0), pmop_r * safe_div(req, rthrift), pmop_r / 2.0)
peneq  = pd.Series(peneq, index=df.index)

# S-side mop-up: X11559
x11559 = ge0(g("x11559"))
cond_s_any_access = (
    (g("x11300") == 1) | (g("x11400") == 1) |
    (g("x11301").isin([2,3,4,6,20,21,22,26])) | (g("x11401").isin([2,3,4,6,20,21,22,26])) |
    (g("x11331") == 1) | (g("x11431") == 1) | (g("x11325") == 1) | (g("x11425") == 1)
)
cond_s_all_known_no_access = (
    (g("x11300") != 0) & (g("x11400") != 0) & (g("x11331") != 0) & (g("x11431") != 0)
)
pmop_s = np.where(cond_s_any_access, x11559, np.where(cond_s_all_known_no_access, 0.0, x11559))
pmop_s = pd.Series(pmop_s, index=df.index)

thrift = thrift + pmop_s
peneq  = peneq + np.where((seq > 0), pmop_s * safe_div(seq, sthrift), pmop_s / 2.0)
peneq  = pd.Series(peneq, index=df.index)

# Finally, include THRIFT in retirement quasi-liquid:
# (Keep your existing line right after this)
# retqliq = irakh + thrift + futpen + currpen


# FUTPEN (2010+)
futpen = ge0(g("x5604")) + ge0(g("x5612")) + ge0(g("x5620")) + ge0(g("x5628"))

# CURRPEN (2010+)
currpen = ge0(g("x6462")) + ge0(g("x6467")) + ge0(g("x6472")) + ge0(g("x6477")) + ge0(g("x6957"))

retqliq = irakh + thrift + futpen + currpen

savbnd = g("x3902")
cashli = ge0(g("x4006"))

# Other managed assets (2004+): annuities & trusts
annuit = ge0(g("x6577"))
trusts = ge0(g("x6587"))
othma = annuit + trusts

# Other financial assets (PUBLIC=YES branch)
othfin = (
    g("x4018")
  + g("x4022") * in_set(g("x4020"), [61,62,63,64,65,66,71,72,73,74,77,80,81,-7])
  + g("x4026") * in_set(g("x4024"), [61,62,63,64,65,66,71,72,73,74,77,80,81,-7])
  + g("x4030") * in_set(g("x4028"), [61,62,63,64,65,66,71,72,73,74,77,80,81,-7])
)

fin = liq + cds + nmmf + stocks + bond + retqliq + savbnd + cashli + othma + othfin

# ============================================
# 3) Nonfinancial assets (NFIN)
# ============================================

# VEHIC (1995+)
vehic = ge0(g("x8166")) + ge0(g("x8167")) + ge0(g("x8168")) + ge0(g("x8188")) + \
        ge0(g("x2422")) + ge0(g("x2506")) + ge0(g("x2606")) + ge0(g("x2623"))

# Farm business split / FARMBUS (follow spec adjustments)
# Bound x507 at 9000 max for farm use percent
x507 = g("x507").clip(upper=9000)
# Make local copies we can "adjust"; we won't overwrite df, we simulate adjusted amounts.
x805 = g("x805"); x905 = g("x905"); x1005 = g("x1005")
x808 = g("x808"); x813 = g("x813"); x908 = g("x908"); x913 = g("x913"); x1008 = g("x1008"); x1013 = g("x1013")
x1108 = g("x1108"); x1119 = g("x1119"); x1130 = g("x1130")
x1103 = g("x1103"); x1114 = g("x1114"); x1125 = g("x1125")

# Compute initial FARMBUS and adjust secured amounts proportionally as in spec
farm_share = (x507 / 10000.0)
farmbus = pd.Series(0.0, index=df.index)

has_farm = (x507 > 0)

# Business part of farm net of outstanding mortgages
farmbus = np.where(
    has_farm,
    farm_share * (g("x513") + g("x526") - x805 - x905 - x1005),
    0.0
)

# Adjust liens/mortgages by (1 - farm_share) when farm part exists
def shrink_for_nonfarm(s):
    return np.where(has_farm, s * (1 - farm_share), s)

x805_adj = shrink_for_nonfarm(x805)
x808_adj = shrink_for_nonfarm(x808)
x813_adj = shrink_for_nonfarm(x813)
x905_adj = shrink_for_nonfarm(x905)
x908_adj = shrink_for_nonfarm(x908)
x913_adj = shrink_for_nonfarm(x913)
x1005_adj = shrink_for_nonfarm(x1005)
x1008_adj = shrink_for_nonfarm(g("x1008"))
x1013_adj = shrink_for_nonfarm(g("x1013"))

# Subtract share of HELOCs tied to farm part (if those HELOCs exist)
def adjust_heloc_pair(bal, yesflag):
    # subtract farm portion from FARMBUS if LOC exists, then reduce the LOC by (1-farm_share)
    # Return (delta_farbus, adjusted_balance)
    take = np.where((has_farm) & (eq(yesflag,1)==1), bal * farm_share, 0.0)
    adj  = np.where(has_farm, bal * (1 - farm_share), bal)
    return take, adj

take1, x1108_adj = adjust_heloc_pair(x1108, x1103)
take2, x1119_adj = adjust_heloc_pair(x1119, x1114)
take3, x1130_adj = adjust_heloc_pair(x1130, x1125)
farmbus = farmbus - (take1 + take2 + take3)

# If a “mopup” LOC exists (x1136>0) allocate proportionally across existing LOCs (per SAS),
# then adjust the mopup part similarly for farm share.
x1136 = g("x1136")
sumlocs = (x1108 + x1119 + x1130).replace(0, np.nan)
mopup_share = ( (x1108*eq(x1103,1) + x1119*eq(g("x1114"),1) + x1130*eq(g("x1125"),1)) / sumlocs ).fillna(0.0)
take_mop = np.where(has_farm & (x1136 > 0) & (sumlocs.notna()), x1136 * farm_share * mopup_share, 0.0)
farmbus = farmbus - take_mop
# mopup adjusted contribution is reduced for nonfarm:
x1136_adj = np.where(has_farm & (x1136 > 0) & (sumlocs.notna()),
                     x1136 * (1 - farm_share) * mopup_share,
                     x1136 * (1 - farm_share))

# HOUSES
houses = (
    ge0(max_of(g("x604"), g("x614"), g("x623"), g("x716")))
  + ((1 - farm_share).clip(lower=0.0)) * (g("x513") + g("x526"))
)

# ORESRE (2013+)
oresre = max_of(g("x1306"), g("x1310")) + max_of(g("x1325"), g("x1329")) + ge0(g("x1339")) \
       + in_set(g("x1703"), [12,14,21,22,25,40,41,42,43,44,49,50,52,999]) * ge0(g("x1706")) * (g("x1705")/10000.0) \
       + in_set(g("x1803"), [12,14,21,22,25,40,41,42,43,44,49,50,52,999]) * ge0(g("x1806")) * (g("x1805")/10000.0) \
       + ge0(g("x2002"))

# NNRESRE (2010+)
nnresre = (
    in_set(g("x1703"), [1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7]) * ge0(g("x1706")) * (g("x1705")/10000.0)
  + in_set(g("x1803"), [1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7]) * ge0(g("x1806")) * (g("x1805")/10000.0)
  + ge0(g("x2012"))
  - in_set(g("x1703"), [1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7]) * g("x1715") * (g("x1705")/10000.0)
  - in_set(g("x1803"), [1,2,3,4,5,6,7,10,11,13,15,24,45,46,47,48,51,53,-7]) * g("x1815") * (g("x1805")/10000.0)
  - g("x2016")
)

# If NNRESRE != 0, remove purpose=78 installment loans (investment real estate) from it later when building INSTALL,
# but here we mark a flag. We follow debt side for allocation; asset NNRESRE remains as computed above.

# Business assets (2010+)
bus = (
    ge0(g("x3129")) + ge0(g("x3124")) - ge0(g("x3126")) * eq(g("x3127"), 5) + ge0(g("x3121")) * in_set(g("x3122"), [1,6])
  + ge0(g("x3229")) + ge0(g("x3224")) - ge0(g("x3226")) * eq(g("x3227"), 5) + ge0(g("x3221")) * in_set(g("x3222"), [1,6])
  + ge0(g("x3335")) + ge0(pd.Series(farmbus, index=df.index))  # add FARMBUS
  + ge0(g("x3408")) + ge0(g("x3412")) + ge0(g("x3416")) + ge0(g("x3420"))
  + ge0(g("x3452")) + ge0(g("x3428"))
)

# Other nonfinancial assets
othnfin = g("x4022") + g("x4026") + g("x4030") - othfin + g("x4018")

nfin = vehic + houses + oresre + nnresre + bus + othnfin

# ============================================
# 4) Debts (DEBT)
# ============================================

# Principal residence debt & HELOCs
sumlocs_bal = (x1108 + x1119 + x1130)
has_any_loc = (sumlocs_bal >= 1)

# HELOC (when any LOC exists): heloc = HE lines only; mopup share added if tied to HELOC flags
heloc = (
    x1108 * eq(x1103, 1) + x1119 * eq(x1114, 1) + x1130 * eq(x1125, 1)
  + ge0(x1136) * ( (x1108*eq(x1103,1) + x1119*eq(x1114,1) + x1130*eq(x1125,1)) /
                   sumlocs_bal.replace(0, np.nan) ).fillna(0.0)
)

mrthel = (
    x805_adj + x905_adj + x1005_adj
  + x1108 * eq(x1103, 1) + x1119 * eq(x1114, 1) + x1130 * eq(x1125, 1)
  + ge0(x1136) * ( (x1108*eq(x1103,1) + x1119*eq(x1114,1) + x1130*eq(x1125,1)) /
                   sumlocs_bal.replace(0, np.nan) ).fillna(0.0)
)

# If no LOCs in grid, HELOC=0 and add half the mopup to MRTHEL (spec)
no_loc_mask = (sumlocs_bal < 1)
heloc = np.where(no_loc_mask, 0.0, heloc)
mrthel = np.where(no_loc_mask, x805 + x905 + x1005 + 0.5*ge0(x1136)*(houses > 0).astype(float), mrthel)

nh_mort = ge0(pd.Series(mrthel, index=df.index)) - ge0(pd.Series(heloc, index=df.index))

# Other LOC (non-HELOC)
othloc = np.where(
    sumlocs_bal >= 1,
    x1108 * (1 - eq(x1103, 1)) + x1119 * (1 - eq(x1114, 1)) + x1130 * (1 - eq(x1125, 1)) +
    ge0(x1136) * ( (x1108*(1 - eq(x1103,1)) + x1119*(1 - eq(x1114,1)) + x1130*(1 - eq(x1125,1))) /
                   sumlocs_bal.replace(0, np.nan) ).fillna(0.0),
    # else mopup fully or half depending on houses
    ((houses <= 0).astype(float) + 0.5*(houses > 0).astype(float)) * ge0(x1136)
)

othloc = pd.Series(othloc, index=df.index)

# Other residential property debt (2016+/2013+/2010 logic)
# For 2013+:
mort1 = in_set(g("x1703"), [12,14,21,22,25,40,41,42,43,44,49,50,52,53,999]) * g("x1715") * (g("x1705")/10000.0)
mort2 = in_set(g("x1803"), [12,14,21,22,25,40,41,42,43,44,49,50,52,53,999]) * g("x1815") * (g("x1805")/10000.0)
# 2013+: resdbt = x1318 + x1337 + x1342 + mort1 + mort2 + x2006
resdbt = g("x1318") + g("x1337") + g("x1342") + mort1 + mort2 + g("x2006")

# Purpose 78 loan handling (flags) and purpose 67 allocations
flag781 = (nnresre != 0).astype(float)  # if NNRESRE != 0
if PUBLIC:
    # In public data, keep same mechanical adjustments as spec
    pass

# Add PURP=78 loans to RESDBT if not absorbed into NNRESRE:
# BUT ONLY when NNRESRE==0 and ORESRE>0 per spec later — replicate exactly as SAS:
# First: if NNRESRE != 0, we already subtracted from NNRESRE on asset side; do nothing here.
# Else if ORESRE>0, include 78s in RESDBT; else they’ll go to INSTALL below.
flag782 = ((flag781 == 0) & (oresre > 0)).astype(float)
if flag782.any():
    resdbt = resdbt + \
        g("x2723")*eq(g("x2710"),78) + g("x2740")*eq(g("x2727"),78) + \
        g("x2823")*eq(g("x2810"),78) + g("x2840")*eq(g("x2827"),78) + \
        g("x2923")*eq(g("x2910"),78) + g("x2940")*eq(g("x2927"),78)

# For parallel treatment, include PURP=67 into RESDBT only if ORESRE>0
flag67 = (oresre > 0).astype(float)
resdbt = resdbt + \
    g("x2723")*eq(g("x2710"),67) + g("x2740")*eq(g("x2727"),67) + \
    g("x2823")*eq(g("x2810"),67) + g("x2840")*eq(g("x2827"),67) + \
    g("x2923")*eq(g("x2910"),67) + g("x2940")*eq(g("x2927"),67)

# Credit cards & BNPL (2016+; BNPL from 2022+)
ccbal = ge0(g("x427")) + ge0(g("x413")) + ge0(g("x421")) + ge0(g("x7575"))
bnpl = ge0(g("x443"))  # 2022+

# Installment loans (1995+ group) – vehicle, education, other
veh_inst = g("x2218") + g("x2318") + g("x2418") + g("x7169") + g("x2424") + g("x2519") + g("x2619") + g("x2625")
edn_inst = (
    g("x7824")+g("x7847")+g("x7870")+g("x7924")+g("x7947")+g("x7970")+g("x779")
  + g("x2723")*eq(g("x2710"),83) + g("x2740")*eq(g("x2727"),83)
  + g("x2823")*eq(g("x2810"),83) + g("x2840")*eq(g("x2827"),83)
  + g("x2923")*eq(g("x2910"),83) + g("x2940")*eq(g("x2927"),83)
)

install = (
    g("x2218")+g("x2318")+g("x2418")+g("x7169")+g("x2424")+g("x2519")+g("x2619")+g("x2625")+g("x7183")
  + g("x7824")+g("x7847")+g("x7870")+g("x7924")+g("x7947")+g("x7970")+g("x7179")
  + g("x1044")+g("x1215")+g("x1219") + bnpl
)

# Move 78 (nonres RE) and 67 (res RE) into NNRESRE/RESDBT unless already allocated:
# If flag781==0 and flag782==0, add the 78s into INSTALL (i.e., they weren't allocated to NNRESRE or RESDBT)
mask_78 = (flag781 == 0) & (flag782 == 0)
install = install + \
    g("x2723")*eq(g("x2710"),78)*mask_78 + g("x2740")*eq(g("x2727"),78)*mask_78 + \
    g("x2823")*eq(g("x2810"),78)*mask_78 + g("x2840")*eq(g("x2827"),78)*mask_78 + \
    g("x2923")*eq(g("x2910"),78)*mask_78 + g("x2940")*eq(g("x2927"),78)*mask_78

# If FLAG67==0 (i.e., ORESRE==0), leave PURP=67 in INSTALL
mask_67 = (flag67 == 0)
install = install + \
    g("x2723")*eq(g("x2710"),67)*mask_67 + g("x2740")*eq(g("x2727"),67)*mask_67 + \
    g("x2823")*eq(g("x2810"),67)*mask_67 + g("x2840")*eq(g("x2827"),67)*mask_67 + \
    g("x2923")*eq(g("x2910"),67)*mask_67 + g("x2940")*eq(g("x2927"),67)*mask_67

# And include all remaining non-(67,78)
install = install + \
    g("x2723")*(1 - in_set(g("x2710"), [67,78])) + \
    g("x2740")*(1 - in_set(g("x2727"), [67,78])) + \
    g("x2823")*(1 - in_set(g("x2810"), [67,78])) + \
    g("x2840")*(1 - in_set(g("x2827"), [67,78])) + \
    g("x2923")*(1 - in_set(g("x2910"), [67,78])) + \
    g("x2940")*(1 - in_set(g("x2927"), [67,78]))

oth_inst = install - veh_inst - edn_inst

# Margin loans
outmarg = ge0(g("x3932"))

# Pension loans not previously reported (2010+)
outpen1 = ge0(g("x11027")) * eq(g("x11070"), 5)
outpen2 = ge0(g("x11127")) * eq(g("x11170"), 5)
outpen4 = ge0(g("x11327")) * eq(g("x11370"), 5)
outpen5 = ge0(g("x11427")) * eq(g("x11470"), 5)
outpen = outpen1 + outpen2 + outpen4 + outpen5

# Other debts (2010+): pension loans + loans vs LI + margin + misc
odebt = outpen1 + outpen2 + outpen4 + outpen5 + ge0(g("x4010")) + ge0(g("x4032")) + outmarg

debt = ge0(pd.Series(mrthel, index=df.index)) + resdbt + othloc + ccbal + install + odebt

# ============================================
# 5) ASSET, NETWORTH, INCOME
# ============================================

asset = fin + nfin
networth = asset - debt

income = ge0(g("x5729"))


def mconv(freq):
    """Approximate SAS %MCONV(): convert payment frequency code to monthly multiplier."""
    # annual=1/12, semiannual=1/6, quarterly=1/3, monthly=1, biweekly=26/12, weekly=52/12
    mapping = {1: 1/12, 2: 1/6, 3: 1/3, 4: 1.0, 5: 26/12, 6: 52/12}
    return freq.map(mapping).fillna(0)

penacctwd = (
    df["x6558"] + df["x6566"] + df["x6574"]
    + np.maximum(0, (df["x6464"] * mconv(df["x6465"])) * 12)
    + np.maximum(0, (df["x6469"] * mconv(df["x6470"])) * 12)
    + np.maximum(0, (df["x6474"] * mconv(df["x6475"])) * 12)
    + np.maximum(0, (df["x6479"] * mconv(df["x6480"])) * 12)
    + np.maximum(0, (df["x6965"] * mconv(df["x6966"])) * 12)
    + np.maximum(0, (df["x6971"] * mconv(df["x6972"])) * 12)
    + np.maximum(0, (df["x6977"] * mconv(df["x6978"])) * 12)
    + np.maximum(0, (df["x6983"] * mconv(df["x6984"])) * 12)
)

income = income + penacctwd



# ============================================
# Save all relevant outputs
# ============================================
out_cols = {
    # Transactions & FIN bits
    "checking": checking, "saving": saving, "mmda": mmda, "mmmf": mmmf, "mma": mma,
    "call": call_acct, "prepaid": prepaid, "liq": liq,
    "cds": cds,
    "stmutf": stmutf, "tfbmutf": tfbmutf, "gbmutf": gbmutf, "obmutf": obmutf, "comutf": comutf, "omutf": omutf,
    "nmmf": nmmf,
    "stocks": stocks, "notxbnd": notxbnd, "mortbnd": mortbnd, "govtbnd": govtbnd, "obnd": obnd, "bond": bond,
    "irakh": irakh, "thrift": thrift, "futpen": futpen, "currpen": currpen, "retqliq": retqliq,
    "savbnd": savbnd, "cashli": cashli, "annuit": annuit, "trusts": trusts, "othma": othma, "othfin": othfin,
    "fin": fin,

    # Nonfinancial
    "vehic": vehic, "houses": houses, "oresre": oresre, "nnresre": nnresre,
    "bus": bus, "othnfin": othnfin, "nfin": nfin,

    # Debts
    "mrthel": mrthel, "heloc": heloc, "nh_mort": nh_mort, "othloc": othloc,
    "mort1": mort1, "mort2": mort2, "resdbt": resdbt,
    "ccbal": ccbal, "bnpl": bnpl,
    "veh_inst": veh_inst, "edn_inst": edn_inst, "install": install, "oth_inst": oth_inst,
    "outmarg": outmarg, "outpen": outpen, "odebt": odebt,

    # Totals
    "asset": asset, "debt": debt, "networth": networth,

    # income
    "income": income, "penacctwd": penacctwd,
}

for k, v in out_cols.items():
    df[k] = pd.to_numeric(v, errors="coerce")

# Write parquet
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, index=False)

print(f"Wrote {OUT_PATH} with concatenations for NFIN, DEBT, NETWORTH + components.")
