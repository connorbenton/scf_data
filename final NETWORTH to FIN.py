import numpy as np
import pandas as pd
import plotly.graph_objects as go

def get_figure() -> go.Figure:
    # =============================
    # User knobs
    # =============================
    X_START = 1.0
    X_END   = 99.0
    STEP    = 0.2
    SMOOTH_WIN = 15   # smooth only the share series (not NW)
    MASK_THRESH = 0.02  # mask components <2% of NW per bin

    NAME_MAP = {
        "NW":   "Net worth",
        "FIN":  "Financial assets",
        "NFIN": "Nonfinancial assets",
        "DEBT": "Debt",
        "LIQ": "All liquid cash",
        "CDS": "CDs",
        "NMMF": "Mutual funds/ETFs",
        "STOCKS": "Stocks",
        "BOND": "Bonds",
        "RETQLIQ": "Retirement account value",
        "SAVBND": "Savings bonds",
        "CASHLI": "Life insurance cash value",
        "OTHMA": "Trusts, annuities, managed assets",
        "OTHFIN": "Other financial assets",
    }

    # palette
    COLOR = {
        "NW":   "#1f77b4",  # blue
        "FIN":  "#2ca02c",  # green
        "NFIN": "#9467bd",  # purple
        "DEBT": "#d62728",  # red
        # FIN subcomponents
        "LIQ":"#17becf",
        "CDS":"#bcbd22",
        "NMMF":"#8c564b",
        "STOCKS":"#ff7f0e",
        "BOND":"#7f7f7f",
        "RETQLIQ":"#1f77b4",
        "SAVBND":"#e377c2",
        "CASHLI":"#2ca02c",
        "OTHMA":"#9467bd",
        "OTHFIN":"#d62728",
    }
    DASH = {
        "NW":"solid", "FIN":"solid", "NFIN":"solid", "DEBT":"solid",
        "LIQ":"dash","CDS":"dash","NMMF":"dash","STOCKS":"dash","BOND":"dash",
        "RETQLIQ":"dash","SAVBND":"dash","CASHLI":"dash","OTHMA":"dash","OTHFIN":"dash",
    }

    # =============================
    # Data
    # =============================
    df = pd.read_parquet("p22i6_with_concats.parquet")
    df.columns = [c.lower().strip() for c in df.columns]

    req = ["networth","fin","nfin","debt","x14","x42001",
           "liq","cds","nmmf","stocks","bond","retqliq","savbnd","cashli","othma","othfin"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns: {miss}")

    for c in req:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["networth","x14","x42001"])
    df = df[df["x42001"] > 0].copy()
    df["wgt"] = df["x42001"] / 5.0

    # =============================
    # Utilities
    # =============================
    def weighted_percentiles(values, weights, probs):
        values = np.asarray(values); weights = np.asarray(weights)
        m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not np.any(m): return np.full_like(probs, np.nan, dtype=float)
        sorter = np.argsort(values[m])
        v = values[m][sorter]; w = weights[m][sorter]
        cw = np.cumsum(w); cw /= cw[-1]
        return np.interp(probs, cw, v)

    def make_percentile_grid(start=1.0, end=99.0, step=0.5):
        centers_pct = np.round(np.arange(start, end + 1e-9, step), 10)
        half = step / 2.0
        edges_pct = np.r_[centers_pct[0] - half,
                          0.5 * (centers_pct[:-1] + centers_pct[1:]),
                          centers_pct[-1] + half]
        edges_pct = np.clip(edges_pct, 1.0, 99.0)
        return centers_pct / 100.0, edges_pct / 100.0

    def decimals_from_step(step):
        s = f"{step:.10f}".rstrip("0").rstrip(".")
        return len(s.split(".")[1]) if "." in s else 0

    def round100(a):
        a = np.asarray(a, float)
        return np.where(np.isfinite(a), np.round(a/100.0)*100.0, np.nan)

    def swatch(color, dash="solid"):
        if dash in ("dot","dotted"): glyph = "⋯⋯⋯"
        elif dash in ("dash","dashed"): glyph = "─ ─ ─"
        else: glyph = "━━━"
        return f"<span style='color:{color};font-weight:700'>{glyph}</span>&nbsp;"

    # =============================
    # NW-aligned curves with FIN tertiary decomposition
    # =============================
    FIN_SUBS = ["liq","retqliq","nmmf","stocks","bond","cds","savbnd","cashli","othma","othfin"]

    def nw_aligned_with_fin_tertiary(df, age_mask, centers_probs, edges_probs, smooth_window=5):
        base = df.copy()
        cols = ["networth","fin","nfin","debt","wgt"] + FIN_SUBS
        for c in cols:
            base[c] = pd.to_numeric(base[c], errors="coerce")
        m = (base["wgt"] > 0) & np.isfinite(base["networth"])
        if age_mask is not None:
            m &= age_mask.reindex(base.index, fill_value=False)
        base = base.loc[m, cols].copy()
        if base.empty: return None

        # Master NW curve
        nw_curve = weighted_percentiles(base["networth"], base["wgt"], centers_probs)

        # NW-bin edges (in NW value space)
        edge_vals = weighted_percentiles(base["networth"], base["wgt"], edges_probs)
        edge_vals = np.maximum.accumulate(edge_vals)

        n_bins = len(centers_probs)
        idx = np.digitize(base["networth"].to_numpy(), edge_vals, right=True) - 1
        idx = np.clip(idx, 0, n_bins - 1)

        # Weighted means per NW bin
        nw_means   = np.full(n_bins, np.nan)
        fin_means  = np.full(n_bins, np.nan)
        nfin_means = np.full(n_bins, np.nan)
        debt_means = np.full(n_bins, np.nan)
        sub_means  = {k: np.full(n_bins, np.nan) for k in FIN_SUBS}

        for b in range(n_bins):
            sel = (idx == b)
            if not np.any(sel): continue
            w = base.loc[sel, "wgt"]
            nwbar = np.average(base.loc[sel, "networth"], weights=w)
            if not np.isfinite(nwbar) or nwbar <= 0:
                continue
            nw_means[b]   = nwbar
            fin_means[b]  = np.average(base.loc[sel, "fin"],  weights=w)
            nfin_means[b] = np.average(base.loc[sel, "nfin"], weights=w)
            debt_means[b] = np.average(base.loc[sel, "debt"], weights=w)  # positive magnitude
            for k in FIN_SUBS:
                sub_means[k][b] = np.average(base.loc[sel, k], weights=w)

        # Shares vs NW
        with np.errstate(divide="ignore", invalid="ignore"):
            r_fin  = fin_means  / nw_means
            r_nfin = nfin_means / nw_means
            r_debt = debt_means / nw_means

        # Shares inside FIN
        with np.errstate(divide="ignore", invalid="ignore"):
            s_sub = {k: sub_means[k] / fin_means for k in FIN_SUBS}

        # Smooth shares only
        if smooth_window and smooth_window > 1:
            def smooth(a):
                return pd.Series(a, dtype="float64").rolling(
                    window=smooth_window, center=True, min_periods=1
                ).median().to_numpy()
            r_fin  = smooth(r_fin)
            r_nfin = smooth(r_nfin)
            r_debt = smooth(r_debt)
            for k in FIN_SUBS:
                s_sub[k] = smooth(s_sub[k])

        # Compose curves
        fin_curve  = r_fin  * nw_curve
        nfin_curve = r_nfin * nw_curve
        debt_curve = r_debt * nw_curve

        fin_sub_curves = {k: s_sub[k] * fin_curve for k in FIN_SUBS}

        # Mask tiny components relative to NW
        cut = MASK_THRESH * nw_curve
        def mask_arr(arr):
            arr[(~np.isfinite(arr)) | (~np.isfinite(nw_curve)) | (nw_curve <= 0) | (arr < cut)] = np.nan
            return arr
        fin_curve  = mask_arr(fin_curve)
        nfin_curve = mask_arr(nfin_curve)
        debt_curve = mask_arr(debt_curve)
        for k in FIN_SUBS:
            fin_sub_curves[k] = mask_arr(fin_sub_curves[k])

        # Rescale FIN subcomponents to sum to FIN exactly
        subsum = np.nansum([fin_sub_curves[k] for k in FIN_SUBS], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            sub_scale = np.where(np.isfinite(subsum) & (subsum > 0) & np.isfinite(fin_curve),
                                 fin_curve / subsum, np.nan)
        for k in FIN_SUBS:
            ok = np.isfinite(sub_scale)
            fin_sub_curves[k][ok] = fin_sub_curves[k][ok] * sub_scale[ok]

        # Enforce FIN + NFIN − DEBT == NW, and propagate scale to subcomponents
        denom = fin_curve + nfin_curve - debt_curve
        with np.errstate(divide="ignore", invalid="ignore"):
            tri_scale = np.where(np.isfinite(denom) & (denom > 0) & np.isfinite(nw_curve),
                                 nw_curve / denom, np.nan)
        ok = np.isfinite(tri_scale)
        if np.any(ok):
            fin_curve[ok]  *= tri_scale[ok]
            nfin_curve[ok] *= tri_scale[ok]
            debt_curve[ok] *= tri_scale[ok]
            for k in FIN_SUBS:
                fin_sub_curves[k][ok] *= tri_scale[ok]

        return {"nw": nw_curve, "fin": fin_curve, "nfin": nfin_curve, "debt": debt_curve,
                "fin_subs": fin_sub_curves, "probs": centers_probs}

    # =============================
    # Build by age groups
    # =============================
    age_labels = ["All ages","<35","35–44","45–54","55–64","65–74","≥75"]

    def age_mask(label):
        s = df["x14"]
        if label=="<35": return s<35
        if label=="35–44": return (s>=35)&(s<=44)
        if label=="45–54": return (s>=45)&(s<=54)
        if label=="55–64": return (s>=55)&(s<=64)
        if label=="65–74": return (s>=65)&(s<=74)
        if label=="≥75": return s>=75
        return pd.Series(True, index=df.index)

    centers_probs, edges_probs = make_percentile_grid(X_START, X_END, STEP)

    curves_by_age = {}
    y_ranges = []  # per-age y-range (log10)
    for lbl in age_labels:
        res = nw_aligned_with_fin_tertiary(
            df, age_mask(lbl), centers_probs, edges_probs, smooth_window=SMOOTH_WIN
        )
        if res is None:
            px = np.arange(X_START, X_END + 1e-9, STEP)
            empty = np.full(len(px), np.nan)
            curves_by_age[lbl] = {"px": px, "nw": empty, "fin": empty, "nfin": empty, "debt": empty,
                                  "fin_subs": {k: empty.copy() for k in FIN_SUBS}}
            y_ranges.append((0.0, 2.0))
        else:
            px = res["probs"] * 100.0
            res["px"] = px
            del res["probs"]
            curves_by_age[lbl] = res

            # y-range that includes master + all components
            pools = [res["nw"], res["fin"], res["nfin"], res["debt"]] + [res["fin_subs"][k] for k in FIN_SUBS]
            pos = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in pools]) if pools else np.array([])

            if pos.size:
                # Always include the true NW max, then give generous headroom
                hard_max = float(np.nanmax(res["nw"]))
                soft_max = float(np.nanpercentile(pos, 99.9))
                ymax = max(hard_max, soft_max) * 1.25
                # Keep bottom fixed at 10^2 = 100
                y_ranges.append((2.0, np.log10(ymax)))
            else:
                y_ranges.append((2.0, 2.0))
    # =============================
    # Tooltip
    # =============================
    def build_tooltip(label):
        nw   = round100(curves_by_age[label]["nw"])
        fin  = round100(curves_by_age[label]["fin"])
        nfin = round100(curves_by_age[label]["nfin"])
        debt = round100(curves_by_age[label]["debt"])
        subs = {k.upper(): round100(curves_by_age[label]["fin_subs"][k]) for k in FIN_SUBS}

        n = len(curves_by_age[label]["px"])
        texts=[]
        for i in range(n):
            lines=[]
            if np.isfinite(nw[i]):   lines.append(f"{swatch(COLOR['NW'])}<b>{NAME_MAP['NW']}</b>: {nw[i]:,.0f}")
            if np.isfinite(fin[i]):  lines.append(f"{swatch(COLOR['FIN'])}<b>{NAME_MAP['FIN']}</b>: {fin[i]:,.0f}")
            # show tertiary FIN breakdown (largest first)
            items = [(lab, subs[lab][i]) for lab in subs if np.isfinite(subs[lab][i])]
            def to_num(s):
                try: return float(s)
                except: return -np.inf
            items.sort(key=lambda x: -to_num(x[1]))
            for lab,val in items:
                name = NAME_MAP.get(lab, lab)
                lines.append(f"{swatch(COLOR.get(lab,'#444'), DASH.get(lab,'dash'))}{name}: {val:,.0f}")
            if np.isfinite(nfin[i]): lines.append(f"{swatch(COLOR['NFIN'])}<b>{NAME_MAP['NFIN']}</b>: {nfin[i]:,.0f}")
            if np.isfinite(debt[i]): lines.append(f"{swatch(COLOR['DEBT'])}<b>{NAME_MAP['DEBT']}</b>: {debt[i]:,.0f}")
            texts.append("<br>".join(lines) if lines else "")
        return texts

    # =============================
    # Figure
    # =============================
    default_age = age_labels[0]
    px0 = curves_by_age[default_age]["px"]

    fig = go.Figure()

    # Tooltip driver
    helper_text = build_tooltip(default_age)
    fig.add_trace(go.Scatter(
        x=px0, y=curves_by_age[default_age]["nw"],
        mode="lines", line=dict(width=0),
        hovertemplate="%{text}<extra></extra>", text=helper_text,
        showlegend=False, name="_tooltip_helper_",
    ))

    # Master + top-level components (visible)
    for key in ["NW","FIN","NFIN","DEBT"]:
        name = NAME_MAP[key]
        y = curves_by_age[default_age]["nw" if key=="NW" else key.lower()]
        fig.add_trace(go.Scatter(
            x=px0, y=y, mode="lines",
            line=dict(width=3 if key=="NW" else 2, dash=DASH[key], shape="spline", color=COLOR[key]),
            name=name, hoverinfo="skip", connectgaps=False,
        ))

    # FIN subcomponents (start legendonly)
    for k in FIN_SUBS:
        lab = k.upper()
        fig.add_trace(go.Scatter(
            x=px0, y=curves_by_age[default_age]["fin_subs"][k],
            mode="lines",
            line=dict(width=2, dash=DASH.get(lab,"dash"), shape="spline", color=COLOR.get(lab,"#666")),
            name=NAME_MAP.get(lab, lab), hoverinfo="skip", connectgaps=False,
            visible="legendonly",
        ))

    # Dropdown: update x/y/text/name and y-range
    DECIMALS = decimals_from_step(STEP)
    buttons = []
    n_fixed = 1 + 4  # helper + (NW, FIN, NFIN, DEBT)
    for i_age, lbl in enumerate(age_labels):
        xs = []
        ys = []
        texts = []
        names = []

        # helper + NW/FIN/NFIN/DEBT
        xs.extend([curves_by_age[lbl]["px"]] * n_fixed)
        ys.extend([
            curves_by_age[lbl]["nw"],  # helper y
            curves_by_age[lbl]["nw"],
            curves_by_age[lbl]["fin"],
            curves_by_age[lbl]["nfin"],
            curves_by_age[lbl]["debt"],
        ])
        texts.append(build_tooltip(lbl))
        names.extend([
            "_tooltip_helper_",
            f"{lbl} – {NAME_MAP['NW']}",
            f"{lbl} – {NAME_MAP['FIN']}",
            f"{lbl} – {NAME_MAP['NFIN']}",
            f"{lbl} – {NAME_MAP['DEBT']}",
        ])

        # FIN subcomponents
        for k in FIN_SUBS:
            xs.append(curves_by_age[lbl]["px"])
            ys.append(curves_by_age[lbl]["fin_subs"][k])
            texts.append(None)
            names.append(f"{lbl} – {NAME_MAP.get(k.upper(), k.upper())}")

        lo_log, hi_log = y_ranges[i_age]
        buttons.append(dict(
            label=lbl,
            method="update",
            args=[
                {"x": xs, "y": ys, "text": texts, "name": names},
                {"title": None,
                 "yaxis": {"type":"log","autorange":False,"range":[2, hi_log]},
                 "xaxis": {"range":[X_START, X_END], "hoverformat": f".{DECIMALS}f"}}
            ],
        ))

    fig.update_layout(
        title=None,
        xaxis=dict(
            title="Percentile",
            range=[X_START, X_END],
            ticks="outside",
            hoverformat=f".{DECIMALS}f",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikethickness=1, spikedash="dot", spikecolor="gray",
        ),
        yaxis=dict(
            title="Value (log scale)",
            type="log", tickformat=",", exponentformat="none", showexponent="none",
            autorange=False, range=(2, y_ranges[0][1]),
        ),
        template="plotly_white",
        hovermode="x unified",
        uirevision="lock-y",
        showlegend=True,
        legend=dict(
            title="Toggle lines (click):<br><span style='font-weight:400'>(FIN Subcomponents start hidden)</span>",
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.7)", bordercolor="gray",
            itemclick="toggle", itemdoubleclick="toggleothers",
        ),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.02, y=1.02, xanchor="left", yanchor="bottom",
            showactive=True, buttons=buttons, pad=dict(r=0, t=0),
        )],
        margin=dict(l=70, r=70, t=40, b=60),
    )

    if fig.layout.updatemenus and len(fig.layout.updatemenus) > 0:
        fig.layout.updatemenus[0].active = 0

    return fig
