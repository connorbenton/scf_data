import numpy as np
import pandas as pd
import plotly.graph_objects as go

def get_figure() -> go.Figure:


    # =============================
    # User knobs (x-grid + smoothing)
    # =============================
    X_START = 1.0     # percentile range start (inclusive)
    X_END   = 99.0    # percentile range end   (inclusive if it lands on grid)
    STEP    = 0.2     # e.g., 0.2, 0.5, 1.0
    SMOOTH_WIN = 35    # smoothing window for SHARES only (not FIN)

    MASK_THRESH = 0.02   # per-bin masking threshold (2% of FIN)
    TOL         = 0.02   # re-scale visible subs to be within ±2% of FIN (we scale to exact)

    NAME_MAP = {
        "FIN": "Total financial assets",
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

    # =============================
    # Data load + weights
    # =============================
    df = pd.read_parquet("p22i6_with_concats.parquet")
    df.columns = [c.lower().strip() for c in df.columns]

    if "x42001" not in df.columns:
        raise KeyError("x42001 not found; required for weights.")
    df["x42001"] = pd.to_numeric(df["x42001"], errors="coerce").fillna(0)
    df = df[df["x42001"] > 0].copy()
    df["wgt"] = df["x42001"] / 5.0

    master = "fin"
    others = ["liq","cds","nmmf","stocks","bond","retqliq","savbnd","cashli","othma","othfin"]
    for c in [master] + others:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    if "x14" not in df.columns:
        raise KeyError("x14 (age) not found; required for age filter.")
    df["x14"] = pd.to_numeric(df["x14"], errors="coerce")

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
        """Return (centers_probs, edges_probs) as fractions in [0,1] on an exact step grid."""
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

    # =============================
    # FIN-aligned curves (true FIN percentiles, smoothed shares)
    # =============================
    def fin_aligned_curves_binned(
        df, components, age_mask=None, centers_probs=None, edges_probs=None, smooth_window=5
    ):
        """
        - FIN curve = true weighted percentile at centers_probs
        - Shares estimated from bin means using edges_probs
        - Smooth shares only (optional), then scale shares by FIN percentile
        """
        if centers_probs is None or edges_probs is None:
            raise ValueError("Provide centers_probs and edges_probs from make_percentile_grid().")

        base = df.copy()
        base["fin"] = pd.to_numeric(base["fin"], errors="coerce")
        base["wgt"] = pd.to_numeric(base["wgt"], errors="coerce")
        for c in components: base[c] = pd.to_numeric(base[c], errors="coerce")

        m = (base["wgt"] > 0) & np.isfinite(base["fin"])
        if age_mask is not None:
            m &= age_mask.reindex(base.index, fill_value=False)
        base = base.loc[m, ["fin","wgt"] + components].copy()
        if base.empty: return None

        # FIN percentile curve at requested centers
        fin_curve = weighted_percentiles(base["fin"].to_numpy(), base["wgt"].to_numpy(), centers_probs)

        # Convert edges (percentile probs) to *value* cutpoints for digitizing
        edge_vals = weighted_percentiles(base["fin"].to_numpy(), base["wgt"].to_numpy(), edges_probs)
        edge_vals = np.maximum.accumulate(edge_vals)

        n_bins = len(centers_probs)
        idx = np.digitize(base["fin"].to_numpy(), edge_vals, right=True) - 1
        idx = np.clip(idx, 0, n_bins - 1)

        fin_means  = np.full(n_bins, np.nan)
        comp_means = {c: np.full(n_bins, np.nan) for c in components}

        for b in range(n_bins):
            sel = (idx == b)
            if not np.any(sel): continue
            w = base.loc[sel, "wgt"]
            fbar = np.average(base.loc[sel, "fin"], weights=w)
            if not np.isfinite(fbar) or fbar <= 0: continue
            fin_means[b] = fbar
            for c in components:
                comp_means[c][b] = np.average(base.loc[sel, c], weights=w)

        shares = {c: comp_means[c] / fin_means for c in components}
        if smooth_window and smooth_window > 1:
            def smooth(a):
                return pd.Series(a, dtype="float64").rolling(
                    window=smooth_window, center=True, min_periods=1
                ).median().to_numpy()
            for c in components:
                shares[c] = smooth(shares[c])

        comp_curves = {c: shares[c] * fin_curve for c in components}
        out = {"fin": fin_curve, "probs": centers_probs}; out.update(comp_curves)
        return out

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
    for lbl in age_labels:
        res = fin_aligned_curves_binned(
            df=df, components=others, age_mask=age_mask(lbl),
            centers_probs=centers_probs, edges_probs=edges_probs,
            smooth_window=SMOOTH_WIN,
        )
        if res is None:
            px = np.arange(X_START, X_END + 1e-9, STEP)
            curves_by_age[lbl] = {"fin": np.full(len(px), np.nan), "px": px}
            for c in others: curves_by_age[lbl][c] = np.full(len(px), np.nan)
        else:
            px = res["probs"] * 100.0
            del res["probs"]
            res["px"] = px
            curves_by_age[lbl] = res

    # =============================
    # Refinement + 2% masking
    # =============================
    def refine_and_mask_for_age(label, mask_thresh=MASK_THRESH, tol=TOL):
        fin = np.asarray(curves_by_age[label]["fin"], float)
        px  = curves_by_age[label]["px"]
        comps = {c: np.asarray(curves_by_age[label][c], float).copy() for c in others}

        # 1) scale all comps to match FIN per bin
        for i,f in enumerate(fin):
            if not np.isfinite(f) or f<=0:
                for c in others: comps[c][i] = np.nan
                continue
            s = np.nansum([comps[c][i] for c in others])
            if np.isfinite(s) and s>0:
                scale = f / s
                for c in others:
                    if np.isfinite(comps[c][i]): comps[c][i] *= scale

        # 2) mask < threshold of FIN
        for i,f in enumerate(fin):
            if not np.isfinite(f) or f<=0:
                for c in others: comps[c][i] = np.nan
                continue
            cut = mask_thresh * f
            for c in others:
                y = comps[c][i]
                if (not np.isfinite(y)) or (y < cut): comps[c][i] = np.nan

        # 3) rescale visible to be within tol of FIN (exact if possible)
        for i,f in enumerate(fin):
            if not np.isfinite(f) or f<=0: continue
            vis = np.array([comps[c][i] for c in others], float)
            s = np.nansum(vis)
            if np.isfinite(s) and s>0:
                if abs(s - f)/f > tol:
                    scale = f / s
                    for c in others:
                        if np.isfinite(comps[c][i]): comps[c][i] *= scale

        out = {"px": px, "fin": fin}
        out.update(comps)
        # keep total_subs_refined internally if you ever want it again:
        out["total_subs_refined"] = np.nansum([out[c] for c in others], axis=0)
        return out

    refined_by_age = {lbl: refine_and_mask_for_age(lbl) for lbl in age_labels}

    # =============================
    # Tooltip (sorted with “swatches”) — FIXED colors
    # =============================
    # One canonical color + dash map used for BOTH traces and tooltip
    COLOR = {
        "FIN":"#1f77b4",
        "LIQ":"#d62728",
        "CDS":"#17becf",
        "NMMF":"#9467bd",
        "STOCKS":"#2ca02c",
        "BOND":"#7f7f7f",
        "RETQLIQ":"#8c564b",
        "SAVBND":"#e377c2",
        "CASHLI":"#bcbd22",
        "OTHMA":"#ff7f0e",
        "OTHFIN":"#1f77b4",
    }
    DASH = {
        "FIN":"solid",
        "LIQ":"dash","CDS":"dash","NMMF":"dash","STOCKS":"dash","BOND":"dash",
        "RETQLIQ":"dash","SAVBND":"dash","CASHLI":"dash","OTHMA":"dash","OTHFIN":"dash",
    }

    def swatch(color, dash="solid"):
        # Unicode-based swatch that renders reliably inside hover text
        if dash in ("dot","dotted"): glyph = "⋯⋯⋯"
        elif dash in ("dash","dashed"): glyph = "─ ─ ─"
        else: glyph = "━━━"
        return f"<span style='color:{color};font-weight:700'>{glyph}</span>&nbsp;"

    def round100(a):
        a = np.asarray(a, float)
        return np.where(np.isfinite(a), np.round(a/100.0)*100.0, np.nan)

    def build_sorted_tooltip_for_age(label, include_fin=True):
        fin  = round100(refined_by_age[label]["fin"])
        # values keyed exactly as trace names (UPPERCASE)
        vals = {c.upper(): round100(refined_by_age[label][c]) for c in others}

        n = len(refined_by_age[label]["px"])
        texts=[]
        for i in range(n):
            lines=[]
            if include_fin and np.isfinite(fin[i]):
                lines.append(f"{swatch(COLOR['FIN'],DASH['FIN'])}<b>{NAME_MAP['FIN']}</b>: {fin[i]:,.0f}")
            items = [(lab, vals[lab][i]) for lab in vals if np.isfinite(vals[lab][i])]
            items.sort(key=lambda x:-x[1])  # largest first
            for lab,val in items:
                # use COLOR[lab] and DASH[lab] to stay in sync with traces
                col = COLOR.get(lab, "#000000")
                dsh = DASH.get(lab, "solid")
                display_name = NAME_MAP.get(lab, lab)
                lines.append(f"{swatch(col, dsh)}<b>{display_name}</b>: {val:,.0f}")
            texts.append("<br>".join(lines) if lines else "")
        return texts

    # =============================
    # Figure
    # =============================
    def y_upper_for(label):
        m = np.nanmax(np.asarray(curves_by_age[label]["fin"], float))
        if not np.isfinite(m) or m<=0: m = 100.0
        return np.log10(m * 1.10)

    default_age = "All ages"
    px_default  = refined_by_age[default_age]["px"]
    initial_upper = y_upper_for(default_age)
    DECIMALS = decimals_from_step(STEP)

    fig = go.Figure()

    # Tooltip helper trace (drives sorted, swatched tooltip)
    helper_text = build_sorted_tooltip_for_age(default_age)
    fig.add_trace(
        go.Scatter(
            x=px_default, y=refined_by_age[default_age]["fin"],
            mode="lines", line=dict(width=0),
            hovertemplate="%{text}<extra></extra>", text=helper_text,
            showlegend=False, name="_tooltip_helper_",
        )
    )

    # FIN line (visible; no own hover)
    fig.add_trace(
        go.Scatter(
            x=px_default, y=refined_by_age[default_age]["fin"],
            mode="lines",
            line=dict(width=3, shape="spline", color=COLOR["FIN"]),
            name=NAME_MAP["FIN"],
            hoverinfo="skip",
        )
    )

    # Secondaries (masked per bin) — colors/dashes match tooltip maps
    for v in others:
        key = v.upper()
        name = NAME_MAP.get(key, key)
        fig.add_trace(
            go.Scatter(
                x=px_default, y=refined_by_age[default_age][v],
                mode="lines",
                line=dict(width=2, dash="dash", shape="spline", color=COLOR[key]),
                name=name, hoverinfo="skip", connectgaps=False,
            )
        )

    # Dropdown (update data, axis, tooltip)
    n_sec = len(others)
    age_dropdown_buttons=[]
    for lbl in age_labels:
        xs = [refined_by_age[lbl]["px"]] * (1 + 1 + n_sec)  # helper + FIN + components
        ys = [refined_by_age[lbl]["fin"],
            refined_by_age[lbl]["fin"]] + [refined_by_age[lbl][v] for v in others]
        new_text = build_sorted_tooltip_for_age(lbl)
        age_dropdown_buttons.append(
            dict(
                label=lbl, method="update",
                args=[
                    {"x": xs, "y": ys, "text": [new_text] + [None]*(len(ys)-1)},
                    # {"title": f"Weighted Percentile Curves — FIN + Components<br><sup>Age filter: {lbl}</sup>",
                    {"title": None,
                    "yaxis": {"type":"log","autorange":False,"range":[2, y_upper_for(lbl)]},
                    "xaxis": {"range":[X_START, X_END], "hoverformat": f".{DECIMALS}f"}},
                ],
            )
        )

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
            autorange=False, range=[2, initial_upper],
        ),
        uirevision="lock-y",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            title="Legend",
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.7)", bordercolor="gray",
            itemclick="toggle", itemdoubleclick="toggleothers",
        ),
        # ↓ move the dropdown down to the plot edge, remove extra padding
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.02, y=1.02,          # was 1.15
            xanchor="left", yanchor="bottom",  # was yanchor="top"
            showactive=True,
            buttons=age_dropdown_buttons,
            pad=dict(r=0, t=0),      # was {"r":10, "t":8}
        )],
        # ↓ shrink the top margin so there isn’t empty space above the plot
        margin=dict(l=70, r=70, t=40, b=60),   # was t=110
    )

    # fig.add_annotation(
    #     x=0, y=1.07, xref="paper", yref="paper", showarrow=False, align="left",
    #     text=f"X-grid step = {STEP}. Tooltip lists FIN, then subcategories sorted with matching swatches. Subcategories <2% of FIN are omitted per bin.",
    #     font=dict(size=12, color="gray"),
    # )

    # fig.show()
    if fig.layout.updatemenus and len(fig.layout.updatemenus) > 0:
        fig.layout.updatemenus[0].active = 0
    return fig
