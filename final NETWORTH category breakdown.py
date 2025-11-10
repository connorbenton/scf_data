import numpy as np
import pandas as pd
import plotly.graph_objects as go

def get_figure() -> go.Figure:
    # ---------- helpers ----------
    def round_to_100(a):
        a = np.asarray(a, float)
        return np.where(np.isfinite(a), np.round(a/100.0)*100.0, np.nan)

    def fmt100(a):
        """Array of strings rounded to $100 with thousands separators ('' if NaN)."""
        v = round_to_100(a)
        return [f"{x:,.0f}" if np.isfinite(x) else "" for x in v]

    def weighted_quantiles(values, weights, qs):
        values = np.asarray(values, float)
        weights = np.asarray(weights, float)
        qs = np.asarray(qs, float)

        m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not np.any(m):
            return np.full_like(qs, np.nan, float)

        order = np.argsort(values[m])
        v = values[m][order]
        w = weights[m][order]
        cw = np.cumsum(w)
        if cw.size == 0 or cw[-1] <= 0:
            return np.full_like(qs, np.nan, float)

        p = (cw - 0.5 * w) / cw[-1]
        p = np.maximum.accumulate(p + np.arange(p.size) * 1e-16)
        return np.interp(qs, p, v)

    def weighted_percentile_curve(values, weights, p_min=0.0, p_max=0.995):
        """Return (px[%], vy) for weighted quantiles on a dense percentile grid."""
        m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        v = np.asarray(values, float)[m]
        w = np.asarray(weights, float)[m]
        if v.size == 0:
            return np.array([]), np.array([])
        px = np.linspace(0, 100, 401)
        qs = np.clip(px / 100.0, p_min, p_max)
        vy = weighted_quantiles(v, w, qs)
        keep = np.isfinite(vy) & (vy > 0)
        return px[keep], vy[keep]

    def age_mask(a, label):
        if label == "All ages": return np.ones(a.size, bool)
        if label == "<35":      return a < 35
        if label == "35–44":    return (a >= 35) & (a <= 44)
        if label == "45–54":    return (a >= 45) & (a <= 54)
        if label == "55–64":    return (a >= 55) & (a <= 64)
        if label == "65–74":    return (a >= 65) & (a <= 74)
        if label == "≥75":      return a >= 75
        raise ValueError("bad age label")

    def mk_trace(name, px, vy, style, visible=True):
        return go.Scatter(
            x=px, y=vy, mode="lines", name=name, line=style,
            text=fmt100(vy),
            hovertemplate="Percentile %{x:.1f}<br>%{text}<extra></extra>",
            visible=visible, showlegend=True
        )

    # ---------- load ----------
    df = pd.read_parquet("p22i6_with_concats.parquet")
    # Required columns
    req = {
        "networth", "x14", "wgt", "x8023",
        "x108","x114","x120","x126","x132","x202","x208","x214","x220","x226",
        "x508","x601","x701","x7133"
    }
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Coerce numerics used below
    for c in req | {"income"} if "income" in df.columns else req:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic cleaning
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["networth","x14","wgt"])
    df = df[df["wgt"] > 0].copy()

    # ---------- household / kids masks ----------
    # Married or Living With Partner
    married_lwp = df["x8023"].isin([1, 2])

    # Any child present if ANY of these equals 4
    kid_cols = ["x108","x114","x120","x126","x132","x202","x208","x214","x220","x226"]
    kids_any = (df[kid_cols].to_numpy() == 4).any(axis=1)

    # Residence ownership per provided rule
    owns_res = (
        df["x508"].isin([1, 2]) |
        df["x601"].isin([1, 2, 3]) |
        df["x701"].isin([1, 3, 4, 5, 6, 8]) |
        ((df["x701"] == -7) & (df["x7133"] == 1))
    ).fillna(False)

    # Category definitions
    cat_defs = [
        ("Not married/doesn't live with partner, no kids",  (~married_lwp) & (~kids_any)),
        ("Not married/doesn't LWP, with kids",(~married_lwp) & (kids_any)),
        ("Married/LWP, no kids",      (married_lwp) & (~kids_any)),
        ("Married/LWP, with kids",    (married_lwp) & (kids_any)),
        ("Owns residence",            owns_res),
        ("Doesn't own residence",     ~owns_res),
    ]

    # ---------- display setup ----------
    age_labels = ["All ages","<35","35–44","45–54","55–64","65–74","≥75"]

    styles = {
        "All":               dict(width=3),
        "Not married/doesn't live with partner, no kids":   dict(width=2, dash="dot"),
        "Not married/doesn't LWP, with kids": dict(width=2, dash="dash"),
        "Married/LWP, no kids":       dict(width=2, dash="longdash"),
        "Married/LWP, with kids":     dict(width=2, dash="dashdot"),
        "Owns residence":             dict(width=2, dash="longdashdot"),
        "Doesn't own residence":      dict(width=2),
    }

    # ---------- precompute curves per age/category + y ranges ----------
    curves_by_age = {}
    y_ranges = []  # per-age (log10 ymin, log10 ymax)

    a = df["x14"].to_numpy()
    for lbl in age_labels:
        m_age = age_mask(a, lbl)
        sub = df.loc[m_age].copy()

        # init dict
        curves_by_age[lbl] = {}
        if sub.empty:
            # placeholders
            px_all, vy_all = np.array([]), np.array([])
            curves_by_age[lbl]["All"] = (px_all, vy_all)
            for nm, _ in cat_defs:
                curves_by_age[lbl][nm] = (np.array([]), np.array([]))
            y_ranges.append((0.0, 2.0))  # log10(1) .. log10(100)
            continue

        # master (All)
        px_all, vy_all = weighted_percentile_curve(sub["networth"].to_numpy(),
                                                   sub["wgt"].to_numpy(), 0.0, 0.995)
        curves_by_age[lbl]["All"] = (px_all, vy_all)

        # categories
        all_ys = [vy_all]
        for nm, m_cat in cat_defs:
            s = sub.loc[m_cat.reindex(sub.index, fill_value=False)]
            if s.empty:
                curves_by_age[lbl][nm] = (np.array([]), np.array([]))
            else:
                px, vy = weighted_percentile_curve(s["networth"].to_numpy(),
                                                   s["wgt"].to_numpy(), 0.0, 0.995)
                curves_by_age[lbl][nm] = (px, vy)
                all_ys.append(vy)

        # dynamic y-range per age (include master + all cats)
        pos_vals = np.concatenate([y[np.isfinite(y) & (y > 0)] for y in all_ys]) \
                   if len(all_ys) else np.array([])

        if pos_vals.size == 0 or (vy_all.size == 0):
            # Keep bottom fixed at 10^2 = 100; degenerate upper bound if no data
            y_ranges.append((2.0, 2.0))
        else:
            # hard_max = true max of the master NW curve for this age group
            hard_max = float(np.nanmax(vy_all))
            # soft_max = robust cap across all displayed series (p99.9)
            soft_max = float(np.nanpercentile(pos_vals, 99.9))
            ymax = max(hard_max, soft_max) * 1.25
            # Keep bottom fixed at 10^2 = 100
            y_ranges.append((2.0, np.log10(ymax)))

    # ---------- build fixed traces: 1 master + 6 categories ----------
    default_age = age_labels[0]
    fig = go.Figure()

    # master visible
    x0, y0 = curves_by_age[default_age]["All"]
    fig.add_trace(mk_trace(f"{default_age} – All", x0, y0, styles["All"], visible=True))

    # categories appear as legend items; start as legend-only
    for nm, _ in cat_defs:
        x, y = curves_by_age[default_age][nm]
        tr = mk_trace(f"{default_age} – {nm}", x, y, styles[nm], visible=True)
        tr.visible = "legendonly"  # clickable immediately
        fig.add_trace(tr)

    # ---------- dropdown: update x/y/text/name and y-range ----------
    age_dropdown_buttons = []
    for i_age, lbl in enumerate(age_labels):
        xs, ys, texts, names = [], [], [], []

        # master first
        x_all, y_all = curves_by_age[lbl]["All"]
        xs.append(x_all); ys.append(y_all); texts.append(fmt100(y_all)); names.append(f"{lbl} – All")

        # categories (in same order as added)
        for nm, _ in cat_defs:
            x, y = curves_by_age[lbl][nm]
            xs.append(x); ys.append(y); texts.append(fmt100(y)); names.append(f"{lbl} – {nm}")

        lo_log, hi_log = y_ranges[i_age]
        age_dropdown_buttons.append(dict(
            label=lbl,
            method="update",
            args=[
                {"x": xs, "y": ys, "text": texts, "name": names},
                {"yaxis": {"type": "log", "autorange": False, "range": [2, hi_log]}}
            ]
        ))

    # ---------- layout ----------
    fig.update_layout(
        title=None,
        xaxis=dict(
            title="Percentile",
            range=[0, 100],
            ticks="outside",
            hoverformat=".1f",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikethickness=1, spikedash="dot", spikecolor="gray",
        ),
        yaxis=dict(
            title="NET WORTH (log scale)",
            type="log", tickformat=",", exponentformat="none", showexponent="none",
            autorange=False, range=(2, y_ranges[0][1]),
        ),
        template="plotly_white",
        hovermode="x unified",
        uirevision="lock-y",
        showlegend=True,
        legend=dict(
            title="Toggle breakdown lines (click):<br><span style='font-weight:400'>(Master line = All)</span>",
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.7)", bordercolor="gray",
            itemclick="toggle", itemdoubleclick="toggleothers",
        ),
        updatemenus=[dict(
            type="dropdown", direction="down",
            x=0.02, y=1.02, xanchor="left", yanchor="bottom",
            showactive=True, buttons=age_dropdown_buttons, pad=dict(r=0, t=0),
        )],
        margin=dict(l=70, r=70, t=40, b=60),
    )

    # default selection highlight
    if fig.layout.updatemenus and len(fig.layout.updatemenus) > 0:
        fig.layout.updatemenus[0].active = 0

    return fig
