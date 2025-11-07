import numpy as np
import pandas as pd
import plotly.graph_objects as go

def get_figure() -> go.Figure:
    # ---------- helpers ----------
    def round_to_100(a):
        a = np.asarray(a, float)
        return np.where(np.isfinite(a), np.round(a/100.0)*100.0, np.nan)

    def fmt100(a):
        """Return array of strings rounded to $100 with thousands separators."""
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

    def income_edges(inc, w, qs=(0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)):
        edges = weighted_quantiles(inc, w, np.array(qs, float))
        edges = np.asarray(edges, float)
        for i in range(1, len(edges)):
            if not np.isfinite(edges[i]) or edges[i] <= edges[i-1]:
                edges[i] = np.nextafter(edges[i-1], np.inf)
        return edges

    def mk_trace(name, px, vy, style, visible=True):
        # Tooltip uses %{text}; we update text on age change
        return go.Scatter(
            x=px, y=vy, mode="lines", name=name, line=style,
            text=fmt100(vy),
            hovertemplate="Percentile %{x:.1f}<br>%{text}<extra></extra>",
            visible=visible, showlegend=True
        )

    # ---------- load ----------
    df = pd.read_parquet("p22i6_with_concats.parquet")
    for c in ["networth", "income", "x14", "wgt"]:
        if c not in df.columns:
            raise KeyError(f"Missing column {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["networth", "income", "x14", "wgt"])
    df = df[df["wgt"] > 0].copy()

    age_labels = ["All ages", "<35", "35–44", "45–54", "55–64", "65–74", "≥75"]
    inc_labels = ["<20", "20–39.9", "40–59.9", "60–79.9", "80–89.9", "90–100"]

    styles = {
        "All incomes": dict(width=3),
        "<20":         dict(width=2, dash="dot"),
        "20–39.9":     dict(width=2, dash="dash"),
        "40–59.9":     dict(width=2, dash="longdash"),
        "60–79.9":     dict(width=2, dash="dashdot"),
        "80–89.9":     dict(width=2, dash="longdashdot"),
        "90–100":      dict(width=2),
    }

    # ---------- precompute curves per age/bin + y ranges ----------
    curves_by_age = {}
    y_ranges = []  # list of (log10 ymin, log10 ymax) per age
    x14 = df["x14"].to_numpy()

    for lbl in age_labels:
        m = age_mask(x14, lbl)
        sub = df.loc[m].copy()

        # default storage
        curves_by_age[lbl] = {"All incomes": (np.array([]), np.array([]))}
        for lab in inc_labels:
            curves_by_age[lbl][lab] = (np.array([]), np.array([]))

        if sub.empty:
            y_ranges.append((0.0, 2.0))  # 1 .. 100
            continue

        # master (all incomes)
        px_all, vy_all = weighted_percentile_curve(
            sub["networth"].to_numpy(), sub["wgt"].to_numpy(), 0.0, 0.995
        )
        curves_by_age[lbl]["All incomes"] = (px_all, vy_all)

        # income bins
        edges = income_edges(sub["income"].to_numpy(), sub["wgt"].to_numpy())
        sub["inc_bin"] = pd.cut(
            sub["income"], bins=edges, include_lowest=True, right=True, labels=inc_labels
        )
        for lab in inc_labels:
            s = sub[sub["inc_bin"] == lab]
            if not s.empty:
                px, vy = weighted_percentile_curve(
                    s["networth"].to_numpy(), s["wgt"].to_numpy(), 0.0, 0.995
                )
                curves_by_age[lbl][lab] = (px, vy)

        # dynamic y-range per age: include master + all bins
        all_ys = [vy_all] + [curves_by_age[lbl][lab][1] for lab in inc_labels]
        pos_vals = np.concatenate([y[np.isfinite(y) & (y > 0)] for y in all_ys]) if len(all_ys) else np.array([])
        if pos_vals.size == 0:
            ymin, ymax = 1.0, 100.0
        else:
            ymin = float(np.nanmin(pos_vals)) / 1.2  # pad a little
            ymax = float(np.nanpercentile(pos_vals, 99.5)) * 1.10  # stable upper
            ymin = max(ymin, 1e-2)  # guard for log
            ymax = max(ymax, 10.0)
        y_ranges.append((np.log10(ymin), np.log10(ymax)))

    # ---------- build 7 traces (master + 6 bins) ----------
    default_age = age_labels[0]
    fig = go.Figure()

    # master visible
    px_m, vy_m = curves_by_age[default_age]["All incomes"]
    fig.add_trace(mk_trace(f"{default_age} – All incomes", px_m, vy_m, styles["All incomes"], visible=True))

    # bins in legend, start legend-only so users can toggle them on
    for lab in inc_labels:
        px, vy = curves_by_age[default_age][lab]
        t = mk_trace(f"{default_age} – {lab}", px, vy, styles[lab], visible=True)
        t.visible = "legendonly"  # <- key: clickable legend item, line initially hidden
        fig.add_trace(t)

    # ---------- dropdown: update x/y/text/name and y-range ----------
    age_dropdown_buttons = []
    for i_age, lbl in enumerate(age_labels):
        xs, ys, texts, names = [], [], [], []

        # master first
        x0, y0 = curves_by_age[lbl]["All incomes"]
        xs.append(x0); ys.append(y0); texts.append(fmt100(y0)); names.append(f"{lbl} – All incomes")

        # then bins
        for lab in inc_labels:
            x, y = curves_by_age[lbl][lab]
            xs.append(x); ys.append(y); texts.append(fmt100(y)); names.append(f"{lbl} – {lab}")

        lo_log, hi_log = y_ranges[i_age]
        age_dropdown_buttons.append(dict(
            label=lbl,
            method="update",
            args=[
                {"x": xs, "y": ys, "text": texts, "name": names},
                {"yaxis": {"type": "log", "autorange": False, "range": [lo_log, hi_log]}}
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
            autorange=False, range=list(y_ranges[0]),
        ),
        template="plotly_white",
        hovermode="x unified",
        uirevision="lock-y",
        showlegend=True,
        legend=dict(
            title="Toggle income lines (click):<br><span style='font-weight:400'>(Master line = All incomes)</span>",
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
