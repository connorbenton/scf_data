import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------
# Load and clean data
# -----------------------------
df = pd.read_parquet("p22i6.parquet")

# Candidate columns
candidate_cols = ["x8022", "x104", "x110", "x116", "x122"]

# Include only columns present
available = [c for c in candidate_cols if c in df.columns]
if not available:
    raise ValueError("None of the requested columns found in p22i6.parquet.")

# Helper to clean a column: numeric only, remove NaN and 0
def clean_numeric(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s[s.notna()]
    s = s[s != 0]
    return s

# Clean all available columns
cleaned = {col: clean_numeric(df[col]) for col in available}

# Combined "All" column
combined_label = "All Combined"
combined_series = pd.concat(cleaned.values(), ignore_index=True)

options = available + [combined_label]

# -----------------------------
# Create histogram traces
# -----------------------------
traces = []
for i, opt in enumerate(options):
    x_vals = combined_series if opt == combined_label else cleaned[opt]
    traces.append(
        go.Histogram(
            x=x_vals,
            name=opt,
            nbinsx=30,  # adjust for bin granularity
            marker=dict(color="#1f77b4"),
            opacity=0.85,
            hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>",
            visible=(i == 0),  # show first trace initially
        )
    )

# -----------------------------
# Build figure
# -----------------------------
fig = go.Figure(data=traces)

# Dropdown controls
buttons = []
for i, opt in enumerate(options):
    vis = [False] * len(options)
    vis[i] = True
    buttons.append(
        dict(
            label=opt,
            method="update",
            args=[
                {"visible": vis},
                {"xaxis": {"title": "Value (Age, etc.)", "tickformat": ",", "showgrid": True},
                 "yaxis": {"title": "Raw Count", "tickformat": ",", "showgrid": True},
                 "title": f"Distribution — {opt} (Zeros Removed)"}
            ],
        )
    )

fig.update_layout(
    title=f"Distribution — {options[0]} (Zeros Removed)",
    template="plotly_white",
    barmode="overlay",
    xaxis=dict(title="Value (Age, etc.)", tickformat=",", showgrid=True),
    yaxis=dict(title="Raw Count", tickformat=",", showgrid=True),
    updatemenus=[
        dict(
            type="dropdown",
            direction="down",
            x=1.0, xanchor="right",
            y=1.15, yanchor="top",
            buttons=buttons,
            showactive=True,
        )
    ],
    margin=dict(l=70, r=40, t=70, b=60),
)

fig.show()
