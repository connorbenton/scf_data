import importlib.util
import sys
import plotly.io as pio

# ------------------------------
# Paths to your 3 Python figure files
# ------------------------------
FIG_PATHS = {
    "NETWORTH": "plotly NETWORTH percentile graph_age filter.py",
    "FIN":  "final FIN percentile graph age filter correct weights crosscheck OK tails.py",
    "DEBT": "final DEBT percentile graph age filter correct weights crosscheck OK tails.py",
    "NFIN": "final NFIN percentile graph age filter correct weights crosscheck OK tails.py",
}

# ------------------------------
# Load module and get figure JSON
# ------------------------------
def load_module_from_file(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def get_fig_json(path, module_name):
    mod = load_module_from_file(path, module_name)
    if not hasattr(mod, "get_figure"):
        raise AttributeError(f"{path} must define get_figure() returning a Plotly Figure.")
    fig = mod.get_figure()

    # Ensure “All ages” is selected in saved layout
    try:
        if fig.layout.updatemenus and len(fig.layout.updatemenus) > 0:
            fig.layout.updatemenus[0].active = 0
    except Exception:
        pass

    return pio.to_json(fig, pretty=False)

fig_json = {k: get_fig_json(v, f"mod_{k.lower()}") for k, v in FIG_PATHS.items()}

# ------------------------------
# Build the combined HTML dashboard
# ------------------------------
html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Assets & Debt Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  :root {{
    --page-pad: 28px;
    --gap: 12px;
    --bottom-pad: 28px;
  }}
  body {{
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    margin: var(--page-pad);
    background: #fafafa;
  }}
  h2 {{ margin: 0 0 8px 0; }}
  .tabs {{ margin-bottom: var(--gap); }}
  .tab {{
    display: inline-block; padding: 10px 18px; margin-right: 6px;
    background: #e5e5e5; border-radius: 6px 6px 0 0; cursor: pointer; font-weight: 600;
    user-select: none;
  }}
  .tab.active {{ background: #fff; border-bottom: 2px solid #fff; }}
  .chart-frame {{
    background: #fff; border: 1px solid #ddd; border-radius: 0 6px 6px 6px;
    padding: 12px;
  }}
  #plot {{
    width: 100%;
    min-height: 420px;
  }}
</style>
</head>
<body>

<h2>Assets &amp; Debt Dashboard</h2>

<div class="tabs">
  <div class="tab active" data-key="NETWORTH">Total Net Worth (by Age and Income)</div>
  <div class="tab" data-key="FIN">FIN (Financial assets breakdown)</div>
  <div class="tab" data-key="NFIN">NFIN (Nonfinancial assets breakdown)</div>
  <div class="tab" data-key="DEBT">DEBT (Liabilities breakdown)</div>
</div>

<div class="chart-frame">
  <div id="plot"></div>
</div>

<!-- Embed figure JSON safely -->
<script id="fig-NETWORTH"  type="application/json">{fig_json["NETWORTH"]}</script>
<script id="fig-FIN"  type="application/json">{fig_json["FIN"]}</script>
<script id="fig-DEBT" type="application/json">{fig_json["DEBT"]}</script>
<script id="fig-NFIN" type="application/json">{fig_json["NFIN"]}</script>

<script>
(function() {{

  // ---- Registry holds the ORIGINAL JSON, never mutated ----
  var registry = {{
    NETWORTH: JSON.parse(document.getElementById('fig-NETWORTH').textContent),
    FIN:      JSON.parse(document.getElementById('fig-FIN').textContent),
    DEBT:     JSON.parse(document.getElementById('fig-DEBT').textContent),
    NFIN:     JSON.parse(document.getElementById('fig-NFIN').textContent)
  }};

  // Return a deep-cloned figure so Plotly/react/legend changes never mutate registry
  function freshFigure(key) {{
    return JSON.parse(JSON.stringify(registry[key]));
  }}

  var plotDiv = document.getElementById('plot');
  var currentKey = 'NETWORTH';
  var renderTicket = 0; // cancels stale renders

  function fitHeight() {{
    var frame = document.querySelector('.chart-frame');
    if (!frame) return;
    var rect = frame.getBoundingClientRect();
    var available = window.innerHeight - rect.top - 28;
    var h = Math.max(420, Math.floor(available));
    plotDiv.style.height = h + 'px';
    try {{ Plotly.relayout(plotDiv, {{ height: h }}); }} catch (e) {{}}
  }}

  // Replay the first dropdown button (apply its visible array + layout)
  function applyFirstDropdown(fig) {{
    try {{
      var menus = (fig.layout && fig.layout.updatemenus) ? fig.layout.updatemenus : null;
      if (!menus || !menus.length) return;
      var btn0 = menus[0].buttons && menus[0].buttons[0];
      if (!btn0 || !btn0.args || !btn0.args.length) return;

      var restyle  = btn0.args[0] || {{}};
      var relayout = btn0.args[1] || {{}};

      var allTraces = (fig.data || []).map(function(_, i) {{ return i; }});
      Plotly.update(plotDiv, restyle, relayout, allTraces);
      Plotly.relayout(plotDiv, {{ 'updatemenus[0].active': 0 }});
    }} catch (e) {{
      console.warn('applyFirstDropdown error:', e);
    }}
  }}

  // Ensure at least one trace is actually drawn (treat 'legendonly' as not visible)
  function ensureDefaultVisible() {{
    try {{
      var gdData = plotDiv.data || [];
      var hasShown = gdData.some(function(t) {{ return t && t.visible === true; }});
      if (hasShown) return;

      var masterIdx = gdData.findIndex(function(t) {{
        var nm = (t && t.name) ? String(t.name) : '';
        return nm.includes('All ages') && nm.includes('All incomes');
      }});
      if (masterIdx < 0) {{
        masterIdx = gdData.findIndex(function(t) {{
          return t && Array.isArray(t.x) && t.x.length && Array.isArray(t.y) && t.y.length;
        }});
      }}
      if (masterIdx >= 0) {{
        Plotly.restyle(plotDiv, {{ visible: true }}, [masterIdx]);
      }}
    }} catch (e) {{
      console.warn('ensureDefaultVisible error:', e);
    }}
  }}

  function activateTabHeader(key) {{
    var tabs = document.querySelectorAll('.tab');
    for (var i = 0; i < tabs.length; i++) tabs[i].classList.remove('active');
    var head = document.querySelector('.tab[data-key="' + key + '"]');
    if (head) head.classList.add('active');
  }}

  var clickTimer = null;
  function queueShow(key) {{
    if (clickTimer) window.clearTimeout(clickTimer);
    clickTimer = window.setTimeout(function() {{ show(key); }}, 80);
  }}

  async function show(key) {{
    currentKey = key;
    activateTabHeader(key);
    var myTicket = ++renderTicket;

    // Deep-clone the figure to avoid cross-tab mutations
    var fig = freshFigure(key);

    // Start from a clean graph div
    try {{ Plotly.purge(plotDiv); }} catch (e) {{}}

    await Plotly.react(
      plotDiv,
      fig.data,
      fig.layout,
      {{ displaylogo: false, responsive: true }}
    );

    if (myTicket !== renderTicket) return;

    // Apply the figure's own default dropdown state, then guarantee visibility
    applyFirstDropdown(fig);
    if (myTicket !== renderTicket) return;

    ensureDefaultVisible();
    if (myTicket !== renderTicket) return;

    fitHeight();
    try {{ Plotly.Plots.resize(plotDiv); }} catch (e) {{}}
  }}

  // Tabs wiring
  var tabs = document.querySelectorAll('.tab');
  for (var i = 0; i < tabs.length; i++) {{
    (function(tab) {{
      tab.addEventListener('click', function() {{
        var key = tab.getAttribute('data-key');
        if (key && key !== currentKey) queueShow(key);
      }});
    }})(tabs[i]);
  }}

  window.addEventListener('resize', fitHeight);

  // Initial render
  show('NETWORTH');

}})();
</script>

</body>
</html>
"""

# ------------------------------
# Write to file
# ------------------------------
with open("combined_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("✅ Dashboard written to combined_dashboard.html")
