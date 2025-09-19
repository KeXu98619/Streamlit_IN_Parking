# app.py — Indiana Truck Parking (finalized)
import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from streamlit_folium import st_folium
import folium
import altair as alt
import re
import base64

# -------- Password gate --------
def require_password():
    def _check():
        if st.session_state.get("pw_input", "") == st.secrets["APP_PASSWORD"]:
            st.session_state["authed"] = True
            st.session_state.pop("pw_input", None)
        else:
            st.session_state["authed"] = False

    if "authed" not in st.session_state or not st.session_state["authed"]:
        st.text_input("Password", type="password", key="pw_input", on_change=_check)
        if "authed" in st.session_state and st.session_state["authed"] is False:
            st.error("Incorrect password.")
        st.stop()

require_password()

st.set_page_config(page_title="Indiana Truck Parking -- County Dashboard", layout="wide")

# --- Global styles: fonts, icons, base UI, "cards" for chart & table, sidebar logo spacing ---
st.markdown("""
<!-- Inter (Google Fonts) -->
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">

<!-- Material Icons (legacy) -->
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<!-- Material Symbols (new) -->
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">

<style>
  /* Apply Inter broadly but do NOT clobber icon fonts */
  html, body, .stApp, .stMarkdown, .stTextInput, .stSelectbox, .stDataFrame, .stButton,
  .stCaption, .stDownloadButton, .stMetric, .stRadio, .stSelectbox div, .stSlider,
  .stCheckbox, .stNumberInput, .stText, .stHeader, h1, h2, h3, h4, h5, h6, p, span, label {
    font-family: 'Inter', sans-serif !important;
  }
  /* Keep icons rendering as icons (prevents 'keyboard_double_arrow_right' text) */
  .material-icons { font-family: 'Material Icons' !important; }
  .material-symbols-outlined { font-family: 'Material Symbols Outlined' !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; }

  /* Tighter table font */
  .stDataFrame table, .dataframe td, .dataframe th { font-size: 12px !important; }

  /* ---- "Card" look applied directly to the chart & dataframe containers ---- */
  /* Altair chart block */
  div[data-testid="stVegaLiteChart"] {
    background: #ffffff;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08), 0 2px 6px rgba(0,0,0,0.06);
    margin-bottom: 16px;
  }
  /* DataFrame block */
  div[data-testid="stDataFrame"] {
    background: #ffffff;
    border-radius: 16px;
    padding: 8px 8px 2px 8px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08), 0 2px 6px rgba(0,0,0,0.06);
    margin-bottom: 16px;
  }

  /* Sidebar: add bottom padding so a normal logo placed after Tip sits lower */
  [data-testid="stSidebar"] .block-container { padding-bottom: 120px; }
</style>
""", unsafe_allow_html=True)

# -------- Assets/paths --------
LOGO_PATH = None
for candidate in [Path("logo.webp"), Path("logo.png")]:
    if candidate.exists():
        LOGO_PATH = candidate
        break

DAILY_CSV = Path("indiana_county_daily_ver2.csv")
COUNTIES_GEOJSON = Path("indiana_counties_500k.geojson")
RAW_HOURLY_CSV = Path("in_parking_demand_data_ver2.xlsx")
SPOTS_GEOJSON = Path("IN_Truck_Spots.geojson")
ROADWAYS_GEOJSON = Path("in_roadway_map_layer.geojson")

# Palettes
PALETTE_5 = ["#e8edb8", "#bbe2c4", "#9bd4d0", "#7cc0db", "#61a1ca"]
PALETTE_4 = ["#e8edb8", "#bbe2c4", "#7cc0db", "#61a1ca"]
DIAG_PALETTE = {
    "High Stress":   "#61a1ca",
    "Elevated":      "#7cc0db",
    "Typical/Other": "#bbe2c4",
    "No Supply":     "#e8edb8",
}

# -------- Cached loaders --------
@st.cache_data(show_spinner=False)
def load_daily():
    return pd.read_csv(DAILY_CSV, dtype={"county_fips": str})

@st.cache_data(show_spinner=False)
def load_counties():
    gdf = gpd.read_file(COUNTIES_GEOJSON)
    gdf["county_fips"] = gdf["county_fips"].astype(str).str.zfill(5)
    return gdf

@st.cache_data(show_spinner=False)
def load_hourly():
    df = pd.read_excel(RAW_HOURLY_CSV, sheet_name='park_dem_calibrtd_by_hour')
    drop_cols = [c for c in ["county_name", "total_expanded_daily_parking_demand"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df.columns = ["county", "hour", "des_demand", "undes_demand", "supply"]
    df.columns = [c.strip().lower() for c in df.columns]
    df["county"] = df["county"].astype(str).str.zfill(5)
    df["hour"] = df["hour"].astype(int)
    for c in ["des_demand", "undes_demand", "supply"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

@st.cache_data(show_spinner=False)
def load_spots(path: Path):
    if not path.exists():
        return None, f"Spots file not found: {path}"
    try:
        gdf = gpd.read_file(path).to_crs(epsg=4326)
        gdf = gdf[gdf.geometry.notna() & gdf.geometry.geom_type.eq("Point")].copy()
        return gdf, None
    except Exception as e:
        return None, f"Could not read truck spots ({path.name}): {e}"

@st.cache_data(show_spinner=False)
def load_roadways(path: Path):
    if not path.exists():
        return None, f"Roadways file not found: {path}"
    try:
        gdf = gpd.read_file(path).to_crs(epsg=4326)
        gdf = gdf[gdf.geometry.notna() & gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
        return gdf, None
    except Exception as e:
        return None, f"Could not read roadways ({path.name}): {e}"

# -------- Map helpers --------
def make_base_map():
    """Leaflet base map with Mapbox if present; hide basemap from layer control."""
    m = folium.Map(location=[39.9, -86.3], zoom_start=7, tiles=None)
    token = st.secrets.get("MAPBOX_TOKEN")
    style = st.secrets.get("MAPBOX_STYLE", "mapbox/streets-v11")

    if token:
        folium.TileLayer(
            tiles=f"https://api.mapbox.com/styles/v1/{style}/tiles/256/{{z}}/{{x}}/{{y}}@2x?access_token={token}",
            attr="Mapbox", name="Basemap", control=False, max_zoom=20
        ).add_to(m)
    else:
        folium.TileLayer("cartodbpositron", name="Basemap", control=False).add_to(m)

    # Inter font inside the map iframe for tooltips
    m.get_root().header.add_child(folium.Element("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet" />
    <style>
      .leaflet-tooltip { font-size: 11px; opacity: 0.85; font-family: 'Inter', sans-serif; }
    </style>
    """))
    return m

def _quantile_edges(vals, q=(0, 0.25, 0.5, 0.75, 1.0)):
    """Robust quantile edges; fallback to equal intervals when degenerate."""
    vals = pd.Series(vals)
    try:
        edges = np.quantile(vals, q)
        edges = np.unique(np.round(edges, 6))
        if len(edges) < 2:
            raise ValueError
        return edges
    except Exception:
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmin == vmax:
            return np.array([vmin, vmax])  # constant
        return np.linspace(vmin, vmax, len(q))

def _bin_and_color_series(vals, palette5, palette4):
    """
    Returns: bin_idx (int series), colors (list), edges (ndarray), zero_heavy (bool)
    """
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    vmin, vmax = float(vals.min()), float(vals.max())
    zero_mask = vals.eq(0)
    zero_share = zero_mask.mean()
    pos_vals = vals[~zero_mask]

    # Case A: zero-heavy & positives -> zero bin + quantiles over positives
    if zero_share > 0.5 and not pos_vals.empty:
        pos_edges = _quantile_edges(pos_vals, q=(0, 0.25, 0.5, 0.75, 1.0))
        pos_edges = np.unique(pos_edges)
        if len(pos_edges) < 2:
            pos_edges = np.linspace(float(pos_vals.min()), float(pos_vals.max()), 5)

        pos_bins = pd.cut(vals, bins=pos_edges, include_lowest=True, labels=False, duplicates="drop").astype("float")
        bin_idx = pos_bins.add(1)  # shift
        bin_idx[zero_mask] = 0

        if not np.isnan(vmax):
            top = np.nanmax(bin_idx)
            bin_idx[vals == vmax] = top

        bin_idx = bin_idx.fillna(0)

        uniq_bins = sorted(pd.Series(bin_idx).unique())
        n_bins = max(1, len(uniq_bins))
        colors = (palette5 if n_bins >= 5 else palette4)[:n_bins]
        edges = np.concatenate(([0.0], pos_edges))
        return bin_idx.astype(int), colors, edges, True

    # Case B: standard quantiles over all values
    edges = _quantile_edges(vals, q=(0, 0.25, 0.5, 0.75, 1.0))
    edges = np.unique(edges)
    if len(edges) < 2:
        bin_idx = pd.Series(0, index=vals.index)
        colors = [palette4[0], palette4[0]]  # dummy
        return bin_idx.astype(int), colors, np.array([vmin, vmax]), False

    raw_bins = pd.cut(vals, bins=edges, include_lowest=True, labels=False, duplicates="drop").astype("float")
    top = np.nanmax(raw_bins)
    raw_bins[vals == vmax] = top
    bin_idx = raw_bins.fillna(0)

    uniq_bins = sorted(pd.Series(bin_idx).unique())
    n_bins = max(1, len(uniq_bins))
    colors = (palette5 if n_bins >= 5 else palette4)[:n_bins]
    return bin_idx.astype(int), colors, edges, False

def _fmt_compact(x: float) -> str:
    x = float(x)
    for unit in ["", "k", "M", "B", "T"]:
        if abs(x) < 1000.0:
            return f"{x:,.0f}{unit}"
        x /= 1000.0
    return f"{x:,.0f}P"

def _add_custom_legend(m, colors, vmin, vmed, vmax, title):
    """
    Adds a custom gradient legend (top-right) with only three labels: min | median | max.
    """
    # Build gradient CSS
    gradient = ",".join(colors)
    legend_html = f"""
    <div style="
      position: fixed; top: 24px; right: 24px; z-index: 9999;
      background: rgba(255,255,255,0.98); border: 1px solid #ddd;
      padding: 8px 10px; border-radius: 8px; font-family: 'Inter', sans-serif; font-size: 10px;">
      <div style="margin-bottom: 4px; font-weight: 600;">{title}</div>
      <div style="width: 240px; height: 12px; background: linear-gradient(90deg, {gradient});
                  border-radius: 4px;"></div>
      <div style="display: flex; justify-content: space-between; margin-top: 4px;">
        <span>{_fmt_compact(vmin)}</span>
        <span>{_fmt_compact(vmed)}</span>
        <span>{_fmt_compact(vmax)}</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

def make_numeric_choropleth(gdf_joined, color_col, legend_label):
    """Discrete (quantile/zero-aware) choropleth with a custom 3-label legend."""
    gdf = gdf_joined.copy()
    vals = gdf[color_col].values

    bin_idx, colors, edges, zero_heavy = _bin_and_color_series(vals, PALETTE_5, PALETTE_4)

    # Precompute hex color per feature
    gdf["_bin"] = bin_idx
    color_map = {i: colors[i] for i in range(len(colors))}
    gdf["_color"] = gdf["_bin"].map(color_map).fillna(colors[-1] if len(colors) else "#cccccc")

    def style_fn(feat):
        return {"fillColor": feat["properties"].get("_color", "#cccccc"),
                "color": "#555", "weight": 0.8, "fillOpacity": 0.8}

    m = make_base_map()
    folium.GeoJson(gdf, style_function=style_fn, name=legend_label).add_to(m)

    # Create a simple linear gradient from the discrete colors for the legend bar
    # (this is just for display; the map polygons use the same colors)
    vmin_all = float(np.nanmin(vals))
    vmed_all = float(np.nanmedian(vals))
    vmax_all = float(np.nanmax(vals))
    _add_custom_legend(m, colors, vmin_all, vmed_all, vmax_all, legend_label)

    return m

def make_categorical_map(gdf_joined, category_col, palette=None):
    palette = palette or DIAG_PALETTE
    m = make_base_map()

    def style_fn(feat):
        cat = feat["properties"].get(category_col, None)
        color = palette.get(cat, "#8c8c8c")
        return {"fillColor": color, "color": "#555", "weight": 0.8, "fillOpacity": 0.8}

    folium.GeoJson(gdf_joined, style_function=style_fn, name="Diagnosis").add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #ccc;">
      <b>Diagnosis</b><br>
    """
    for label, color in palette.items():
        legend_html += f'<span style="display:inline-block;width:12px;height:12px;background:{color};margin-right:6px;border:1px solid #666;"></span>{label}<br>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

def attach_tooltip_and_popup(m, gdf_joined):
    fields = [
        ("County", "county_name"),
        ("FIPS", "county_fips"),
        ("Supply (hourly fixed)", "supply_fmt"),
        ("Max hourly total demand", "max_hourly_total_demand_fmt"),
    ]
    tooltip = folium.features.GeoJsonTooltip(
        fields=[f for _, f in fields],
        aliases=[a for a, _ in fields],
        sticky=True, localize=True, labels=True,
        style=("background-color: rgba(255,255,255,0.9);"
               "border: 1px solid #ccc; border-radius: 4px; padding: 6px;"
               "box-shadow: 0 1px 3px rgba(0,0,0,0.2);")
    )
    gj = folium.GeoJson(
        gdf_joined,
        name="Counties",
        style_function=lambda _: {"fillOpacity": 0, "color": "#555", "weight": 0.8},
        highlight_function=lambda x: {"weight": 2, "color": "black"},
        tooltip=tooltip,
    )
    folium.GeoJsonPopup(fields=["county_fips"]).add_to(gj)
    gj.add_to(m)

def add_roadways_layer(m, road_gdf):
    if road_gdf is None or road_gdf.empty:
        return
    fg = folium.FeatureGroup(name="Roadways", show=True)
    folium.GeoJson(road_gdf, name="Roadways",
                   style_function=lambda _: {"color": "#4d4d4d", "weight": 1.0, "opacity": 0.8}).add_to(fg)
    fg.add_to(m)

def add_truck_spots_layer(m, spots_gdf):
    if spots_gdf is None or spots_gdf.empty:
        return
    fg = folium.FeatureGroup(name="Truck parking spots", show=True)
    for _, r in spots_gdf.iterrows():
        geom = r.geometry
        if geom and geom.geom_type == "Point":
            folium.CircleMarker(location=[geom.y, geom.x], radius=2.5, weight=0,
                                fill=True, fill_opacity=0.8).add_to(fg)
    fg.add_to(m)

# -------- UI --------
st.title("Indiana Truck Parking — County Dashboard")

metric_label_to_key = {
    "Max hourly designated demand": "max_hourly_des_demand",
    "Max hourly undesignated demand": "max_hourly_undes_demand",
    "Max hourly total demand": "max_hourly_total_demand",
    "Acc. designated demand (truck-hours)": "acc_des_demand",
    "Acc. undesignated demand (truck-hours)": "acc_undes_demand",
    "Acc. total demand (truck-hours)": "acc_total_demand",
    "Supply (hourly fixed)": "supply",
    "Max hourly designated deficit": "max_hourly_des_deficit",
    "Max hourly total deficit": "max_hourly_total_deficit",
    "Acc. designated deficit (truck-hours)": "acc_des_deficit",
    "Acc. total deficit (truck-hours)": "acc_total_deficit",
}
labels_numeric = list(metric_label_to_key.keys())

with st.sidebar:
    map_metric_label = st.selectbox(
        "Map: choose metric (or diagnosis)", options=["Diagnosis"] + labels_numeric, index=0
    )
    st.caption("Tip: Click a county to update the stacked hourly chart and the profile on the right.")
    # Put the logo under the Tip, with spacing
    if LOGO_PATH:
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        ext = LOGO_PATH.suffix[1:]
        b64 = base64.b64encode(open(LOGO_PATH, "rb").read()).decode("ascii")
        st.markdown(f"<img src='data:image/{ext};base64,{b64}' style='height:28px;'>", unsafe_allow_html=True)

# data
daily = load_daily()
counties = load_counties()
hourly = load_hourly()
spots_gdf, spots_err = load_spots(SPOTS_GEOJSON)
road_gdf, road_err = load_roadways(ROADWAYS_GEOJSON)

# join & fill
gdf_joined = counties.merge(daily, on="county_fips", how="left")
num_cols = [c for c in daily.columns if c not in ("diagnosis", "county_fips")]
for c in num_cols:
    if c in gdf_joined:
        gdf_joined[c] = pd.to_numeric(gdf_joined[c], errors="coerce").fillna(0)

# fmt columns for display
fmt_targets = [
    "max_hourly_des_demand", "max_hourly_undes_demand", "max_hourly_total_demand",
    "acc_des_demand", "acc_undes_demand", "acc_total_demand",
    "supply",
    "max_hourly_des_deficit", "max_hourly_total_deficit",
    "acc_des_deficit", "acc_total_deficit",
]
for col in fmt_targets:
    fmt_col = f"{col}_fmt"
    gdf_joined[fmt_col] = gdf_joined.get(col, 0).round(0).astype(int)

# notices
if spots_err: st.info(spots_err)
if road_err: st.info(road_err)

# defaults
if "selected_fips" not in st.session_state or not st.session_state.selected_fips:
    st.session_state.selected_fips = "18097"  # Marion
if "ignore_next_click" not in st.session_state:
    st.session_state.ignore_next_click = False

# layout
MAP_HEIGHT = 900  # adjust 880–920 to align with right panel height
col_map, col_right = st.columns([3, 2], gap="large")

with col_map:
    if map_metric_label == "Diagnosis":
        m = make_categorical_map(gdf_joined, "diagnosis")
    else:
        m = make_numeric_choropleth(
            gdf_joined, color_col=metric_label_to_key[map_metric_label], legend_label=map_metric_label
        )

    attach_tooltip_and_popup(m, gdf_joined)
    add_roadways_layer(m, road_gdf)
    add_truck_spots_layer(m, spots_gdf)

    folium.LayerControl(collapsed=False).add_to(m)
    map_state = st_folium(
        m, height=MAP_HEIGHT, use_container_width=True,
        returned_objects=["last_object_clicked_popup"]
    )

# pick up county clicks
if map_state and map_state.get("last_object_clicked_popup") and not st.session_state.ignore_next_click:
    raw = str(map_state["last_object_clicked_popup"])
    st.session_state.selected_fips = re.sub(r"\D", "", raw).zfill(5)
if st.session_state.ignore_next_click:
    st.session_state.ignore_next_click = False

# fips -> name helper
fips_to_name = dict(zip(gdf_joined["county_fips"], gdf_joined["county_name"]))

with col_right:
    # Title OUTSIDE card (above chart widget)
    title = fips_to_name.get(st.session_state.selected_fips, f"County {st.session_state.selected_fips}")
    st.markdown(f"### Hourly demand distribution — **{title}**")

    def hourly_long(df_hourly, fips=None):
        if fips:
            sub = df_hourly[df_hourly["county"] == fips].copy()
            supply_const = float(daily.loc[daily["county_fips"] == fips, "supply"].fillna(0).max())
        else:
            sub = df_hourly.copy()
            supply_const = float(daily["supply"].fillna(0).sum())

        agg = sub.groupby("hour", as_index=False)[["des_demand", "undes_demand"]].sum()
        agg["supply"] = supply_const
        long_df = agg.melt(
            id_vars="hour",
            value_vars=["des_demand", "undes_demand"],
            var_name="type", value_name="value"
        ).replace({"type": {"des_demand": "Designated", "undes_demand": "Undesignated"}})
        return long_df.sort_values("hour"), agg[["hour", "des_demand", "undes_demand", "supply"]]

    bars_long, hourly_table = hourly_long(hourly, st.session_state.selected_fips)
    bars_long["type_order"] = bars_long["type"].map({"Designated": 0, "Undesignated": 1})

    stacked = (
        alt.Chart(bars_long)
          .mark_bar()
          .encode(
              x=alt.X("hour:O", title="Hour of day",
                      axis=alt.Axis(labelAngle=0, labelOverlap=True, titlePadding=12)),
              y=alt.Y("sum(value):Q", title="Demand (truck-hours)",
                      axis=alt.Axis(format=",.0f")),
              color=alt.Color("type:N", title="",
                              scale=alt.Scale(domain=["Designated","Undesignated"]),
                              sort=["Designated","Undesignated"]),
              order=alt.Order("type_order:Q"),
              tooltip=[alt.Tooltip("hour:O", title="Hour"),
                       alt.Tooltip("type:N", title="Type"),
                       alt.Tooltip("sum(value):Q", title="Demand", format=",.0f")]
          )
          .properties(height=320)
    )

    # Yellow supply rule with friendly tooltip (thicker)
    supply_const = float(hourly_table["supply"].iloc[0]) if not hourly_table.empty else 0.0
    rule_df = pd.DataFrame({"y": [supply_const], "label": [f"Supply {supply_const:,.0f}"]})
    rule = (
        alt.Chart(rule_df)
          .mark_rule(color="#e8edb8", size=4)
          .encode(y="y:Q", tooltip=alt.Tooltip("label:N", title=""))
    )

    chart = (stacked + rule) \
            .configure_axis(labelFont="Inter", titleFont="Inter") \
            .configure_legend(labelFont="Inter", titleFont="Inter")

    st.altair_chart(chart, use_container_width=True)

    # County profile
    st.markdown("### County profile")
    profile_fields = [
        ("County", "county_name"),
        ("FIPS", "county_fips"),
        ("Diagnosis", "diagnosis"),
        ("Max hourly designated demand", "max_hourly_des_demand_fmt"),
        ("Max hourly undesignated demand", "max_hourly_undes_demand_fmt"),
        ("Max hourly total demand", "max_hourly_total_demand_fmt"),
        ("Acc. designated demand (truck-hrs)", "acc_des_demand_fmt"),
        ("Acc. undesignated demand (truck-hrs)", "acc_undes_demand_fmt"),
        ("Acc. total demand (truck-hrs)", "acc_total_demand_fmt"),
        ("Supply (hourly fixed)", "supply_fmt"),
        ("Max hourly designated deficit", "max_hourly_des_deficit_fmt"),
        ("Max hourly total deficit", "max_hourly_total_deficit_fmt"),
        ("Acc. designated deficit (truck-hrs)", "acc_des_deficit_fmt"),
        ("Acc. total deficit (truck-hrs)", "acc_total_deficit_fmt"),
    ]

    def county_profile(gdf, fips):
        row = gdf[gdf["county_fips"] == fips].head(1)
        if row.empty:
            return pd.DataFrame({"Metric": [], "Value": []})
        items = [(label, row.iloc[0].get(col, "")) for label, col in profile_fields]
        return pd.DataFrame(items, columns=["Metric", "Value"])

    profile_df = county_profile(gdf_joined, st.session_state.selected_fips)
    st.dataframe(profile_df, hide_index=True, use_container_width=True)

with st.expander("Metrics & diagnosis"):
    st.markdown(r"""
**Daily metrics (per county)** shown in tooltips & map selector:

- **Max hourly designated demand** - highest designated count in any hour  
- **Max hourly undesignated demand** - highest undesignated count in any hour  
- **Max hourly total demand** - highest (designated + undesignated) in any hour  
- **Acc. designated demand (truck-hours)** - sum of designated across 24 hours  
- **Acc. undesignated demand (truck-hours)** - sum of undesignated across 24 hours  
- **Acc. total demand (truck-hours)** - sum of (designated + undesignated) across 24 hours  
- **Supply (hourly fixed)** - available designated stalls (capacity)  
- **Max hourly designated deficit** - max(0, designated - supply) over 24 hours  
- **Max hourly total deficit** - max(0, total - supply) over 24 hours  
- **Acc. designated deficit (truck-hours)** - sum(max(0, designated - supply))  
- **Acc. total deficit (truck-hours)** - sum(max(0, total - supply))

**Diagnosis rules (per county):**
- **High Stress** — Total demand hours ≥ 1000 and either (max hourly designated demand ÷ supply ≥ 0.9) or (undesigned share > 0.5).  
- **Elevated** — Not High Stress, and total demand hours ≥ 300 and either (max hourly designated demand ÷ supply ≥ 0.7) or (undesigned share > 0.2).  
- **Typical/Other** — All others (i.e., not High Stress, not Elevated, not No Supply).  
- **No Supply** — Not High Stress, not Elevated, and supply = 0 parking spaces.  
""")
