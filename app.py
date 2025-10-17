# app.py — Indiana Truck Parking (final)
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

# --- Global styles: Inter font, protect icon fonts, card look, logo pin, legend ---
st.markdown("""
<!-- Inter (Google Fonts) -->
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<!-- Material Icons / Symbols -->
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">

<style>
  /* Apply Inter broadly (but NOT to generic <span>, to avoid clobbering icon spans) */
  html, body, .stApp, .stMarkdown, .stTextInput, .stSelectbox, .stDataFrame, .stButton,
  .stCaption, .stDownloadButton, .stMetric, .stRadio, .stSlider, .stCheckbox,
  .stNumberInput, .stText, .stHeader, h1, h2, h3, h4, h5, h6, p, label, div {
    font-family: 'Inter', sans-serif !important;
  }
  /* Force any Material icon/symbol classes to render as icons (not text) */
  .material-icons,
  .material-icons-outlined,
  .material-icons-round,
  .material-icons-sharp,
  .material-icons-two-tone,
  .material-symbols-outlined,
  [class^="material-"],
  [class*=" material-"] {
    font-family: 'Material Symbols Outlined','Material Icons' !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    font-style: normal; font-weight: normal; letter-spacing: normal;
    text-transform: none; display: inline-block; white-space: nowrap;
    line-height: 1; -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
  }

  /* Tighter table font */
  .stDataFrame table, .dataframe td, .dataframe th { font-size: 12px !important; }

  /* Card look for chart & dataframe containers */
  div[data-testid="stVegaLiteChart"]{
    background:#fff;border-radius:16px;padding:16px;
    box-shadow:0 8px 24px rgba(0,0,0,0.08),0 2px 6px rgba(0,0,0,0.06);
    margin-bottom:16px;
  }
  div[data-testid="stDataFrame"]{
    background:#fff;border-radius:16px;padding:8px 8px 2px 8px;
    box-shadow:0 8px 24px rgba(0,0,0,0.08),0 2px 6px rgba(0,0,0,0.06);
    margin-bottom:16px;
  }

  /* Pin brand logo bottom-left inside the sidebar (no background) */
  [data-testid="stSidebar"] { position: relative; }
  #sidebar-brand{
    position: fixed; left: 14px; bottom: 12px; z-index: 10;
    background: transparent; border: none; padding: 0; border-radius: 0;
  }
  #sidebar-brand img{ height: 70px; display:block; }

  /* Leaflet tooltips font */
  .leaflet-tooltip { font-size:11px; opacity:0.85; font-family:'Inter',sans-serif; }

  /* Custom legend (bottom-right, small font) */
  .custom-legend {
    position: fixed; right: 24px; bottom: 24px; z-index: 10050;
    background: rgba(255,255,255,0.98); border: 1px solid #ddd;
    padding: 8px 10px; border-radius: 8px; font-family: 'Inter', sans-serif; font-size: 10px;
    pointer-events: none;  /* don't block map controls */
  }
  .custom-legend .title{ font-weight:600; margin-bottom:4px; }
  .custom-legend .bar{ width: 240px; height: 10px; border-radius: 4px; }
  .custom-legend .ticks{ display:flex; justify-content:space-between; margin-top:4px; }
  .custom-legend .ticks span{ font-size:9px; color:#111827; }
</style>
""", unsafe_allow_html=True)

# -------- Assets/paths --------
LOGO_PATH = None
for candidate in [Path("logo.webp"), Path("logo.png")]:
    if candidate.exists():
        LOGO_PATH = candidate
        break
#currently it's ver4
DAILY_CSV = Path("indiana_county_daily_ver4.csv")
COUNTIES_GEOJSON = Path("indiana_counties_500k.geojson")
RAW_HOURLY_CSV = Path("in_parking_demand_data_ver4.xlsx")
SPOTS_GEOJSON = Path("IN_Truck_Spots_v2.geojson")
ROADWAYS_GEOJSON = Path("in_roadway_map_layer.geojson")

# Palettes
PALETTE_5 = ["#e8edb8", "#bbe2c4", "#9bd4d0", "#7cc0db", "#4e9dcf"]
PALETTE_4 = ["#e8edb8", "#bbe2c4", "#7cc0db", "#4e9dcf"]
DIAG_PALETTE = {
    "High Stress":   "#4e9dcf",
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
    df = pd.read_excel(RAW_HOURLY_CSV, sheet_name = 'parking_demand_data_2024_calibr')
    for size in ["small", "medium", "large"]:
        df[f"supply_{size}"] = df[f"truck_parking_spaces : private - {size}"] + df[f"truck_parking_spaces : public - {size}"]
    print(df.columns)
    df = df.drop(columns = {"county_name", "total_expanded_daily_parking_demand", 'truck_parking_lots: total',
       'truck_parking_lots : private - large',
       'truck_parking_lots : private - medium',
       'truck_parking_lots : private - small',
       'truck_parking_lots : public - large',
       'truck_parking_lots : public - medium',
       'truck_parking_lots : public - small', 
       'truck_parking_spaces : private - large',
       'truck_parking_spaces : private - medium',
       'truck_parking_spaces : private - small',
       'truck_parking_spaces : public - large',
       'truck_parking_spaces : public - medium',
       'truck_parking_spaces : public - small'
})
    
    df.columns = ["county","hour","des_demand", "undes_demand", "supply","supply_small","supply_medium","supply_large"]
    df.columns = [c.strip().lower() for c in df.columns]
    # normalize types
    df["county"] = df["county"].astype(str).str.zfill(5)
    df["hour"] = df["hour"].astype(int)
    for c in ["des_demand", "undes_demand", "supply","supply_small","supply_medium","supply_large"]:
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

# -------- Utils --------
def _quantile_edges(vals, q=(0, 0.25, 0.5, 0.75, 1.0)):
    """Robust quantile edges; fallback to equal intervals when degenerate."""
    vals = pd.Series(vals)
    try:
        edges = np.quantile(vals, q)
        edges = np.round(edges, 6)
        # Ensure monotonic non-decreasing
        edges = np.maximum.accumulate(edges)
        return edges
    except Exception:
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmin == vmax:
            return np.array([vmin, vmax])
        return np.linspace(vmin, vmax, len(q))

def _bin_and_color_series(vals, palette5, palette4):
    """
    Returns: (bin_idx:int series, colors:list, edges:ndarray)
    - Quantile binning (Q0,Q25,Q50,Q75,Q100)
    - Forces max->top bin
    - Colors matched to number of *intervals* (len(edges)-1)
    """
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    vmin, vmax = float(vals.min()), float(vals.max())
    edges = _quantile_edges(vals, q=(0, 0.25, 0.5, 0.75, 1.0))

    # If constant
    if float(edges[0]) == float(edges[-1]):
        bins = pd.Series(0, index=vals.index)
        return bins.astype(int), [palette4[0], palette4[0]], np.array([vmin, vmax])

    # Cut to quantile edges, allow duplicate edges but drop duplicate intervals
    raw = pd.cut(vals, bins=np.unique(edges), include_lowest=True, labels=False, duplicates="drop").astype("float")
    # Force the absolute max into the top interval
    top = np.nanmax(raw)
    raw[vals == vmax] = top
    bins = raw.fillna(0)

    uniq_bins = sorted(pd.Series(bins).unique())
    n_intervals = max(1, len(uniq_bins))
    colors = (palette5 if n_intervals >= 5 else palette4)[:n_intervals]

    # If after dropping duplicates we have <2 edges, synthesize min/max
    reduced_edges = np.unique(edges)
    if len(reduced_edges) < 2:
        reduced_edges = np.array([vmin, vmax])

    return bins.astype(int), colors, reduced_edges

def _fmt_compact(x: float) -> str:
    x = float(x)
    for unit in ["", "k", "M", "B", "T"]:
        if abs(x) < 1000.0:
            return f"{x:,.0f}{unit}"
        x /= 1000.0
    return f"{x:,.0f}P"

def _add_quartile_legend(m, colors, edges, title):
    """
    Inline-styled bottom-right legend with quartile labels.
    - edges can include duplicates (e.g., many zeros) — we dedupe adjacent labels.
    - colors should be len(edges)-1 after your binning.
    """
    def _fmt_compact(x: float) -> str:
        x = float(x)
        for unit in ["", "k", "M", "B", "T"]:
            if abs(x) < 1000.0:
                return f"{x:,.0f}{unit}"
            x /= 1000.0
        return f"{x:,.0f}P"

    # Build deduped tick labels from edges
    labels = []
    for i, v in enumerate(edges):
        lab = _fmt_compact(float(v))
        if i == 0 or lab != labels[-1]:
            labels.append(lab)

    gradient = ",".join(colors)
    ticks_html = "".join(f"<span style='font-size:9px;color:#111827'>{t}</span>" for t in labels)

    html = f"""
    <div style="
      position: fixed; right: 24px; bottom: 24px; z-index: 10050;
      background: rgba(255,255,255,0.98); border: 1px solid #ddd;
      padding: 8px 10px; border-radius: 8px; font-family: 'Inter', sans-serif; font-size: 10px;
      pointer-events: none;">
      <div style="font-weight:600; margin-bottom:4px;">{title}</div>
      <div style="width: 240px; height: 10px; border-radius: 4px; background: linear-gradient(90deg, {gradient});"></div>
      <div style="display:flex; justify-content:space-between; margin-top:4px;">{ticks_html}</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))

# -------- Map builders --------
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
        #default version
        folium.TileLayer("cartodbpositron", name="Basemap", control=False).add_to(m)
        # #Carto Voyager (more detailed, includes major roads and highways):
        # folium.TileLayer(
        #     tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        #     attr='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> '
        #          'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        #     name="Basemap", control=False, max_zoom=20
        # ).add_to(m)
        # #OpenStreetMap default (shows most roads, including numbered highways):
        # folium.TileLayer("openstreetmap", name="Basemap", control=False).add_to(m)
        
        # #ESRI World Street Map
        # folium.TileLayer(
        #     tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        #     attr='Tiles &copy; Esri — Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom',
        #     name="Basemap", control=False, max_zoom=19
        # ).add_to(m)




    # Inter inside the map iframe for tooltips
    m.get_root().header.add_child(folium.Element("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet" />
    <style>.leaflet-tooltip{font-size:11px;opacity:.85;font-family:'Inter',sans-serif;}</style>
    """))
    return m

def make_numeric_choropleth(gdf_joined, color_col, legend_label):
    gdf = gdf_joined.copy()
    vals = gdf[color_col].values

    bin_idx, colors, edges = _bin_and_color_series(vals, PALETTE_5, PALETTE_4)

    # map color per feature
    gdf["_bin"] = bin_idx
    color_map = {i: colors[i] for i in range(len(colors))}
    gdf["_color"] = gdf["_bin"].map(color_map).fillna(colors[-1] if len(colors) else "#cccccc")

    def style_fn(feat):
        return {"fillColor": feat["properties"].get("_color", "#cccccc"),
                "color": "#555", "weight": 0.8, "fillOpacity": 0.8}

    m = make_base_map()
    folium.GeoJson(gdf, style_function=style_fn, name=legend_label).add_to(m)

    # Quartile legend at bottom-right (small, deduped)
    _add_quartile_legend(m, colors, edges, legend_label)

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
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #ddd; border-radius:8px;">
      <b style="font-family:'Inter',sans-serif;font-size:12px;">Diagnosis</b><br>
    """
    for label, color in palette.items():
        legend_html += f'<span style="display:inline-block;width:12px;height:12px;background:{color};margin-right:6px;border:1px solid #666;"></span><span style="font-size:11px;font-family:\'Inter\',sans-serif;">{label}</span><br>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

def attach_tooltip_and_popup(m, gdf_joined):
    # Tooltip (unchanged fields)
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

    # Build a popup field that shows only County Name visually, but contains hidden FIPS for click parsing
    gdf = gdf_joined.copy()
    gdf["popup_html"] = gdf["county_name"].astype(str) + \
        " <span style='display:none;'>" + gdf["county_fips"].astype(str) + "</span>"

    gj = folium.GeoJson(
        gdf,
        name="Counties",
        style_function=lambda _: {"fillOpacity": 0, "color": "#555", "weight": 0.8},
        highlight_function=lambda x: {"weight": 2, "color": "black"},
        tooltip=tooltip,
    )
    # Show only the county name (FIPS is present but hidden)
    folium.GeoJsonPopup(
        fields=["popup_html"],
        aliases=["County"],
        parse_html=True,            # IMPORTANT: allow the hidden span to render/hide
        labels=True,
    ).add_to(gj)
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

    import re

    def _extract_raw(row):
        for k in ["Parking Type", "parking_type", "Parking_Type", "parkingType", "ParkingType"]:
            if k in row and pd.notna(row[k]):
                return row[k]
        return None

    def _canonical(pt_raw):
        if pt_raw is None or (isinstance(pt_raw, float) and pd.isna(pt_raw)):
            return "Unknown"
        s = str(pt_raw).strip().lower()
        m = re.search(r'(public|private).{0,10}?(small|medium|large)', s)
        if m:
            return f"{m.group(1).title()} - {m.group(2).title()}"
        m = re.search(r'(small|medium|large).{0,10}?(public|private)', s)
        if m:
            return f"{m.group(2).title()} - {m.group(1).title()}"
        return "Unknown"

    categories = [
        "Private - Small",
        "Private - Medium",
        "Private - Large",
        "Public - Small",
        "Public - Medium",
        "Public - Large",
    ]

    # keep pink family like you wanted
    color_map = {
        "Private - Small":  "#fde6f0",
        "Private - Medium": "#f9cfe0",
        "Private - Large":  "#f3a8c7",
        "Public - Small":   "#ef82ad",
        "Public - Medium":  "#e95b93",
        "Public - Large":   "#e13678",
        "Unknown":          "#8c8c8c",
    }

    gdf = spots_gdf.copy()
    gdf["__ptype"] = gdf.apply(lambda r: _canonical(_extract_raw(r)), axis=1)

    # One FeatureGroup per category (sub-layer)
    layer_by_cat = {}
    for cat in categories + ["Unknown"]:
        subset = gdf[gdf["__ptype"] == cat]
        if subset.empty:
            continue
        fg = folium.FeatureGroup(name=f"Spots — {cat}", show=(cat != "Unknown"))
        for _, r in subset.iterrows():
            geom = r.geometry
            if geom is None or geom.geom_type != "Point":
                continue
            folium.CircleMarker(
                location=[geom.y, geom.x],
                radius=2.5,
                weight=0,
                fill=True,
                fill_opacity=0.85,
                color=color_map.get(cat, "#8c8c8c")
            ).add_to(fg)
        fg.add_to(m)
        layer_by_cat[cat] = fg
    # (LayerControl already exists outside; no legend needed because layers are toggleable)


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

    # Logo pinned bottom-left of sidebar
    if LOGO_PATH:
        ext = LOGO_PATH.suffix[1:]
        b64 = base64.b64encode(open(LOGO_PATH, "rb").read()).decode("ascii")
        st.markdown(f"<div id='sidebar-brand'><img src='data:image/{ext};base64,{b64}'/></div>", unsafe_allow_html=True)

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
    # NEW:
    "supply_small", "supply_medium", "supply_large"
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
MAP_HEIGHT = 900  # raise/lower to align with right panel
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
    title = fips_to_name.get(st.session_state.selected_fips, f"County {st.session_state.selected_fips}")
    st.markdown(f"### Hourly demand distribution — **{title}**")

    def hourly_long(df_hourly, fips=None):
        if fips:
            sub = df_hourly[df_hourly["county"] == fips].copy()
        else:
            sub = df_hourly.copy()

        # Aggregate per hour
        agg = sub.groupby("hour", as_index=False)[
            ["des_demand", "undes_demand", "supply_small", "supply_medium", "supply_large"]
        ].sum()

        # Total supply = small + medium + large (per hour)
        agg["supply_total"] = agg["supply_small"] + agg["supply_medium"] + agg["supply_large"]

        long_df = agg.melt(
            id_vars="hour",
            value_vars=["des_demand", "undes_demand"],
            var_name="type", value_name="value"
        ).replace({"type": {"des_demand": "Designated", "undes_demand": "Undesignated"}})

        return long_df.sort_values("hour"), agg[["hour", "des_demand", "undes_demand",
                                                "supply_small", "supply_medium", "supply_large", "supply_total"]]

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

    # --- Stacked supply lines (cumulative y; component tooltips) ---
    # Labels (legend + tooltip Type)
    LABEL_SMALL = "Small lots"
    LABEL_MED   = "Medium lots"
    LABEL_TOTAL = "Total (all sizes)"  # cumulative all sizes
    
    if not hourly_table.empty:
        s_small  = hourly_table["supply_small"].reset_index(drop=True)
        s_medium = hourly_table["supply_medium"].reset_index(drop=True)
        s_large  = hourly_table["supply_large"].reset_index(drop=True)
    
        cum_small  = s_small
        cum_medium = s_small + s_medium
        cum_total  = s_small + s_medium + s_large
    
        supply_lines = pd.DataFrame({
            "hour": list(hourly_table["hour"]) * 3,
            "y":    pd.concat([cum_small, cum_medium, cum_total], ignore_index=True),  # plotted cumulative
            "component": pd.concat([s_small, s_medium, s_large], ignore_index=True),   # tooltip component
            "type": ([LABEL_SMALL] * len(hourly_table)
                     + [LABEL_MED] * len(hourly_table)
                     + [LABEL_TOTAL] * len(hourly_table))
        })
    else:
        supply_lines = pd.DataFrame(columns=["hour","y","component","type"])
    
    # Demand bars (unchanged)
    stacked = (
        alt.Chart(bars_long)
          .mark_bar()
          .encode(
              x=alt.X("hour:O", title="Hour of day",
                      axis=alt.Axis(labelAngle=0, labelOverlap=True, titlePadding=12)),
              y=alt.Y("sum(value):Q", title="Demand (truck-hours)", axis=alt.Axis(format=",.0f")),
              color=alt.Color("type:N", title="",
                              scale=alt.Scale(domain=["Designated","Undesignated"]),
                              sort=["Designated","Undesignated"]),
              order=alt.Order("type_order:Q"),
              tooltip=[
                  alt.Tooltip("hour:O", title="Hour"),
                  alt.Tooltip("type:N", title="Type"),
                  alt.Tooltip("sum(value):Q", title="Demand", format=",.0f")
              ]
          )
          .properties(height=320)
    )
    
    # Supply lines (greens, independent color scale)
    greens = ["#bfe7c2", "#59c27f", "#0f7a3a"]  # light → mid → dark
    supply_chart = (
        alt.Chart(supply_lines)
          .mark_line(size=2)
          .encode(
              x=alt.X("hour:O"),
              y=alt.Y("y:Q"),
              color=alt.Color(
                  "type:N",
                  title="",
                  sort=[LABEL_SMALL, LABEL_MED, LABEL_TOTAL],
                  scale=alt.Scale(
                      domain=[LABEL_SMALL, LABEL_MED, LABEL_TOTAL],
                      range=greens
                  )
              ),
              tooltip=[
                  alt.Tooltip("hour:O", title="Hour"),
                  alt.Tooltip("type:N", title="Type"),
                  alt.Tooltip("component:Q", title="Capacity", format=",.0f")
              ]
          )
    )
    
    chart = (stacked + supply_chart).resolve_scale(
        color='independent'   # keep bar palette separate from supply lines
    ).properties(
        padding={"left": 4, "right": 4, "top": 4, "bottom": 36}
    ).configure_axis(
        labelFont="Inter", titleFont="Inter", titleFontSize=11
    ).configure_legend(
        labelFont="Inter", titleFont="Inter"
    )
    
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
    ("Total Supply", "supply_fmt"),                      # renamed label
    ("Supply (small size)", "supply_small_fmt"),         # new
    ("Supply (medium size)", "supply_medium_fmt"),       # new
    ("Supply (large size)", "supply_large_fmt"),         # new
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












