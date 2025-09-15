# app.py — Indiana Truck Parking (backend truck-spots overlay ON by default)
import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from streamlit_folium import st_folium
import folium
import altair as alt
import re
import branca.colormap as cm

# --- password gate (keep at the very top) ---
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

# ---- fonts (Streamlit-side; load Inter + apply broadly) ----
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root, html, body, .stApp, .stMarkdown, .stTextInput, .stSelectbox, .stDataFrame, .stButton, .stCaption, .stDownloadButton, .stMetric {
    font-family: Inter, Arial, sans-serif !important;
  }
  /* Tighter table text */
  .stDataFrame table, .dataframe td, .dataframe th {
    font-size: 12px !important;
  }
</style>
""", unsafe_allow_html=True)

# ---- logo path (try webp then png) ----
LOGO_PATH = None
for candidate in [Path("logo.webp"), Path("logo.png")]:
    if candidate.exists():
        LOGO_PATH = candidate
        break

# ---- data paths ----
DAILY_CSV = Path("indiana_county_daily_ver2.csv")
COUNTIES_GEOJSON = Path("indiana_counties_500k.geojson")
# V2 hourly input
RAW_HOURLY_CSV = Path("in_parking_demand_data_ver2.xlsx")  # used for stacked bars & hourly download
SPOTS_GEOJSON = Path("IN_Truck_Spots.geojson")              # backend truck parking spots
ROADWAYS_GEOJSON = Path("in_roadway_map_layer.geojson")     # roadway lines (no tooltip)

# ---- choropleth palettes (5 and 4 stops) ----
PALETTE_5 = ["#e8edb8", "#bbe2c4", "#9bd4d0", "#7cc0db", "#61a1ca"]
PALETTE_4 = ["#e8edb8", "#bbe2c4", "#7cc0db", "#61a1ca"]

# ---- diagnosis palette (yellow→blue ramp per your request) ----
DIAG_PALETTE = {
    "High Stress":   "#61a1ca",  # blue
    "Elevated":      "#7cc0db",  # light blue
    "Typical/Other": "#bbe2c4",  # light green
    "No Supply":     "#e8edb8",  # yellow
}

# ---------- cached loaders ----------
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
    # V2 Excel input
    df = pd.read_excel(RAW_HOURLY_CSV, sheet_name='park_dem_calibrtd_by_hour')
    # quick processing for the new data format
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

# ---------- map builders ----------
def make_base_map():
    """Create a Leaflet map using Mapbox if token present, else cartodbpositron.
       Hide basemap from LayerControl."""
    m = folium.Map(location=[39.9, -86.3], zoom_start=7, tiles=None)

    token = st.secrets.get("MAPBOX_TOKEN")
    style = st.secrets.get("MAPBOX_STYLE", "mapbox/streets-v11")  # e.g., "mapbox/streets-v11" or "user/style-id"

    if token:
        # 512px tiles + zoomOffset for crisp rendering
        folium.TileLayer(
            tiles=f"https://api.mapbox.com/styles/v1/{style}/tiles/512/{{z}}/{{x}}/{{y}}?access_token={token}",
            attr="Mapbox",
            name="Basemap",
            control=False,
            max_zoom=20,
            tileSize=512,      # Leaflet option via **kwargs (note camelCase)
            zoomOffset=-1      # Leaflet option via **kwargs
        ).add_to(m)
    else:
        folium.TileLayer("cartodbpositron", name="Basemap", control=False).add_to(m)

    # Smaller & slightly transparent tooltips inside the map iframe + Inter font
    m.get_root().header.add_child(folium.Element("""
    <style>
      .leaflet-tooltip {
        font-size: 11px;
        opacity: 0.85;
        font-family: Inter, Arial, sans-serif;
      }
    </style>
    """))
    return m

def make_numeric_choropleth(gdf_joined, color_col, legend_label):
    """Quantile-bucketed choropleth with top-right colorbar ('ruler')."""
    gdf = gdf_joined.copy()
    vals = pd.to_numeric(gdf[color_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    # Quantile edges (0, 25, 50, 75, 100). Dedup to handle flat distributions.
    raw_edges = np.quantile(vals, [0, 0.25, 0.5, 0.75, 1.0])
    edges = np.unique(np.round(raw_edges, 6))

    # Choose palette size to match bins actually used
    colors = PALETTE_5 if len(edges) >= 5 else PALETTE_4

    # If too few unique values, fallback to min/max
    if len(edges) < 2:
        edges = np.array([vals.min(), vals.max()])

    # Bin by quantile edges; labels=False -> integer bin index
    gdf["_bin"] = pd.cut(vals, bins=edges, include_lowest=True, labels=False, duplicates="drop")
    unique_bins = sorted(gdf["_bin"].dropna().unique())
    n_bins = len(unique_bins)

    # If duplicates collapsed, rebuild uniform edges across value range for a clean bar,
    # then re-bin so map and colorbar always align.
    if n_bins != len(colors):
        edges = np.linspace(vals.min(), vals.max(), n_bins + 1) if n_bins > 0 else np.array([0, 1])
        colors = colors[:n_bins]
        gdf["_bin"] = pd.cut(vals, bins=edges, include_lowest=True, labels=False, duplicates="drop")
        unique_bins = sorted(gdf["_bin"].dropna().unique())
        n_bins = len(unique_bins)

    color_map = {i: colors[i] for i in range(n_bins)}

    def style_fn(feat):
        b = feat["properties"].get("_bin")
        color = color_map.get(b, "#cccccc")
        return {"fillColor": color, "color": "#555", "weight": 0.8, "fillOpacity": 0.8}

    m = make_base_map()
    folium.GeoJson(gdf, style_function=style_fn, name=legend_label).add_to(m)

    # Top-right colorbar (continuous ramp using chosen colors)
    if n_bins > 0:
        colormap = cm.LinearColormap(colors=colors, vmin=float(edges[0]), vmax=float(edges[-1]))
        colormap.caption = legend_label
        colormap.add_to(m)
        # Reposition to top-right and apply Inter
        m.get_root().header.add_child(folium.Element("""
        <style>
          .branca-colormap {
            z-index: 9999;
            position: fixed !important;
            top: 30px;
            right: 30px;
            bottom: auto !important;
            left: auto !important;
            font-family: Inter, Arial, sans-serif;
          }
          .branca-colormap .caption {
            font-family: Inter, Arial, sans-serif;
            font-size: 12px;
          }
        </style>
        """))

    return m

def make_categorical_map(gdf_joined, category_col, palette=None):
    """Diagnosis categorical map with palette + legend bottom-left."""
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
    """Small, semi-transparent tooltip with 4 fields + popup FIPS to capture clicks."""
    fields = [
        ("County", "county_name"),
        ("FIPS", "county_fips"),
        ("Supply (hourly fixed)", "supply_fmt"),
        ("Max hourly total demand", "max_hourly_total_demand_fmt"),
    ]
    tooltip = folium.features.GeoJsonTooltip(
        fields=[f for _, f in fields],
        aliases=[a for a, _ in fields],
        sticky=True,
        localize=True,
        labels=True,
        style=("background-color: rgba(255,255,255,0.9);"
               "border: 1px solid #ccc;"
               "border-radius: 4px;"
               "padding: 6px;"
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
    """Add roadways (lines) as a toggleable layer (ON by default), no tooltip."""
    if road_gdf is None or road_gdf.empty:
        return
    fg = folium.FeatureGroup(name="Roadways", show=True)
    folium.GeoJson(
        road_gdf,
        name="Roadways",
        style_function=lambda _: {"color": "#4d4d4d", "weight": 1.0, "opacity": 0.8},
    ).add_to(fg)
    fg.add_to(m)

def add_truck_spots_layer(m, spots_gdf):
    """Add truck spots as a toggleable layer (ON by default), no tooltip/popup."""
    if spots_gdf is None or spots_gdf.empty:
        return
    fg = folium.FeatureGroup(name="Truck parking spots", show=True)
    for _, r in spots_gdf.iterrows():
        geom = r.geometry
        if geom and geom.geom_type == "Point":
            folium.CircleMarker(
                location=[geom.y, geom.x],
                radius=2.5,
                weight=0,
                fill=True,
                fill_opacity=0.8,
            ).add_to(fg)
    fg.add_to(m)

# ---------- UI ----------
st.title("Indiana Truck Parking — County Dashboard")

# Friendly names for sidebar & legend
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
        "Map: choose metric (or diagnosis)",
        options=["Diagnosis"] + labels_numeric,
        index=0
    )
    st.caption("Tip: Click a county to update the stacked hourly chart and the download on the right.")
    if LOGO_PATH:
        st.image(str(LOGO_PATH), use_container_width=True)

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

# --- Create *_fmt (integer) columns for tooltip display only ---
fmt_targets = [
    "max_hourly_des_demand",
    "max_hourly_undes_demand",
    "max_hourly_total_demand",
    "acc_des_demand",
    "acc_undes_demand",
    "acc_total_demand",
    "supply",
    "max_hourly_des_deficit",
    "max_hourly_total_deficit",
    "acc_des_deficit",
    "acc_total_deficit",
]
for col in fmt_targets:
    fmt_col = f"{col}_fmt"
    if col in gdf_joined.columns:
        gdf_joined[fmt_col] = gdf_joined[col].round(0).astype(int)
    else:
        gdf_joined[fmt_col] = 0

# optional notices if overlays missing
if spots_err:
    st.info(spots_err)
if road_err:
    st.info(road_err)

# Default county (Marion)
if "selected_fips" not in st.session_state or not st.session_state.selected_fips:
    st.session_state.selected_fips = "18097"  # Marion
if "ignore_next_click" not in st.session_state:
    st.session_state.ignore_next_click = False

# layout
col_map, col_right = st.columns([3, 2], gap="large")

with col_map:
    if map_metric_label == "Diagnosis":
        m = make_categorical_map(gdf_joined, "diagnosis")
    else:
        m = make_numeric_choropleth(
            gdf_joined,
            color_col=metric_label_to_key[map_metric_label],
            legend_label=map_metric_label
        )

    # tooltip + popup on top of counties
    attach_tooltip_and_popup(m, gdf_joined)

    # --- Layer order: heatmap/categorical -> Roadways -> Spots ---
    add_roadways_layer(m, road_gdf)     # middle
    add_truck_spots_layer(m, spots_gdf) # top

    folium.LayerControl(collapsed=False).add_to(m)
    map_state = st_folium(
        m, height=650, use_container_width=True,
        returned_objects=["last_object_clicked_popup"]
    )

# sanitize popup → fips, unless we're ignoring the next click (after a clear)
if map_state and map_state.get("last_object_clicked_popup") and not st.session_state.ignore_next_click:
    raw = str(map_state["last_object_clicked_popup"])
    cleaned = re.sub(r"\D", "", raw).zfill(5)
    st.session_state.selected_fips = cleaned

# clear the guard once we've passed the read phase
if st.session_state.ignore_next_click:
    st.session_state.ignore_next_click = False

# helper: fips → county name
fips_to_name = dict(zip(gdf_joined["county_fips"], gdf_joined["county_name"]))

with col_right:
    st.markdown("### Hourly demand distribution")

    def hourly_long(df_hourly, fips=None):
        if fips:
            sub = df_hourly[df_hourly["county"] == fips].copy()
            title = fips_to_name.get(fips, f"County {fips}")
            # supply constant for this county from daily metrics
            supply_const = float(daily.loc[daily["county_fips"] == fips, "supply"].fillna(0).max())
        else:
            sub = df_hourly.copy()
            title = "Indiana (statewide)"
            # statewide supply = sum of county supplies (constant across hours)
            supply_const = float(daily["supply"].fillna(0).sum())

        # aggregate demand by hour; set constant supply per hour
        agg = sub.groupby("hour", as_index=False)[["des_demand", "undes_demand"]].sum()
        agg["supply"] = supply_const

        # long form for stacked bars (Designated bottom, Undesignated top)
        long_df = agg.melt(
            id_vars="hour",
            value_vars=["des_demand", "undes_demand"],
            var_name="type",
            value_name="value"
        ).replace({"type": {"des_demand": "Designated", "undes_demand": "Undesignated"}})

        return title, long_df.sort_values("hour"), agg[["hour", "des_demand", "undes_demand", "supply"]]

    # Add horizontal supply line
    title, bars_long, hourly_table = hourly_long(hourly, st.session_state.selected_fips)
    st.write(f"**{title}**")

    bars_long["type_order"] = bars_long["type"].map({"Designated": 0, "Undesignated": 1})

    stacked = (
        alt.Chart(bars_long)
          .mark_bar()
          .encode(
              x=alt.X("hour:O", title="Hour of day"),
              y=alt.Y("sum(value):Q", title="Demand (truck-hours)", axis=alt.Axis(format=",.0f")),
              color=alt.Color(
                  "type:N",
                  title="",
                  scale=alt.Scale(domain=["Designated", "Undesignated"]),
                  sort=["Designated", "Undesignated"]
              ),
              order=alt.Order("type_order:Q"),
              tooltip=[
                  alt.Tooltip("hour:O", title="Hour"),
                  alt.Tooltip("type:N", title="Type"),
                  alt.Tooltip("sum(value):Q", title="Demand", format=",.0f")
              ]
          )
          .properties(height=400)
    )

    # horizontal supply line (constant across hours)
    supply_const = float(hourly_table["supply"].iloc[0]) if not hourly_table.empty else 0.0
    rule = alt.Chart(pd.DataFrame({"y": [supply_const]})).mark_rule().encode(y="y:Q")

    chart = (stacked + rule).configure_axis(
        labelFont="Inter", titleFont="Inter"
    ).configure_legend(
        labelFont="Inter", titleFont="Inter"
    )

    st.altair_chart(chart, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear selection"):
            st.session_state.selected_fips = None
            st.session_state.ignore_next_click = True  # ignore the next map click event
            st.rerun()
    with c2:
        # Download HOURLY (scoped to selection; default statewide)
        csv_bytes = hourly_table.to_csv(index=False).encode("utf-8")
        label = "Download hourly demand (statewide)" if st.session_state.selected_fips is None \
                else f"Download hourly demand ({title})"
        st.download_button(
            label=label,
            data=csv_bytes,
            file_name="hourly_demand.csv",
            mime="text/csv",
        )

    # --- County profile under the chart (right column) ---
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
        items = []
        for label, col in profile_fields:
            val = row.iloc[0].get(col, "")
            items.append((label, val))
        dfp = pd.DataFrame(items, columns=["Metric", "Value"])
        return dfp

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
