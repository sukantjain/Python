#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_catchment_map.py
==============================================================================
Publication-ready hydrological / catchment cartography script.

Produces a scientific-quality map (PNG, PDF, SVG) showing:
    - DEM with a hypsometric (elevation) colour ramp
    - Hillshade-blended relief
    - Catchment boundary
    - Drainage network (stream-order-weighted line widths where available)
    - Latitude/longitude graticule (decimal degrees)
    - North arrow
    - Dynamically-computed scale bar
    - Legend
    - Overview/location inset map (with graceful offline fallback)
    - Metadata / source information box

REQUIRED PACKAGES
------------------------------------------------------------------------------
    pip install geopandas rasterio numpy matplotlib shapely pyproj
Optional (used for the online overview basemap, will gracefully degrade
if unavailable):
    pip install contextily xyzservices

Tested conceptually against geopandas>=0.13 / rasterio>=1.3 / matplotlib>=3.7.
Works on Windows, macOS and Linux. All paths use pathlib.

Author: generated for the user's hydrological mapping workflow.
==============================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------------
# Third-party geospatial / plotting imports (hard requirements)
# ------------------------------------------------------------------------
try:
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.io import MemoryFile
    from shapely.geometry import mapping, box
    import matplotlib
    matplotlib.use("Agg")  # safe non-interactive backend; still saves crisp vector/raster output
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LightSource
    from matplotlib.patches import Rectangle, FancyArrow
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredText
    from matplotlib import font_manager
    from pyproj import Transformer, CRS
except ImportError as exc:  # pragma: no cover - environment guidance only
    sys.exit(
        "ERROR: A required package is missing ({}).\n"
        "Install the required packages with:\n"
        "    pip install geopandas rasterio numpy matplotlib shapely pyproj\n"
        "Optional (for the online overview basemap):\n"
        "    pip install contextily xyzservices".format(exc)
    )

# Optional dependency: contextily (used only for the overview/inset basemap).
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False


# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
# --- Input datasets ----------------------------------------------------------
CATCHMENT_FILE = r"data/catchment_boundary.shp"
DRAINAGE_FILE = r"data/drainage_network.shp"
DEM_FILE = r"data/dem.tif"

# --- Output -------------------------------------------------------------------
OUTPUT_DIR = r"output"
OUTPUT_BASENAME = "catchment_map"  # files will be OUTPUT_BASENAME.png / .pdf / .svg

# --- Map identity ---------------------------------------------------------
CATCHMENT_NAME = "Upper Narmada"          # used to build the dynamic title
MAP_TITLE_TEMPLATE = "Topography and Drainage Network of {name}"

# --- Figure geometry --------------------------------------------------------
FIG_WIDTH = 10          # inches
FIG_HEIGHT = 8          # inches
DPI = 600               # raster export resolution (300-600 recommended)

# --- Map extent ---------------------------------------------------------------
MAP_MARGIN = 0.05        # fractional margin added around the catchment bounds

# --- CRS handling ---------------------------------------------------------
AUTO_DETECT_CRS = True        # if True, best UTM zone is derived from the catchment centroid
MANUAL_CRS = None             # e.g. "EPSG:32644" -- used only if AUTO_DETECT_CRS is False
ASSUME_CRS_IF_MISSING = "EPSG:4326"  # fallback if an input file has no CRS defined

# --- DEM processing ---------------------------------------------------------
CLIP_BUFFER_KM = 1.0                 # buffer distance (km) applied around the catchment before clipping the DEM
DEM_PERCENTILE_STRETCH = True        # robust colour stretch to avoid single-pixel outliers dominating the ramp
DEM_STRETCH_PERCENTILES = (1, 99)    # used only if DEM_PERCENTILE_STRETCH is True

# --- Hillshade ---------------------------------------------------------------
HILLSHADE_AZIMUTH = 315
HILLSHADE_ALTITUDE = 45
HILLSHADE_ALPHA = 0.35        # blend strength (used as vert_exag proxy / soft blend weighting)
HILLSHADE_VERT_EXAG = 1.0     # vertical exaggeration factor

# --- Symbology ---------------------------------------------------------------
BOUNDARY_COLOR = "#7a1f1f"     # dark maroon - reads clearly in colour and grayscale
BOUNDARY_LINEWIDTH = 1.8
BOUNDARY_FILL_ALPHA = 0.04     # very subtle catchment fill

DRAINAGE_COLOR = "#1c4f8c"     # strong blue, contrasts with hypsometric ramp
DRAINAGE_BASE_LINEWIDTH = 0.6
DRAINAGE_MAX_LINEWIDTH = 2.4

# --- Map elements toggles ---------------------------------------------------
SHOW_OVERVIEW_MAP = True
SHOW_GRID = True
SHOW_SCALE_BAR = True
SHOW_NORTH_ARROW = True
SHOW_INFO_BOX = True

# --- Metadata (shown in the info box) ---------------------------------------
DEM_SOURCE = "SRTM 30m DEM (USGS EarthExplorer)"
CATCHMENT_SOURCE = "Delineated from DEM (user-supplied)"
DRAINAGE_SOURCE = "Derived drainage network (user-supplied)"
DATA_DATE = "2024"

# --- Fonts --------------------------------------------------------------------
FONT_FAMILY = "DejaVu Sans"   # widely-available sans-serif; falls back automatically if unavailable


# ==============================================================================
# END OF USER CONFIGURATION -- no changes required below this line
# ==============================================================================


@dataclass
class MapExtent:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def as_tuple(self):
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin


# ------------------------------------------------------------------------
# Small compatibility helper (geopandas has renamed unary_union -> union_all)
# ------------------------------------------------------------------------
def _dissolve_geometry(gdf: "gpd.GeoDataFrame"):
    if hasattr(gdf, "union_all"):
        return gdf.union_all()
    return gdf.unary_union


# ------------------------------------------------------------------------
# 1. VALIDATION
# ------------------------------------------------------------------------
def validate_inputs(catchment_path: Path, drainage_path: Path, dem_path: Path) -> None:
    """Check that all required input files exist before any processing begins."""
    missing = [p for p in (catchment_path, drainage_path, dem_path) if not p.exists()]
    if missing:
        names = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "The following required input file(s) could not be found:\n"
            f"{names}\n"
            "Please check CATCHMENT_FILE, DRAINAGE_FILE and DEM_FILE in the "
            "USER CONFIGURATION section."
        )

    # Quick sanity check that the DEM is actually readable as a raster.
    try:
        with rasterio.open(dem_path):
            pass
    except Exception as exc:
        raise ValueError(f"DEM file '{dem_path}' could not be opened as a raster: {exc}")

    # Quick sanity check that the vector files are readable.
    for label, p in (("catchment", catchment_path), ("drainage", drainage_path)):
        try:
            test_gdf = gpd.read_file(p, rows=1)
            if test_gdf.empty and len(gpd.read_file(p)) == 0:
                raise ValueError(f"The {label} file '{p}' contains no features.")
        except Exception as exc:
            raise ValueError(f"Could not read the {label} file '{p}': {exc}")


def ensure_crs(gdf: "gpd.GeoDataFrame", label: str) -> "gpd.GeoDataFrame":
    """Ensure a GeoDataFrame has a CRS, falling back to a configured default with a warning."""
    if gdf.crs is None:
        warnings.warn(
            f"The {label} layer has no CRS defined. Assuming {ASSUME_CRS_IF_MISSING} "
            "as configured in ASSUME_CRS_IF_MISSING. Verify this is correct."
        )
        gdf = gdf.set_crs(ASSUME_CRS_IF_MISSING)
    return gdf


# ------------------------------------------------------------------------
# 2. CRS SELECTION (automatic UTM zone detection)
# ------------------------------------------------------------------------
def detect_best_utm_crs(catchment_gdf: "gpd.GeoDataFrame") -> str:
    """
    Automatically determine an appropriate projected CRS (UTM) for the catchment.

    Logic:
      1. Reproject the catchment to geographic coordinates (EPSG:4326) to get a
         reliable longitude/latitude centroid regardless of the source CRS.
      2. The UTM zone number is derived from longitude: zone = floor((lon+180)/6) + 1.
      3. The hemisphere (and therefore whether to use the 326xx or 327xx EPSG
         family) is derived from the sign of the centroid latitude.
    This gives a locally accurate, equal-ish-area, low-distortion CRS suitable
    for area/distance calculations and scale-bar generation for a single
    catchment (which is always small relative to a UTM zone).
    """
    geom_wgs84 = gpd.GeoSeries(_dissolve_geometry(catchment_gdf), crs=catchment_gdf.crs).to_crs(4326)
    centroid = geom_wgs84.iloc[0].centroid
    lon, lat = centroid.x, centroid.y

    utm_zone = int((lon + 180) / 6) + 1
    utm_zone = min(max(utm_zone, 1), 60)  # clamp to valid UTM zone range
    epsg_code = (32600 if lat >= 0 else 32700) + utm_zone
    crs_str = f"EPSG:{epsg_code}"
    print(f"[CRS] Auto-detected projected CRS: {crs_str} "
          f"(UTM zone {utm_zone}{'N' if lat >= 0 else 'S'}, from centroid {lon:.4f}, {lat:.4f})")
    return crs_str


def resolve_target_crs(catchment_gdf: "gpd.GeoDataFrame") -> str:
    if not AUTO_DETECT_CRS:
        if not MANUAL_CRS:
            raise ValueError(
                "AUTO_DETECT_CRS is False but MANUAL_CRS is not set. "
                "Please provide a CRS such as 'EPSG:32644' in MANUAL_CRS."
            )
        print(f"[CRS] Using manually specified CRS: {MANUAL_CRS}")
        return MANUAL_CRS
    return detect_best_utm_crs(catchment_gdf)


# ------------------------------------------------------------------------
# 3. VECTOR LOADING
# ------------------------------------------------------------------------
def load_vector(path: Path, label: str, target_crs: str) -> "gpd.GeoDataFrame":
    gdf = gpd.read_file(path)
    gdf = ensure_crs(gdf, label)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    if gdf.empty:
        raise ValueError(f"The {label} layer has no valid geometries after cleaning.")
    gdf = gdf.to_crs(target_crs)
    return gdf


STREAM_ORDER_FIELD_CANDIDATES = [
    "STRAHLER", "strahler", "STRAHLER_O", "ORDER", "order", "Order",
    "STRM_ORDER", "STREAM_ORD", "grid_code", "GRID_CODE", "strmOrder",
    "ORDER_", "HORTON", "SHREVE",
]


def find_stream_order_field(drainage_gdf: "gpd.GeoDataFrame"):
    for field in STREAM_ORDER_FIELD_CANDIDATES:
        if field in drainage_gdf.columns:
            try:
                values = drainage_gdf[field].astype(float)
                if values.notna().sum() > 0 and values.nunique() > 1:
                    print(f"[Drainage] Stream order field detected: '{field}'")
                    return field
            except (ValueError, TypeError):
                continue
    print("[Drainage] No usable stream-order field found; using a uniform line width.")
    return None


# ------------------------------------------------------------------------
# 4. DEM LOADING, REPROJECTION & CLIPPING
# ------------------------------------------------------------------------
def reproject_dem_if_needed(dem_path: Path, target_crs: str):
    """
    Reproject the DEM to the target CRS if necessary. Returns an open
    in-memory rasterio dataset (plus the MemoryFile, kept alive by the caller)
    so downstream masking can be performed directly.
    """
    src = rasterio.open(dem_path)

    if src.crs is None:
        raise ValueError(
            f"The DEM file '{dem_path}' has no CRS defined. Please assign a CRS "
            "to the DEM (e.g. using gdal_translate/gdal_edit) before running this script."
        )

    if CRS(src.crs).to_string() == CRS(target_crs).to_string():
        return None, src  # no reprojection needed; caller owns 'src'

    transform, width, height = calculate_default_transform(
        src.crs, target_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update({
        "crs": target_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "dtype": "float32",
    })

    memfile = MemoryFile()
    dst = memfile.open(**kwargs)
    for band_index in range(1, src.count + 1):
        reproject(
            source=rasterio.band(src, band_index),
            destination=rasterio.band(dst, band_index),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
    src.close()
    print(f"[DEM] Reprojected DEM to {target_crs}.")
    return memfile, dst


def clip_dem_to_catchment(dem_dataset, catchment_gdf: "gpd.GeoDataFrame", buffer_km: float):
    """Clip (mask) the DEM to the catchment boundary plus a buffer, handling NoData robustly."""
    buffered_geom = _dissolve_geometry(catchment_gdf).buffer(buffer_km * 1000.0)
    geoms = [mapping(buffered_geom)]

    nodata_val = dem_dataset.nodata
    out_image, out_transform = rio_mask(
        dem_dataset, geoms, crop=True, filled=True,
        nodata=nodata_val if nodata_val is not None else np.nan,
    )
    dem_array = out_image[0].astype("float64")

    if nodata_val is not None:
        dem_array[dem_array == nodata_val] = np.nan
    # Guard against implausible sentinel values sometimes used for NoData (-9999, etc.)
    dem_array[dem_array < -1000] = np.nan

    if np.all(np.isnan(dem_array)):
        raise ValueError(
            "The DEM contains no valid data within the catchment boundary + buffer. "
            "Check that the DEM actually covers the catchment and that the CRS/extent align."
        )

    return dem_array, out_transform


def dem_extent_from_transform(transform, width, height):
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + transform.a * width
    ymin = ymax + transform.e * height
    return (xmin, xmax, ymin, ymax)


# ------------------------------------------------------------------------
# 5. HILLSHADE + COLOUR-MAPPED RELIEF
# ------------------------------------------------------------------------
def build_hypsometric_cmap() -> LinearSegmentedColormap:
    """A muted, desaturated hypsometric-tint colour ramp suitable for scientific print."""
    stops = [
        (0.00, "#2f6d3c"),  # valley floors - muted green
        (0.20, "#6a9a4b"),
        (0.40, "#a9b96a"),
        (0.60, "#d9c98a"),
        (0.75, "#c39a6b"),
        (0.90, "#a9795a"),
        (1.00, "#f2ede4"),  # highest peaks - pale, near-white
    ]
    return LinearSegmentedColormap.from_list("hypsometric", stops, N=256)


def compute_shaded_relief(dem_array: np.ndarray, transform):
    """
    Blend a hypsometric elevation colour ramp with a hillshade using
    matplotlib's LightSource, which implements a correct, standard
    hillshading algorithm (avoids re-deriving gradients by hand).
    """
    dx = abs(transform.a)
    dy = abs(transform.e)

    if DEM_PERCENTILE_STRETCH:
        lo, hi = np.nanpercentile(dem_array, DEM_STRETCH_PERCENTILES)
    else:
        lo, hi = np.nanmin(dem_array), np.nanmax(dem_array)

    cmap = build_hypsometric_cmap()
    ls = LightSource(azdeg=HILLSHADE_AZIMUTH, altdeg=HILLSHADE_ALTITUDE)

    dem_filled = np.where(np.isnan(dem_array), np.nanmin(dem_array), dem_array)

    rgb = ls.shade(
        dem_filled,
        cmap=cmap,
        vmin=lo,
        vmax=hi,
        blend_mode="soft",
        vert_exag=HILLSHADE_VERT_EXAG,
        dx=dx,
        dy=dy,
        fraction=1.0 + HILLSHADE_ALPHA,  # nudges relief contrast without overpowering colour
    )

    # Restore transparency for NoData pixels (outside the catchment/buffer).
    alpha_mask = (~np.isnan(dem_array)).astype("float64")
    rgba = np.dstack([rgb[..., :3], alpha_mask])
    return rgba, lo, hi, cmap


# ------------------------------------------------------------------------
# 6. MAP EXTENT
# ------------------------------------------------------------------------
def compute_map_extent(catchment_gdf: "gpd.GeoDataFrame", margin_fraction: float) -> MapExtent:
    xmin, ymin, xmax, ymax = catchment_gdf.total_bounds
    width, height = xmax - xmin, ymax - ymin
    mx, my = width * margin_fraction, height * margin_fraction
    return MapExtent(xmin - mx, xmax + mx, ymin - my, ymax + my)


# ------------------------------------------------------------------------
# 7. GRATICULE (lat/lon grid in decimal degrees)
# ------------------------------------------------------------------------
def choose_nice_degree_step(span_degrees: float) -> float:
    candidates = [0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10]
    target_lines = 5
    for c in candidates:
        if span_degrees / c <= target_lines * 1.6:
            return c
    return candidates[-1]


def format_lat(value: float) -> str:
    return f"{abs(value):.3f}°{'N' if value >= 0 else 'S'}"


def format_lon(value: float) -> str:
    return f"{abs(value):.3f}°{'E' if value >= 0 else 'W'}"


def add_graticule(ax, target_crs: str, extent: MapExtent):
    """
    Draw a light lat/lon graticule and label the map frame in decimal degrees.
    For a catchment-scale extent, meridians/parallels are treated as
    approximately straight lines within the plotted CRS, which is an
    accurate approximation at this scale and keeps the implementation
    robust without requiring a full cartographic-projection library.
    """
    to_geo = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    to_proj = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    xmin, xmax, ymin, ymax = extent.as_tuple()

    edge_x = np.linspace(xmin, xmax, 60)
    edge_y = np.linspace(ymin, ymax, 60)

    lon_b, lat_b = to_geo.transform(edge_x, np.full_like(edge_x, ymin))
    lon_t, lat_t = to_geo.transform(edge_x, np.full_like(edge_x, ymax))
    lon_l, lat_l = to_geo.transform(np.full_like(edge_y, xmin), edge_y)
    lon_r, lat_r = to_geo.transform(np.full_like(edge_y, xmax), edge_y)

    lon_min = float(np.min(np.concatenate([lon_b, lon_t, lon_l, lon_r])))
    lon_max = float(np.max(np.concatenate([lon_b, lon_t, lon_l, lon_r])))
    lat_min = float(np.min(np.concatenate([lat_b, lat_t, lat_l, lat_r])))
    lat_max = float(np.max(np.concatenate([lat_b, lat_t, lat_l, lat_r])))

    step = choose_nice_degree_step(max(lon_max - lon_min, lat_max - lat_min))

    lon_ticks = np.arange(np.ceil(lon_min / step) * step, lon_max, step)
    lat_ticks = np.arange(np.ceil(lat_min / step) * step, lat_max, step)

    lat_samples = np.linspace(lat_min - step, lat_max + step, 80)
    lon_samples = np.linspace(lon_min - step, lon_max + step, 80)

    for lon in lon_ticks:
        x, y = to_proj.transform(np.full_like(lat_samples, lon), lat_samples)
        ax.plot(x, y, color="#888888", linewidth=0.4, linestyle=(0, (1, 3)),
                 alpha=0.6, zorder=2, clip_on=True)

    for lat in lat_ticks:
        x, y = to_proj.transform(lon_samples, np.full_like(lon_samples, lat))
        ax.plot(x, y, color="#888888", linewidth=0.4, linestyle=(0, (1, 3)),
                 alpha=0.6, zorder=2, clip_on=True)

    # Frame tick positions/labels (bottom edge for longitude, left edge for latitude).
    xtick_pos, xtick_lab = [], []
    for lon in lon_ticks:
        x, _ = to_proj.transform(lon, lat_min)
        if xmin <= x <= xmax:
            xtick_pos.append(x)
            xtick_lab.append(format_lon(lon))

    ytick_pos, ytick_lab = [], []
    for lat in lat_ticks:
        _, y = to_proj.transform(lon_min, lat)
        if ymin <= y <= ymax:
            ytick_pos.append(y)
            ytick_lab.append(format_lat(lat))

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=10)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab, fontsize=10, rotation=90, va="center")
    ax.tick_params(direction="out", length=3, width=0.6)

    # Mirror ticks on the top/right for a closed, publication-style neatline frame.
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(xtick_pos)
    ax_top.set_xticklabels([])
    ax_top.tick_params(direction="in", length=3, width=0.6, labelsize=10)

    ax_right = ax.secondary_yaxis("right")
    ax_right.set_yticks(ytick_pos)
    ax_right.set_yticklabels([])
    ax_right.tick_params(direction="in", length=3, width=0.6, labelsize=10)


# ------------------------------------------------------------------------
# 8. NORTH ARROW
# ------------------------------------------------------------------------
def add_north_arrow(ax, fig, location: str = "upper right"):
    positions = {
        "upper right": (0.93, 0.95),
        "upper left": (0.08, 0.95),
    }
    x, y = positions.get(location, positions["upper right"])
    arrow_len = 0.07

    # Scale font/arrow thickness gently with figure size so it stays proportionate.
    scale_factor = (fig.get_size_inches()[0] + fig.get_size_inches()[1]) / 2.0
    fontsize = max(9, min(14, scale_factor))
    linewidth = max(1.2, min(2.2, scale_factor / 6))

    ax.annotate(
        "N",
        xy=(x, y), xytext=(x, y - arrow_len),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center",
        fontsize=fontsize, fontweight="bold", color="black",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=linewidth,
                         mutation_scale=14 + scale_factor),
        zorder=12,
    )


# ------------------------------------------------------------------------
# 9. SCALE BAR (dynamically computed, never hard-coded)
# ------------------------------------------------------------------------
def add_scale_bar(ax, extent: MapExtent, location: str = "lower center"):
    """
    Draw a dynamically-computed scale bar. Default position is bottom-centre so
    it does not collide with the legend (bottom-left) or the info box (bottom-right).
    """
    xmin, xmax, ymin, ymax = extent.as_tuple()
    map_width_m = xmax - xmin

    candidates_km = [0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]
    target_km = (map_width_m / 1000.0) * 0.18  # a more compact target so it clears the legend/info box
    bar_km = min(candidates_km, key=lambda c: abs(c - target_km))
    bar_m = bar_km * 1000.0

    margin_x = extent.width * 0.06
    # Sits clear ABOVE the bottom row occupied by the legend (bottom-left) and
    # info box (bottom-right), rather than sharing that crowded strip.
    margin_y = extent.height * 0.145

    if location == "lower center":
        x0 = xmin + (extent.width - bar_m) / 2.0
    elif "right" in location:
        x0 = xmax - margin_x - bar_m
    else:
        x0 = xmin + margin_x
    y0 = ymin + margin_y

    bar_height = extent.height * 0.012
    n_segments = 4
    seg_len = bar_m / n_segments

    # White backing plate so the bar reads clearly over any DEM colour beneath it.
    pad = extent.width * 0.01
    ax.add_patch(Rectangle(
        (x0 - pad, y0 - pad * 0.6), bar_m + 2 * pad, bar_height + pad * 2.2,
        facecolor="white", edgecolor="black", linewidth=0.5, alpha=0.85, zorder=10,
    ))

    for i in range(n_segments):
        color = "black" if i % 2 == 0 else "white"
        ax.add_patch(Rectangle(
            (x0 + i * seg_len, y0), seg_len, bar_height,
            facecolor=color, edgecolor="black", linewidth=0.6, zorder=11,
        ))

    ax.text(x0, y0 + bar_height * 1.6, "0", ha="center", va="bottom", fontsize=7, zorder=12)
    ax.text(x0 + bar_m, y0 + bar_height * 1.6, f"{bar_km:g} km",
            ha="center", va="bottom", fontsize=7, zorder=12)


# ------------------------------------------------------------------------
# 10. LEGEND
# ------------------------------------------------------------------------
def add_legend(ax, stream_order_used: bool):
    handles = [
        Line2D([0], [0], color=BOUNDARY_COLOR, lw=BOUNDARY_LINEWIDTH, label="Catchment Boundary"),
        Line2D([0], [0], color=DRAINAGE_COLOR,
               lw=(DRAINAGE_BASE_LINEWIDTH + DRAINAGE_MAX_LINEWIDTH) / 2,
               label="Drainage Network" + (" (width \u221d stream order)" if stream_order_used else "")),
    ]
    legend = ax.legend(
        handles=handles, loc="lower left", fontsize=7.5,
        frameon=True, fancybox=False, edgecolor="black", facecolor="white",
        framealpha=0.85, borderpad=0.6, handlelength=2.2,
    )
    legend.set_zorder(13)


# ------------------------------------------------------------------------
# 11. INFO BOX
# ------------------------------------------------------------------------
def add_info_box(ax, target_crs: str):
    crs_obj = CRS(target_crs)
    text = (
        f"DEM source: {DEM_SOURCE}\n"
        f"Catchment source: {CATCHMENT_SOURCE}\n"
        f"Drainage source: {DRAINAGE_SOURCE}\n"
        f"CRS: {crs_obj.name} ({target_crs})\n"
        f"Data date/version: {DATA_DATE}"
    )
    box_artist = AnchoredText(
        text, loc="lower right", prop=dict(size=6, family=FONT_FAMILY),
        frameon=True, borderpad=0.6, pad=0.5,
    )
    box_artist.patch.set_facecolor("white")
    box_artist.patch.set_alpha(0.85)
    box_artist.patch.set_edgecolor("#666666")
    box_artist.patch.set_linewidth(0.6)
    box_artist.zorder = 13
    ax.add_artist(box_artist)


# ------------------------------------------------------------------------
# 12. OVERVIEW / LOCATION INSET MAP
# ------------------------------------------------------------------------
def _load_natural_earth_fallback():
    """Try a couple of offline-friendly sources for a coarse country layer."""
    try:
        return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        pass
    try:
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        return gpd.read_file(url)
    except Exception:
        return None


def add_overview_map(fig, catchment_gdf: "gpd.GeoDataFrame", target_crs: str, extent: MapExtent):
    """
    Small inset map showing the catchment's location in its wider geographic
    context. Tries an online basemap (contextily/OSM) first; falls back to a
    coarse Natural Earth country outline; falls back again to a simple
    labelled placeholder if fully offline.
    """
    #inset_ax = fig.add_axes([0.065, 0.62, 0.24, 0.24])
    inset_ax = fig.add_axes([0.60, 0.24, 0.24, 0.22])
    inset_ax.set_facecolor("white")
    for spine in inset_ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")

    catchment_wgs84 = catchment_gdf.to_crs(4326)
    cxmin, cymin, cxmax, cymax = catchment_wgs84.total_bounds
    pad_lon = max((cxmax - cxmin) * 6, 2.0)
    pad_lat = max((cymax - cymin) * 6, 2.0)
    view_bounds_wgs84 = (cxmin - pad_lon, cymin - pad_lat, cxmax + pad_lon, cymax + pad_lat)

    basemap_drawn = False

    if HAS_CONTEXTILY:
        try:
            catchment_3857 = catchment_gdf.to_crs(3857)
            cx3min, cy3min, cx3max, cy3max = catchment_3857.total_bounds
            pad_x = max((cx3max - cx3min) * 6, 200_000)
            pad_y = max((cy3max - cy3min) * 6, 200_000)
            inset_ax.set_xlim(cx3min - pad_x, cx3max + pad_x)
            inset_ax.set_ylim(cy3min - pad_y, cy3max + pad_y)
            ctx.add_basemap(inset_ax, source=ctx.providers.OpenStreetMap.Mapnik,
                             crs="EPSG:3857", attribution=False)
            catchment_3857.plot(ax=inset_ax, facecolor="none",
                                 edgecolor=BOUNDARY_COLOR, linewidth=1.2, zorder=5)
            centroid = catchment_3857.geometry.iloc[0].centroid if len(catchment_3857) == 1 \
                else _dissolve_geometry(catchment_3857).centroid
            inset_ax.plot(centroid.x, centroid.y, marker="*", color=BOUNDARY_COLOR,
                          markersize=9, zorder=6)
            basemap_drawn = True
        except Exception as exc:
            print(f"[Overview] Online basemap unavailable ({exc}); falling back to Natural Earth outline.")

    if not basemap_drawn:
        world = _load_natural_earth_fallback()
        if world is not None:
            try:
                world.plot(ax=inset_ax, color="#e6e6e6", edgecolor="#999999", linewidth=0.4, zorder=1)
                inset_ax.set_xlim(view_bounds_wgs84[0], view_bounds_wgs84[2])
                inset_ax.set_ylim(view_bounds_wgs84[1], view_bounds_wgs84[3])
                catchment_wgs84.plot(ax=inset_ax, facecolor=BOUNDARY_COLOR, edgecolor=BOUNDARY_COLOR,
                                      alpha=0.9, linewidth=1.0, zorder=5)
                centroid = catchment_wgs84.geometry.iloc[0].centroid if len(catchment_wgs84) == 1 \
                    else _dissolve_geometry(catchment_wgs84).centroid
                inset_ax.plot(centroid.x, centroid.y, marker="*", color="black", markersize=8, zorder=6)
                basemap_drawn = True
            except Exception as exc:
                print(f"[Overview] Natural Earth fallback failed ({exc}); using placeholder inset.")

    if not basemap_drawn:
        # Final, fully-offline fallback: just show the catchment shape with a note.
        catchment_wgs84.plot(ax=inset_ax, facecolor=BOUNDARY_COLOR, edgecolor=BOUNDARY_COLOR, alpha=0.8)
        inset_ax.text(0.5, 0.04, "Basemap unavailable\n(offline)",
                      transform=inset_ax.transAxes, ha="center", va="bottom",
                      fontsize=5.5, style="italic", color="#444444")

    inset_ax.set_title("Location", fontsize=7.5, pad=2)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])


# ------------------------------------------------------------------------
# 13. DRAINAGE NETWORK PLOTTING (stream-order-aware)
# ------------------------------------------------------------------------
def plot_drainage(ax, drainage_gdf: "gpd.GeoDataFrame"):
    order_field = find_stream_order_field(drainage_gdf)

    if order_field is None:
        drainage_gdf.plot(ax=ax, color=DRAINAGE_COLOR, linewidth=DRAINAGE_BASE_LINEWIDTH,
                           zorder=6)
        return False

    orders = drainage_gdf[order_field].astype(float)
    omin, omax = orders.min(), orders.max()
    if omax == omin:
        drainage_gdf.plot(ax=ax, color=DRAINAGE_COLOR, linewidth=DRAINAGE_BASE_LINEWIDTH,
                           zorder=6)
        return False

    for order_value in sorted(orders.unique()):
        subset = drainage_gdf[orders == order_value]
        frac = (order_value - omin) / (omax - omin)
        lw = DRAINAGE_BASE_LINEWIDTH + frac * (DRAINAGE_MAX_LINEWIDTH - DRAINAGE_BASE_LINEWIDTH)
        subset.plot(ax=ax, color=DRAINAGE_COLOR, linewidth=lw, zorder=6)
    return True


# ------------------------------------------------------------------------
# 14. MAIN MAP ASSEMBLY
# ------------------------------------------------------------------------
def create_map(catchment_gdf, drainage_gdf, dem_array, dem_transform, target_crs, extent: MapExtent):
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1.0

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax = fig.add_axes([0.10, 0.09, 0.76, 0.80])
    ax.set_aspect("equal")  # preserves correct geographic proportions, avoids distortion

    # --- DEM + hillshade blended relief -------------------------------------
    rgba, elev_min, elev_max, cmap = compute_shaded_relief(dem_array, dem_transform)
    dem_h, dem_w = dem_array.shape
    dem_extent = dem_extent_from_transform(dem_transform, dem_w, dem_h)
    ax.imshow(rgba, extent=dem_extent, origin="upper", zorder=1, interpolation="bilinear")

    # --- Catchment subtle fill + boundary ------------------------------------
    catchment_gdf.plot(ax=ax, facecolor=BOUNDARY_COLOR, edgecolor="none",
                        alpha=BOUNDARY_FILL_ALPHA, zorder=3)
    catchment_gdf.boundary.plot(ax=ax, color=BOUNDARY_COLOR, linewidth=BOUNDARY_LINEWIDTH,
                                 zorder=7, linestyle="-")

    # --- Drainage network -----------------------------------------------------
    stream_order_used = plot_drainage(ax, drainage_gdf)

    # --- Extent / frame ---------------------------------------------------------
    ax.set_xlim(extent.xmin, extent.xmax)
    ax.set_ylim(extent.ymin, extent.ymax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # --- Graticule --------------------------------------------------------------
    if SHOW_GRID:
        add_graticule(ax, target_crs, extent)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # --- North arrow --------------------------------------------------------------
    if SHOW_NORTH_ARROW:
        add_north_arrow(ax, fig, location="upper right")

    # --- Scale bar --------------------------------------------------------------
    if SHOW_SCALE_BAR:
        add_scale_bar(ax, extent, location="lower center")
        add_scale_bar(ax, extent, location="lower left")

    # --- Legend --------------------------------------------------------------------
    add_legend(ax, stream_order_used)

    # --- Info box --------------------------------------------------------------
    if SHOW_INFO_BOX:
        add_info_box(ax, target_crs)

    # --- Elevation colourbar -----------------------------------------------------
    cbar_ax = fig.add_axes([0.88, 0.20, 0.025, 0.45])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=elev_min, vmax=elev_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Elevation (m)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7)

    # --- Overview inset map --------------------------------------------------
    if SHOW_OVERVIEW_MAP:
        add_overview_map(fig, catchment_gdf, target_crs, extent)

    # --- Title ------------------------------------------------------------------
    title = MAP_TITLE_TEMPLATE.format(name=CATCHMENT_NAME)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.965)
    ax.set_xlabel("Longitude", fontsize=8.5, labelpad=6)
    ax.set_ylabel("Latitude", fontsize=8.5, labelpad=6)

    return fig, elev_min, elev_max


# ------------------------------------------------------------------------
# 15. OUTPUT
# ------------------------------------------------------------------------
def save_outputs(fig, output_dir: Path, basename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ("png", "pdf", "svg"):
        out_path = output_dir / f"{basename}.{ext}"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        paths[ext] = out_path
    return paths


# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------
def main():
    catchment_path = Path(CATCHMENT_FILE)
    drainage_path = Path(DRAINAGE_FILE)
    dem_path = Path(DEM_FILE)
    output_dir = Path(OUTPUT_DIR)

    print("=" * 70)
    print(f"Generating catchment map for: {CATCHMENT_NAME}")
    print("=" * 70)

    # 1. Validate ---------------------------------------------------------------
    validate_inputs(catchment_path, drainage_path, dem_path)

    # 2. Load catchment & determine target CRS -----------------------------------
    catchment_raw = gpd.read_file(catchment_path)
    catchment_raw = ensure_crs(catchment_raw, "catchment")
    target_crs = resolve_target_crs(catchment_raw)

    # 3. Reproject vector layers --------------------------------------------------
    catchment_gdf = load_vector(catchment_path, "catchment", target_crs)
    drainage_gdf = load_vector(drainage_path, "drainage", target_crs)

    # 4. Load & reproject DEM, then clip ------------------------------------------
    memfile, dem_dataset = reproject_dem_if_needed(dem_path, target_crs)
    try:
        dem_array, dem_transform = clip_dem_to_catchment(dem_dataset, catchment_gdf, CLIP_BUFFER_KM)
    finally:
        dem_dataset.close()
        if memfile is not None:
            memfile.close()

    # 5. Map extent -----------------------------------------------------------------
    extent = compute_map_extent(catchment_gdf, MAP_MARGIN)

    # 6. Build the map ----------------------------------------------------------------
    fig, elev_min, elev_max = create_map(
        catchment_gdf, drainage_gdf, dem_array, dem_transform, target_crs, extent
    )

    # 7. Save outputs -----------------------------------------------------------------
    output_paths = save_outputs(fig, output_dir, OUTPUT_BASENAME)
    plt.close(fig)

    # 8. Summary ------------------------------------------------------------------------
    print("\nMap successfully generated.\n")
    print("Outputs:")
    for ext, path in output_paths.items():
        print(f"  \u2713 {ext.upper()}: {path}")
    print(f"\nCRS: {target_crs}")
    print(f"Map extent (projected units): "
          f"xmin={extent.xmin:.1f}, xmax={extent.xmax:.1f}, ymin={extent.ymin:.1f}, ymax={extent.ymax:.1f}")
    print(f"Elevation range: {elev_min:.1f} - {elev_max:.1f} m")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] Map generation failed: {exc}", file=sys.stderr)
        sys.exit(1)
