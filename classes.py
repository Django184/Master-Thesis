import glob
import os
import re
from datetime import datetime
from pathlib import Path

import contextily as cx
import gstatsim as gs
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
import scipy.stats
import seaborn as sns
import skgstat as skg
from affine import Affine
from matplotlib import patheffects as pe
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pykrige.ok import OrdinaryKriging
from rasterio.features import rasterize
from rasterio.plot import plotting_extent
from rasterio.transform import xy
from rasterio.warp import Resampling
from rasterio.warp import reproject
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator  # discrete label fill
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import QhullError
from scipy.spatial import cKDTree
from scipy.spatial import distance as dist
from scipy.stats import f_oneway
from shapely.geometry import Polygon
from skgstat import models
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import QuantileTransformer


def _find_tdr_dir():
    # support both "Data" and "data" folders (Windows OK; Linux is case-sensitive)
    candidates = [Path("Data") / "VWC verification", Path("data") / "VWC verification"]
    for d in candidates:
        if d.exists() and d.is_dir():
            return d
    raise FileNotFoundError(
        "Could not find 'VWC verification' directory under 'Data/' or 'data/'. " "Please verify the folder location."
    )


_TDR_DIR = _find_tdr_dir()

# normalise to absolute OS-native paths; keep only real files
TDR_PATHS = sorted(p.resolve(strict=True) for p in _TDR_DIR.glob("*.xlsx") if p.is_file())


# =========================
# File discovery helpers
# =========================
def _glob_tifs_ci(root, sub, recursive=True):
    """
    Case-insensitive *.tif search under root/sub (returns sorted list of Path-like strings).
    """
    pat = os.path.join(root, sub, "**", "*.[tT][iI][fF]")
    if not recursive:
        pat = os.path.join(root, sub, "*.[tT][iI][fF]")
    return sorted(glob.glob(pat, recursive=recursive))


def _glob_multi(*patterns):
    out = []
    for p in patterns:
        out.extend(glob.glob(p, recursive=True))
    return sorted(out)


# ---- RASTER FILE LISTS (define before any code uses/prints them) ----
# Thermal (all common TIFF casings, recursively)
TEMP_RASTER = _glob_multi(
    os.path.join("Data", "thermal", "**", "*.tif"),
    os.path.join("Data", "thermal", "**", "*.TIF"),
    os.path.join("Data", "thermal", "**", "*.tiff"),
    os.path.join("Data", "thermal", "**", "*.TIFF"),
)

# Multispectral (recursive, case-insensitive *.tif)
NDVI_RASTER = _glob_tifs_ci("Data", "Multispectral/NDVI")
BLUE_RASTER = _glob_tifs_ci("Data", "Multispectral/Blue")
GREEN_RASTER = _glob_tifs_ci("Data", "Multispectral/Green")
RED_RASTER = _glob_tifs_ci("Data", "Multispectral/Red")
NIR_RASTER = _glob_tifs_ci("Data", "Multispectral/NIR")
RED_EDGE_RASTER = _glob_tifs_ci("Data", "Multispectral/Rededge")

# Optional debug counts (gate with a flag so importing this module is quiet by default)
_DEBUG_RASTER_LISTS = False
if _DEBUG_RASTER_LISTS:
    print("[DEBUG] NDVI_RASTER count:", len(NDVI_RASTER))
    if NDVI_RASTER:
        print("  e.g.", NDVI_RASTER[0])
    print("[DEBUG] TEMP_RASTER count:", len(TEMP_RASTER))
    if TEMP_RASTER:
        print("  e.g.", TEMP_RASTER[0])

# =========================
# GPR analysis
# =========================
GPR_A_PATHS = sorted(glob.glob(os.path.join("Data", "Drone GPR", "Field A", "*.txt")))
GPR_B_PATHS = sorted(glob.glob(os.path.join("Data", "Drone GPR", "Field B", "*.txt")))


class GprAnalysis:
    """Visualization & analysis utilities for drone GPR field data."""

    # ---------- construction ----------
    def __init__(self, field_letter="A", sample_number=0):
        if field_letter not in {"A", "B"}:
            raise ValueError("field_letter must be either 'A' or 'B'")
        self.field_letter = field_letter
        self.sample_number = int(sample_number)

        self.field_paths = GPR_A_PATHS if self.field_letter == "A" else GPR_B_PATHS

        # caches
        self._data_cache = None  # list[pd.DataFrame] in original order
        self._dates_cache = None  # list[str] DD/MM/YYYY
        self._utm_transformer_cache = {}  # key=(zone, hemi)->Transformer

    @staticmethod
    def _mad_std(a):
        """
        Robust spread estimate from Median Absolute Deviation (≈ standard deviation).
        """
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan
        med = np.nanmedian(a)
        mad = np.nanmedian(np.abs(a - med))
        return 1.4826 * mad

    # ---------- small utils ----------
    @staticmethod
    def _parse_date_from_filename(fname):
        """
        Try to parse a date from a GPR file name.
        Primary: any 6 consecutive digits interpreted as YYMMDD.
        Fallback: legacy fixed slicing [8:10]=DD, [6:8]=MM, [4:6]=YY.
        Returns dd/mm/YYYY string (or 'unknown date').
        """
        base = os.path.splitext(os.path.basename(fname))[0]
        m = re.search(r"(\d{6})", base)  # YYMMDD somewhere in the name
        if m:
            y, mo, d = m.group(1)[:2], m.group(1)[2:4], m.group(1)[4:]
            try:
                dt = datetime.strptime(f"20{y}-{mo}-{d}", "%Y-%m-%d").date()
                return dt.strftime("%d/%m/%Y")
            except Exception:
                pass
        # fallback to brittle legacy slicing
        try:
            d = base[8:10]
            mo = base[6:8]
            y = base[4:6]
            dt = datetime.strptime(f"20{y}-{mo}-{d}", "%Y-%m-%d").date()
            return dt.strftime("%d/%m/%Y")
        except Exception:
            return "unknown date"

    @staticmethod
    def _auto_utm_epsg(lon, lat):
        """
        Compute UTM EPSG code from lon/lat arrays (north=326xx, south=327xx).
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        if lon.size == 0 or lat.size == 0 or not np.isfinite(lon).any() or not np.isfinite(lat).any():
            # fallback to UTM 32N if we can't infer
            return 32632
        lon_m = float(np.nanmean(lon))
        lat_m = float(np.nanmean(lat))
        zone = int(np.floor((lon_m + 180.0) / 6.0) + 1)
        epsg = (32600 if lat_m >= 0 else 32700) + zone
        return epsg

    def _get_transformer(self, lon, lat, epsg=None):
        """
        Get (and cache) a pyproj Transformer for WGS84->UTM.
        If epsg is None, auto-detect UTM zone from the data.
        """
        if epsg is None:
            epsg = self._auto_utm_epsg(lon, lat)
        hemi = "N" if epsg // 100 == 326 else "S"
        key = (epsg, hemi)
        if key not in self._utm_transformer_cache:
            self._utm_transformer_cache[key] = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        return self._utm_transformer_cache[key]

    # ---------- IO ----------
    def import_data(self, show=False):
        """
        Read all GPR files for the field (cached). Tolerant whitespace parsing.
        Returns list of DataFrames with columns ['y','x','vwc'] (as in your original).
        """
        if self._data_cache is not None:
            if show:
                print(self._data_cache)
            return self._data_cache

        tbl = []
        for p in self.field_paths:
            # Tolerant parse: any whitespace, take first 3 columns
            df = pd.read_csv(p, sep=r"\s+", engine="python", header=None)
            if df.shape[1] < 3:
                # try original heuristic (double space)
                df = pd.read_csv(p, sep="  ", engine="python", header=None)
            # truncate to 3 cols, name like original
            df = df.iloc[:, :3].copy()
            df.columns = ["y", "x", "vwc"]
            # coerce numeric
            for c in ("x", "y", "vwc"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            tbl.append(df)

        self._data_cache = tbl
        if show:
            print(tbl)
        return tbl

    def extract_dates(self):
        """
        Extract per-file dates as DD/MM/YYYY strings (cached).
        """
        if self._dates_cache is not None:
            return self._dates_cache
        dates = [self._parse_date_from_filename(p) for p in self.field_paths]
        self._dates_cache = dates
        return dates

    # ---------- coordinates ----------
    def convert_to_utm(self, lon, lat, epsg=None, normalize_origin=True):
        """
        Convert lon/lat arrays to UTM metres. Auto-detect UTM zone unless epsg is given.
        Returns (x, y) arrays; optionally shifts to start at 0 m.
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        tr = self._get_transformer(lon, lat, epsg=epsg)
        # always_xy=True -> transform(lon, lat)
        x, y = tr.transform(lon, lat)
        if normalize_origin and x.size and y.size:
            x = x - np.nanmin(x)
            y = y - np.nanmin(y)
        return x, y

    def _field_polygon_utm(self):
        """
        Field outline polygon in the same UTM frame you use for GPR.
        Returns (poly_xy ndarray shape (N,2), xlim_max, ylim_max).
        """
        if self.field_letter == "A":
            poly = np.array([[75, 0], [190, 110], [120, 210], [60, 225], [0, 175], [75, 0]], dtype=float)
            xlim, ylim = 200, 250
        else:
            poly = np.array([[70, 0], [150, 75], [90, 175], [10, 130], [70, 0]], dtype=float)
            xlim, ylim = 150, 175
        return poly, xlim, ylim

    # ---------- plots ----------
    def plot_raw_data(self):
        """Plot the raw GPR data (multispectral-style map)."""
        df = self.import_data()[self.sample_number]

        # Your file columns are named ['y','x','vwc']; here we treat x=lon, y=lat
        utm_x, utm_y = self.convert_to_utm(df["x"].values, df["y"].values)

        poly, _, _ = self._field_polygon_utm()
        poly_closed = np.vstack([poly, poly[0]])

        date_str = self.extract_dates()[self.sample_number]
        title = f"GPR Raw • Field {self.field_letter} • {date_str}"

        fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=300)
        sc = ax.scatter(utm_x, utm_y, c=df["vwc"].values, s=12, cmap="viridis_r")

        # thin black outline
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], "k-", linewidth=1.5)

        ax.set_aspect("equal", adjustable="box")
        pad = 5
        ax.set_xlim(min(utm_x.min(), poly[:, 0].min()) - pad, max(utm_x.max(), poly[:, 0].max()) + pad)
        ax.set_ylim(min(utm_y.min(), poly[:, 1].min()) - pad, max(utm_y.max(), poly[:, 1].max()) + pad)

        self._apply_map_grid(ax, major=100, minor=50, labelsize=4)
        self._add_north_arrow(ax, loc="NE", size=0.09, inset=(0.03, 0.01), font_size=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("VWC [-]")
        ax.set_title(title, pad=10)
        plt.show()

    def plot_mean_median(self, plot=True):
        """Compute & optionally plot GPR mean and median across dates."""
        studied_field = self.import_data()
        mean_evolution = [np.nanmean(df["vwc"].values) for df in studied_field]
        median_evolution = [np.nanmedian(df["vwc"].values) for df in studied_field]

        dates = pd.to_datetime(self.extract_dates(), format="%d/%m/%Y", errors="coerce")

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(dates, median_evolution, marker="o", label="Median")
            plt.plot(dates, mean_evolution, marker="o", label="Mean")
            plt.xlabel("Date")
            plt.ylabel("VWC [/]")
            plt.title(f"Evolution of GPR derived VWC - Field {self.field_letter}")
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return mean_evolution, median_evolution

    def zonal_check(self):
        """Split a polygonal zone vs outside and compare medians over time (with variance bars)."""
        if self.field_letter == "A":
            polygon_coords = np.array([[0, 50], [150, 200], [75, 250], [0, 200], [0, 50]], dtype=float)
        else:
            polygon_coords = np.array([[30, 50], [140, 125], [140, 200], [0, 200], [0, 125], [30, 50]], dtype=float)

        zone_path = mpath.Path(polygon_coords)
        self.plot_raw_data_by_zone(zone_path)

        upper_evolution, lower_evolution = [], []
        upper_sigma, lower_sigma = [], []
        for df in self.import_data():
            utm_x, utm_y = self.convert_to_utm(df["x"].values, df["y"].values)
            pts = np.column_stack([utm_x, utm_y])
            inside = zone_path.contains_points(pts)
            upper = df["vwc"].values[inside]
            lower = df["vwc"].values[~inside]
            upper_evolution.append(np.nanmedian(upper) if upper.size else np.nan)
            lower_evolution.append(np.nanmedian(lower) if lower.size else np.nan)
            upper_sigma.append(self._mad_std(upper))
            lower_sigma.append(self._mad_std(lower))

        dates = pd.to_datetime(self.extract_dates(), format="%d/%m/%Y", errors="coerce")
        plt.figure(figsize=(8, 6))
        (ln_u,) = plt.plot(dates, upper_evolution, marker="o", label="Zone 2")
        (ln_l,) = plt.plot(dates, lower_evolution, marker="o", label="Zone 1")
        plt.errorbar(
            dates,
            upper_evolution,
            yerr=upper_sigma,
            fmt="none",
            elinewidth=1,
            capsize=2,
            alpha=0.7,
            ecolor=ln_u.get_color(),
        )
        plt.errorbar(
            dates,
            lower_evolution,
            yerr=lower_sigma,
            fmt="none",
            elinewidth=1,
            capsize=2,
            alpha=0.7,
            ecolor=ln_l.get_color(),
        )

        plt.xlabel("Date")
        plt.ylabel("VWC [/]")
        plt.title(f"Evolution of GPR VWC by zone - Field {self.field_letter}")
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_raw_data_by_zone(self, zone_1):
        """Plot the raw GPR data split by a Path polygon (zone_1 vs outside)."""
        df = self.import_data()[self.sample_number]
        utm_x, utm_y = self.convert_to_utm(df["x"].values, df["y"].values)
        pts = np.column_stack([utm_x, utm_y])
        inside = zone_1.contains_points(pts)

        z1_x, z1_y, z1_v = utm_x[inside], utm_y[inside], df["vwc"].values[inside]
        z2_x, z2_y, z2_v = utm_x[~inside], utm_y[~inside], df["vwc"].values[~inside]

        plt.figure(figsize=(10, 6))
        sc1 = plt.scatter(z1_x, z1_y, c=z1_v, cmap="viridis_r", label="Zone 2")
        sc2 = plt.scatter(z2_x, z2_y, c=z2_v, cmap="BrBG_r", label="Zone 1")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title(f"GPR sampling by zone - Field {self.field_letter} ({self.extract_dates()[self.sample_number]})")
        cb1 = plt.colorbar(sc1)
        cb1.set_label("Zone 2 VWC [/]")
        cb2 = plt.colorbar(sc2)
        cb2.set_label("Zone 1 VWC [/]")
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _standardize_tdr_frame(self, df):
        """
        Normalize TDR dataframe to have columns: Lat, Lon, VWC, SD (case-insensitive).
        Returns filtered frame for current field (A/B) with those columns only.
        """
        # Build a case-insensitive rename map
        rename = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in ("lat",):
                rename[c] = "Lat"
            elif cl in ("lon", "long", "longitude"):
                rename[c] = "Lon"
            elif cl in ("vwc", "sm", "soilmoisture", "soil_moisture"):
                rename[c] = "VWC"
            elif cl in ("sd", "std", "stddev", "stdev", "sigma"):
                rename[c] = "SD"

        df = df.rename(columns=rename)

        # Keep only the expected columns if present
        keep = [c for c in ["Lat", "Lon", "VWC", "SD"] if c in df.columns]
        if not keep or ("Lat" not in keep or "Lon" not in keep or "VWC" not in keep):
            raise ValueError("TDR file must contain at least Lat, Lon, VWC columns (SD optional).")

        df = df[keep].copy()

        # Numeric coercion
        for c in keep:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Field split by latitude threshold (your rule)
        if self.field_letter == "A":
            df = df[df["Lat"] < 50.496773]
        else:
            df = df[df["Lat"] >= 50.496773]

        return df.dropna(subset=["Lat", "Lon", "VWC"])

    def kriging(self, plot=True):
        """Ordinary Kriging interpolation (multispectral-style map)."""
        df = self.import_data()[self.sample_number]
        utm_x, utm_y = self.convert_to_utm(df["x"].values, df["y"].values)

        # grid params (kept as in your code)
        x_min, x_max = 0.0, 250.0
        y_min, y_max = 0.0, 250.0
        step = 1.0
        grid_x = np.arange(x_min, x_max, step)
        grid_y = np.arange(y_min, y_max, step)

        poly, xlim_max, ylim_max = self._field_polygon_utm()
        poly_closed = np.vstack([poly, poly[0]])

        # vectorized mask (True outside polygon) for pykrige's "masked" mode
        gx, gy = np.meshgrid(grid_x, grid_y)  # note: shape (ny, nx)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        pth = mpath.Path(poly)
        inside = pth.contains_points(pts).reshape(gx.shape)
        mask = ~inside  # pykrige 'masked' expects True where masked

        ok = OrdinaryKriging(
            utm_x, utm_y, df["vwc"].values, variogram_model="exponential", verbose=False, enable_plotting=False
        )
        z, ss = ok.execute("masked", grid_x, grid_y, mask=mask)

        if plot:
            date_str = self.extract_dates()[self.sample_number]
            title = f"Kriging • Field {self.field_letter} • {date_str}"

            fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=300)
            im = ax.imshow(z, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap="viridis_r", aspect="equal")

            # thicker outline to hide edge irregularities
            ax.plot(poly_closed[:, 0], poly_closed[:, 1], "k-", linewidth=5.0)

            ax.set_xlim(-5, xlim_max)
            ax.set_ylim(-5, ylim_max)
            self._apply_map_grid(ax, major=100, minor=50, labelsize=4)
            self._add_north_arrow(ax, loc="NE", size=0.09, inset=(0.03, 0.01), font_size=9)
            ax.set_xlabel("")
            ax.set_ylabel("")

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("VWC [-]")
            ax.set_title(title, pad=10)
            plt.show()

    # ---------- verification & correlations ----------
    def import_tdr_data(self):
        """
        Import TDR data for the campaign corresponding to self.sample_number.
        Keeps your existing 'sample_number - 3' offset logic, but with guards.
        Returns columns renamed to ['y','x','vwc','sd'] for consistency.
        """
        if "TDR_PATHS" not in globals() or not TDR_PATHS:
            raise RuntimeError(
                "TDR_PATHS is not defined or empty. Make sure the *.xlsx files are in 'Data/VWC verification/'."
            )

        idx = self.sample_number - 3
        if idx < 0 or idx >= len(TDR_PATHS):
            raise IndexError(
                f"TDR index {idx} out of range for TDR_PATHS of length {len(TDR_PATHS)}. "
                "Check your sample_number or add more TDR files."
            )

        path = TDR_PATHS[idx]  # <- this is a pathlib.Path now
        if not path.exists():
            raise FileNotFoundError(f"TDR file not found: {path}")

        # Pandas can read pathlib.Path directly
        tdr_data_AB = pd.read_excel(path)

        # Split A/B by latitude threshold (your original behaviour)
        if self.field_letter == "A":
            tdr_data = tdr_data_AB[tdr_data_AB["Lat"].values < 50.496773].copy()
        else:
            tdr_data = tdr_data_AB[tdr_data_AB["Lat"].values >= 50.496773].copy()

        # Make columns consistent with the rest of your class
        # Expecting columns: Lat, Lon, VWC, SD (rename to y,x,vwc,sd)
        # If your Excel headers differ, tweak the mapping below.
        rename_map = {}
        for col in tdr_data.columns:
            c = str(col).strip().lower()
            if c == "lat":
                rename_map[col] = "y"
            elif c == "lon":
                rename_map[col] = "x"
            elif c == "vwc":
                rename_map[col] = "vwc"
            elif c in ("sd", "std", "stddev", "stdev"):
                rename_map[col] = "sd"
        tdr_data = tdr_data.rename(columns=rename_map)

        # keep only the needed columns and enforce order if present
        want = [c for c in ["y", "x", "vwc", "sd"] if c in tdr_data.columns]
        tdr_data = tdr_data[want].copy()

        # final sanity: numeric types
        for c in ("x", "y", "vwc", "sd"):
            if c in tdr_data:
                tdr_data[c] = pd.to_numeric(tdr_data[c], errors="coerce")

        return tdr_data

    def _add_north_arrow(self, ax, loc="NE", size=0.09, color="k", inset=(0.03, 0.01), font_size=9):
        corners = {"NE": (1.0, 1.0), "NW": (0.0, 1.0), "SE": (1.0, 0.0), "SW": (0.0, 0.0)}
        cx, cy = corners.get(loc, corners["NE"])
        x = cx - inset[0] if cx == 1.0 else cx + inset[0]
        y_top = cy - inset[1] if cy == 1.0 else cy + inset[1]
        y_bot = y_top - size if cy == 1.0 else y_top + size
        arrow_kwargs = dict(
            facecolor=color, edgecolor=color, width=1.0, headwidth=8, headlength=10, shrinkA=0, shrinkB=0
        )
        ax.annotate(
            "N",
            xy=(x, y_top),
            xytext=(x, y_bot),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            ha="center",
            va="center",
            color=color,
            fontsize=font_size,
            arrowprops=arrow_kwargs,
            annotation_clip=False,
        )

    def _apply_map_grid(self, ax, major=100, minor=50, labelsize=4):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.set_autoscale_on(False)

        def _ticks_within(lo, hi, step):
            if not step or step <= 0:
                return np.array([])
            start = np.ceil(min(lo, hi) / step) * step
            stop = np.floor(max(lo, hi) / step) * step
            if stop < start:
                return np.array([])
            return np.arange(start, stop + 0.5 * step, step)

        ax.set_xticks(_ticks_within(*xlim, major))
        ax.set_yticks(_ticks_within(*ylim, major))
        if minor and minor > 0:
            ax.set_xticks(_ticks_within(*xlim, minor), minor=True)
            ax.set_yticks(_ticks_within(*ylim, minor), minor=True)

        ax.grid(which="major", linewidth=0.5, alpha=0.35)
        ax.grid(which="minor", linewidth=0.4, alpha=0.15)
        ax.tick_params(top=True, right=True, direction="out", length=3, labelsize=labelsize)
        ax.tick_params(which="minor", length=2)

        fmtx = ScalarFormatter(useMathText=False)
        fmty = ScalarFormatter(useMathText=False)
        fmtx.set_scientific(False)
        fmty.set_scientific(False)
        fmtx.set_useOffset(False)
        fmty.set_useOffset(False)
        ax.xaxis.set_major_formatter(fmtx)
        ax.yaxis.set_major_formatter(fmty)
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.margins(0)

    def tdr_verification(self, verification_radius=10):
        """
        Compare Kriging VWC medians within 'verification_radius' of each TDR point.
        Uses kriged surface instead of raw GPR data and plots:
        (1) bar chart of kriged vs TDR,
        (2) map with kriged field background + TDR points (+ optional GPR samples).
        """
        # --- Kriging on current campaign ---
        df = self.import_data()[self.sample_number]
        utm_x, utm_y = self.convert_to_utm(df["x"].values, df["y"].values)

        # grid params
        x_min, x_max = 0.0, 250.0
        y_min, y_max = 0.0, 250.0
        step = 1.0
        grid_x = np.arange(x_min, x_max, step)
        grid_y = np.arange(y_min, y_max, step)
        gx, gy = np.meshgrid(grid_x, grid_y)

        poly, xlim_max, ylim_max = self._field_polygon_utm()
        poly_closed = np.vstack([poly, poly[0]])
        pth = mpath.Path(poly)
        mask = ~pth.contains_points(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)

        ok = OrdinaryKriging(
            utm_x, utm_y, df["vwc"].values, variogram_model="exponential", verbose=False, enable_plotting=False
        )
        z, ss = ok.execute("masked", grid_x, grid_y, mask=mask)

        # --- TDR data & kriged sampling around each TDR point ---
        tdr_data = self.import_tdr_data()
        tdr_xs, tdr_ys = self.convert_to_utm(tdr_data["x"].values, tdr_data["y"].values)
        krig_vals = []
        for tx, ty in zip(tdr_xs, tdr_ys):
            dist = np.sqrt((gx - tx) ** 2 + (gy - ty) ** 2)
            within = dist < verification_radius
            vals = z[within]
            krig_vals.append(np.nanmedian(vals) if np.isfinite(vals).any() else 0.0)

        # Filter out missing / out-of-domain extractions
        tdr_vwcs = list(tdr_data["vwc"].values)
        tdr_sds = list(tdr_data["sd"].values)
        for i in range(len(krig_vals) - 1, -1, -1):
            if krig_vals[i] == 0.0 or np.isnan(krig_vals[i]):
                krig_vals.pop(i)
                tdr_vwcs.pop(i)
                tdr_sds.pop(i)

        date = self.extract_dates()[self.sample_number]

        # --- (1) Bar Plot: Kriged vs TDR ---
        plt.figure(figsize=(10, 6))
        x = np.arange(len(krig_vals))
        width = 0.25
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        col_krig, col_tdr = colors[0], colors[1]

        plt.bar(x, krig_vals, width, label="Kriged GPR", color=col_krig)
        plt.bar(
            x + width,
            tdr_vwcs,
            width,
            label="TDR",
            yerr=tdr_sds,
            color=col_tdr,
            error_kw=dict(ecolor=col_tdr, elinewidth=1, capsize=2),
        )
        plt.ylabel("VWC [m³/m³]")
        plt.xlabel("TDR verification points")
        plt.title(f"Kriged GPR vs TDR (radius={verification_radius} m) • Field {self.field_letter} • {date}")
        plt.legend()
        plt.ylim(0, 1.15)
        plt.tight_layout()
        plt.show()

        # --- (2) Map Plot: kriged field + TDR points (+ buffers) (+ optional GPR samples) ---
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

        im = ax.imshow(z, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap="viridis_r", aspect="equal")

        # field outline (thicker to hide masked edge)
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], "k-", linewidth=5.0, alpha=0.9)

        # optional: overlay original GPR sample locations for context (faint)
        ax.scatter(utm_x, utm_y, s=8, c="k", alpha=0.25, label="GPR samples")

        # TDR points + search radius circles
        ax.scatter(tdr_xs, tdr_ys, c="red", marker="s", s=40, label="TDR")
        for tx, ty in zip(tdr_xs, tdr_ys):
            circ = patches.Circle((tx, ty), verification_radius, fill=False, linestyle="--", linewidth=0.9)
            ax.add_patch(circ)

        # formatting
        ax.set_xlim(-5, xlim_max)
        ax.set_ylim(-5, ylim_max)
        self._apply_map_grid(ax, major=100, minor=50, labelsize=6)
        self._add_north_arrow(ax, loc="NE", size=0.09, inset=(0.03, 0.01), font_size=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"GPR sampling (kriged field) • Field {self.field_letter} • {date}", pad=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Kriged VWC [m³/m³]")

        ax.legend(loc="lower left", frameon=True)
        plt.tight_layout()
        plt.show()

    # ---------- correlations ----------
    def correlate_gpr_terros(self, plot=True):
        """Correlation of GPR (mean/median) vs Teros daily median (all probes averaged) + RMSE/MAE.
        Adds variance bars (robust σ via MAD) to the evolution plot.
        """
        gpr_mean, gpr_median = self.plot_mean_median(plot=False)
        # per-campaign robust sigma for GPR
        gpr_sig = [self._mad_std(df["vwc"].values) for df in self.import_data()]
        dates = pd.to_datetime(self.extract_dates(), format="%d/%m/%Y", errors="coerce")

        terros = pd.read_csv("Data/Teros Piezo/teros_piezo.csv")
        terros["Dates (hours)"] = pd.to_datetime(terros["Dates (hours)"], errors="coerce")
        terros["Date"] = terros["Dates (hours)"].dt.date

        # Daily median per probe -> mean across probes (unchanged)
        vcols = [c for c in terros.columns if c.startswith("T_")]
        per_day_per_probe = terros.groupby("Date")[vcols].median()
        terros_median = per_day_per_probe.mean(axis=1)
        # robust σ across probes (per day)
        terros_sigma = per_day_per_probe.apply(lambda row: self._mad_std(row.values), axis=1)

        # Align by GPR dates (as before)
        mask = terros_median.index.isin(dates.date)
        terros_median = terros_median[mask]
        terros_sigma = terros_sigma[mask]

        gpr_med_np = np.asarray(gpr_median, dtype=float)[: len(terros_median)]
        gpr_sig_np = np.asarray(gpr_sig, dtype=float)[: len(terros_median)]
        terros_vals = terros_median.values

        # Metrics
        beta, alpha = np.polyfit(gpr_med_np, terros_vals, 1)
        preds = alpha + beta * gpr_med_np
        r2 = r2_score(terros_vals, preds)
        rmse = mean_squared_error(terros_vals, preds, squared=False)
        mae = mean_absolute_error(terros_vals, preds)
        print(f"y = {beta:.4f}x + {alpha:.4f} | R² = {r2:.3f} | RMSE = {rmse:.3f} | MAE = {mae:.3f}")

        if plot:
            # Scatter + regression
            plt.figure(figsize=(10, 5))
            plt.scatter(gpr_med_np, terros_vals, label="Data points")
            plt.plot(gpr_med_np, preds, label=f"Fit: y = {beta:.2f}x + {alpha:.2f}")
            plt.xlabel("GPR Median VWC")
            plt.ylabel("Teros Median VWC")
            plt.title(f"Correlation: GPR vs Teros • Field {self.field_letter}")
            plt.grid(True)
            plt.legend()
            plt.text(
                0.05,
                0.95,
                f"R² = {r2:.2f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}",
                transform=plt.gca().transAxes,
                verticalalignment="top",
            )
            plt.tight_layout()
            plt.show()

            # Time series + variance bars
            plt.figure(figsize=(10, 5))
            (ln_gpr,) = plt.plot(dates[: len(gpr_med_np)], gpr_med_np, "s-", label="GPR Median VWC")
            plt.errorbar(
                dates[: len(gpr_med_np)],
                gpr_med_np,
                yerr=gpr_sig_np,
                fmt="none",
                elinewidth=1,
                capsize=2,
                alpha=0.7,
                ecolor=ln_gpr.get_color(),
            )

            (ln_ter,) = plt.plot(dates[: len(terros_vals)], terros_vals, "o-", label="Teros Median VWC")
            plt.errorbar(
                dates[: len(terros_vals)],
                terros_vals,
                yerr=terros_sigma.values,
                fmt="none",
                elinewidth=1,
                capsize=2,
                alpha=0.7,
                ecolor=ln_ter.get_color(),
            )
            plt.xlabel("Date")
            plt.ylabel("VWC")
            plt.title(f"Evolution: GPR vs Teros • Field {self.field_letter}")
            plt.legend()
            plt.grid(True)
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.show()

    def extract_tdr_dates(self):
        """Dates extraction from TDR file names (assumes YYMMDD near end)."""
        if "TDR_PATHS" not in globals():
            raise RuntimeError("TDR_PATHS is not defined.")
        dates = []
        for p in TDR_PATHS:
            base = os.path.splitext(os.path.basename(p))[0]
            m = re.search(r"(\d{6})", base)  # YYMMDD
            if m:
                y, mo, d = m.group(1)[:2], m.group(1)[2:4], m.group(1)[4:]
                try:
                    dt = datetime.strptime(f"20{y}-{mo}-{d}", "%Y-%m-%d").date()
                    dates.append(dt.strftime("%d/%m/%Y"))
                    continue
                except Exception:
                    pass
            # fallback legacy slicing
            try:
                d = base[-2:]
                mo = base[-5:-3]
                y = base[-8:-6]
                dt = datetime.strptime(f"20{y}-{mo}-{d}", "%Y-%m-%d").date()
                dates.append(dt.strftime("%d/%m/%Y"))
            except Exception:
                dates.append("unknown date")
        return dates

    def correlate_gpr_tdr(self, verification_radius=10, plot=True):
        """
        Correlate kriged VWC (sampled around each TDR point within 'verification_radius')
        vs. TDR medians, *matching campaigns by date*. Adds variance bars to the evolution plot.
        """
        if "TDR_PATHS" not in globals() or not TDR_PATHS:
            raise RuntimeError("TDR_PATHS is not defined or empty.")

        tdr_tables = [pd.read_excel(p) for p in TDR_PATHS]
        tdr_dates = pd.to_datetime(self.extract_tdr_dates(), format="%d/%m/%Y", errors="coerce")

        gpr_tables = self.import_data()
        gpr_dates = pd.to_datetime(self.extract_dates(), format="%d/%m/%Y", errors="coerce")

        krig_means, tdr_means, paired_dates = [], [], []
        krig_sigs, tdr_sigs = [], []

        x_min, x_max = 0.0, 250.0
        y_min, y_max = 0.0, 250.0
        step = 1.0
        grid_x = np.arange(x_min, x_max, step)
        grid_y = np.arange(y_min, y_max, step)
        gx, gy = np.meshgrid(grid_x, grid_y)

        poly, _, _ = self._field_polygon_utm()
        pth = mpath.Path(poly)
        inside_mask = pth.contains_points(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        mask = ~inside_mask

        r2_pix = int(np.ceil(verification_radius / step))
        r2_sq = float(verification_radius) ** 2

        for df, gdate in zip(gpr_tables, gpr_dates):
            if df.empty or pd.isna(gdate):
                continue

            match_idx = np.where(tdr_dates == gdate)[0]
            if match_idx.size == 0:
                continue
            tdr_idx = int(match_idx[0])
            tdr_raw = tdr_tables[tdr_idx]

            try:
                tdr_df = self._standardize_tdr_frame(tdr_raw)
            except Exception:
                continue
            if tdr_df.empty:
                continue

            utm_x, utm_y = self.convert_to_utm(df["x"].values, df["y"].values)
            ok = OrdinaryKriging(
                utm_x, utm_y, df["vwc"].values, variogram_model="exponential", verbose=False, enable_plotting=False
            )
            z, _ = ok.execute("masked", grid_x, grid_y, mask=mask)
            z = np.ma.filled(z, np.nan)

            tx, ty = self.convert_to_utm(tdr_df["Lon"].values, tdr_df["Lat"].values)
            v_est = []
            for px, py in zip(tx, ty):
                ix = int(np.round((px - x_min) / step))
                iy = int(np.round((py - y_min) / step))
                if ix < 0 or iy < 0 or ix >= z.shape[1] or iy >= z.shape[0]:
                    continue
                x0 = max(ix - r2_pix, 0)
                x1 = min(ix + r2_pix, z.shape[1] - 1)
                y0 = max(iy - r2_pix, 0)
                y1 = min(iy + r2_pix, z.shape[0] - 1)
                Zsub = z[y0 : y1 + 1, x0 : x1 + 1]
                if not np.isfinite(Zsub).any():
                    continue
                GX = gx[y0 : y1 + 1, x0 : x1 + 1]
                GY = gy[y0 : y1 + 1, x0 : x1 + 1]
                m = (GX - px) ** 2 + (GY - py) ** 2 <= r2_sq
                vals = Zsub[m]
                if np.isfinite(vals).any():
                    v_est.append(np.nanmedian(vals))

            if not v_est:
                continue

            krig_means.append(float(np.nanmean(v_est)))
            tdr_means.append(float(np.nanmedian(tdr_df["VWC"].values)))
            paired_dates.append(gdate)
            # --- new: robust σ for the time series error bars ---
            krig_sigs.append(self._mad_std(v_est))
            tdr_sigs.append(self._mad_std(tdr_df["VWC"].values))

        if len(krig_means) < 3:
            print("Not enough paired (GPR↔TDR) campaign dates for correlation.")
            return

        krig_means = np.asarray(krig_means, dtype=float)
        tdr_means = np.asarray(tdr_means, dtype=float)
        krig_sigs = np.asarray(krig_sigs, dtype=float)
        tdr_sigs = np.asarray(tdr_sigs, dtype=float)

        beta, alpha = np.polyfit(krig_means, tdr_means, 1)
        preds = alpha + beta * krig_means
        r2 = r2_score(tdr_means, preds)
        rmse = mean_squared_error(tdr_means, preds, squared=False)
        mae = mean_absolute_error(tdr_means, preds)
        print(
            f"Kriged vs TDR (date-matched): y = {beta:.4f}x + {alpha:.4f} | R²={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}"
        )

        if plot:
            # Scatter
            plt.figure(figsize=(10, 5))
            plt.scatter(krig_means, tdr_means, label="Campaign medians")
            xline = np.linspace(np.nanmin(krig_means), np.nanmax(krig_means), 100)
            plt.plot(xline, alpha + beta * xline, label=f"Fit: y={beta:.2f}x+{alpha:.2f}")
            plt.xlabel("Kriged GPR VWC (median of TDR-radius samples)")
            plt.ylabel("TDR VWC (campaign median)")
            plt.title(f"Correlation: Kriged GPR vs TDR • Field {self.field_letter}")
            plt.grid(True)
            plt.legend()
            plt.text(
                0.05, 0.95, f"R²={r2:.2f}\nRMSE={rmse:.3f}\nMAE={mae:.3f}", transform=plt.gca().transAxes, va="top"
            )
            plt.tight_layout()
            plt.show()

            # Time series + variance bars
            plt.figure(figsize=(10, 5))
            pdts = pd.to_datetime(paired_dates)
            (ln_k,) = plt.plot(pdts, krig_means, "s-", label="Kriged GPR (median around TDR points)")
            plt.errorbar(
                pdts,
                krig_means,
                yerr=krig_sigs,
                fmt="none",
                elinewidth=1,
                capsize=2,
                alpha=0.7,
                ecolor=ln_k.get_color(),
            )

            (ln_t,) = plt.plot(pdts, tdr_means, "o-", label="TDR (campaign median)")
            plt.errorbar(
                pdts, tdr_means, yerr=tdr_sigs, fmt="none", elinewidth=1, capsize=2, alpha=0.7, ecolor=ln_t.get_color()
            )
            plt.xlabel("Date")
            plt.ylabel("VWC [m³/m³]")
            plt.title(f"Time Series: Kriged GPR vs TDR • Field {self.field_letter}")
            plt.grid(True)
            plt.legend()
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.show()

    def calibrate_tvdi_to_teros(
        self,
        msa,  # MultispecAnalysis(field_letter=..., ...)
        date_tolerance_days=0,  # 0 = exact same day; >0 = nearest within N days
        plot=False,
        daily_agg="median",  # "median" or "mean" over the day (per probe)
        sensor_agg="mean",  # "mean" or "median" across probes for that field
        teros_csv="Data/Teros Piezo/teros_piezo.csv",
    ):
        """
        Fit VWC ≈ a − b·TVDI using Teros continuous VWC (daily aggregated) and TVDI means.
        Returns dict with a, b, diagnostics, and the matched series.
        Notes:
        - Uses only Teros probes for the same field (suffix A/B) if available,
            otherwise falls back to all 'T_*' probes.
        - Daily aggregation (median/mean) is applied per probe, then sensor_agg
            (mean/median) across probes for that day.
        """
        if msa.field_letter != self.field_letter:
            raise ValueError("msa.field_letter must match GPR field_letter.")

        # ---- TVDI time series (mean over masked field, already computed by msa) ----
        ts_tvdi = msa.plot_indices_evolution(plot=False)["tvdi"]
        tvdi_df = (
            pd.DataFrame(
                {
                    "date": pd.to_datetime(ts_tvdi["dates"]).tz_localize(None).normalize(),
                    "TVDI": ts_tvdi["mean"].astype(float),
                }
            )
            .dropna()
            .sort_values("date")
        )

        # ---- Teros continuous → daily VWC for the same field ----
        ter = pd.read_csv(teros_csv, parse_dates=["Dates (hours)"])
        ter = ter.rename(columns={"Dates (hours)": "datetime"})

        # Pick Teros columns for this field if they exist, else all T_* columns
        fld = self.field_letter.upper()
        fld_cols = [c for c in ter.columns if c.startswith("T_") and c.endswith(fld)]
        if not fld_cols:
            fld_cols = [c for c in ter.columns if c.startswith("T_")]
        if not fld_cols:
            raise RuntimeError("No Teros VWC columns found (expected columns starting with 'T_').")

        ter = ter[["datetime"] + fld_cols].copy().set_index("datetime").sort_index()

        # Daily aggregation per probe
        if daily_agg.lower() == "mean":
            daily = ter.resample("D").mean()
        else:
            daily = ter.resample("D").median()

        # Aggregate across probes for that day
        if sensor_agg.lower() == "median":
            vwc_daily = daily.median(axis=1)
        else:
            vwc_daily = daily.mean(axis=1)

        teros_df = vwc_daily.reset_index().rename(columns={"datetime": "date", 0: "VWC"})
        teros_df["date"] = teros_df["date"].dt.tz_localize(None).dt.normalize()
        teros_df = teros_df[["date", "VWC"]].dropna().sort_values("date")

        # ---- Pair by date (exact or nearest within tolerance) ----
        if date_tolerance_days == 0:
            pairs = pd.merge(tvdi_df, teros_df, on="date", how="inner")
        else:
            pairs = pd.merge_asof(
                tvdi_df,
                teros_df,
                on="date",
                direction="nearest",
                tolerance=pd.Timedelta(days=date_tolerance_days),
            ).dropna(subset=["VWC"])

        X = pairs["TVDI"].to_numpy(dtype=float)  # predictor
        Y = pairs["VWC"].to_numpy(dtype=float)  # response

        if X.size < 3:
            raise RuntimeError(f"Not enough paired dates to fit a linear model (got {X.size}).")

        # ---- Fit: VWC = α + β·TVDI (report as VWC ≈ a − b·TVDI) ----
        beta, alpha = np.polyfit(X, Y, 1)  # slope, intercept
        a = float(alpha)
        b = float(-beta)  # so VWC ≈ a − b·TVDI
        r2 = float(r2_score(Y, alpha + beta * X))

        if plot:
            plt.figure(figsize=(6, 4), dpi=300)
            plt.scatter(X, Y, s=25, label=f"Pairs (n={X.size})")
            xline = np.linspace(np.nanmin(X), np.nanmax(X), 100)
            plt.plot(xline, alpha + beta * xline, label=f"Fit: VWC={a:.3f}−{b:.3f}·TVDI\nR²={r2:.2f}")
            plt.xlabel("TVDI [-]")
            plt.ylabel("VWC [m³/m³]")
            plt.title(f"TVDI vs Teros VWC • Field {self.field_letter}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return {
            "a": a,
            "b": b,
            "intercept_alpha": float(alpha),
            "slope_beta": float(beta),
            "r2": r2,
            "n_pairs": int(X.size),
            "dates": pairs["date"].to_list(),
            "TVDI": X,
            "VWC": Y,
            "used_teros_columns": fld_cols,
            "daily_agg": daily_agg,
            "sensor_agg": sensor_agg,
        }

    def calibrate_tvdi_to_vwc(
        self,
        msa,  # MultispecAnalysis instance for the same field
        use="median",  # "median" | "mean" across GPR points per date
        date_tolerance_days=0,  # 0 = exact date, >0 = nearest within N days
        plot=True,
    ):
        """
        Fit VWC ≈ a − b·TVDI using GPR campaign VWC (per-date mean/median) and TVDI medians.
        Returns dict with a, b, diagnostics, and the matched series.
        """
        if msa.field_letter != self.field_letter:
            raise ValueError("msa.field_letter must match GPR field_letter.")

        # ---- TVDI time series from multispec (median over field, one value per date) ----
        ts_tvdi = msa.plot_indices_evolution(plot=False)["tvdi"]
        tvdi_df = (
            pd.DataFrame(
                {
                    "date": pd.to_datetime(ts_tvdi["dates"]).tz_localize(None).normalize(),
                    "TVDI": ts_tvdi["mean"].astype(float),
                }
            )
            .dropna()
            .sort_values("date")
            .drop_duplicates(subset=["date"])
        )

        # ---- GPR VWC per campaign date (mean/median across points) ----
        tables = self.import_data(show=False)
        gpr_dates = pd.to_datetime(self.extract_dates(), format="%d/%m/%Y", errors="coerce")
        gpr_dates = gpr_dates.tz_localize(None).normalize()

        stat_fn = np.nanmean if str(use).lower() == "mean" else np.nanmedian
        vwc_vals = []
        for df in tables:
            v = pd.to_numeric(df["vwc"], errors="coerce").to_numpy(dtype=float)
            vwc_vals.append(stat_fn(v) if np.isfinite(v).any() else np.nan)

        gpr_df = (
            pd.DataFrame({"date": gpr_dates, "VWC": vwc_vals})
            .dropna()
            .sort_values("date")
            .drop_duplicates(subset=["date"])
        )

        # ---- Pair by date (exact or nearest within tolerance) ----
        if date_tolerance_days == 0:
            pairs = pd.merge(tvdi_df, gpr_df, on="date", how="inner")
        else:
            pairs = pd.merge_asof(
                tvdi_df.sort_values("date"),
                gpr_df.sort_values("date"),
                on="date",
                direction="nearest",
                tolerance=pd.Timedelta(days=int(date_tolerance_days)),
            ).dropna(subset=["VWC"])

        X = pairs["TVDI"].to_numpy(dtype=float)
        Y = pairs["VWC"].to_numpy(dtype=float)
        if X.size < 3:
            raise RuntimeError(f"Not enough paired dates to fit a linear model (got {X.size}).")

        # ---- Fit: VWC = α + β·TVDI  (report as VWC ≈ a − b·TVDI) ----
        beta, alpha = np.polyfit(X, Y, 1)
        a = float(alpha)
        b = float(-beta)  # so VWC ≈ a − b·TVDI
        r2 = float(r2_score(Y, alpha + beta * X))

        if plot:
            plt.figure(figsize=(6, 4), dpi=300)
            plt.scatter(X, Y, s=25, label=f"Pairs (n={X.size})")
            xline = np.linspace(np.nanmin(X), np.nanmax(X), 100)
            plt.plot(xline, alpha + beta * xline, label=f"Fit: VWC={a:.3f}−{b:.3f}·TVDI\nR²={r2:.2f}")
            plt.xlabel("TVDI [-]")
            plt.ylabel("VWC [m³/m³]")
            plt.title(f"TVDI vs GPR VWC • Field {self.field_letter}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return {
            "a": a,
            "b": b,
            "intercept_alpha": float(alpha),
            "slope_beta": float(beta),
            "r2": r2,
            "n_pairs": int(X.size),
            "dates": pairs["date"].to_list(),
            "TVDI": X,
            "VWC": Y,
            "use": str(use).lower(),
            "date_tolerance_days": int(date_tolerance_days),
        }
        # ========= K-MEANS ZONING (add inside GprAnalysis) =========

    def _kmeans_prepare_global_scalers(self):
        """Collect global mins/maxs to min-max normalise x,y,VWC across all campaigns."""
        tables = self.import_data(show=False)
        xs = []
        ys = []
        vs = []
        for df in tables:
            xutm, yutm = self.convert_to_utm(df["x"].values, df["y"].values)
            v = pd.to_numeric(df["vwc"], errors="coerce").to_numpy(float)
            m = np.isfinite(xutm) & np.isfinite(yutm) & np.isfinite(v)
            if m.any():
                xs.append(xutm[m])
                ys.append(yutm[m])
                vs.append(v[m])
        if not xs:
            raise RuntimeError("No finite GPR samples available to build k-means scalers.")
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        vs = np.concatenate(vs)
        eps = 1e-9
        return {
            "x_min": float(np.nanmin(xs)),
            "x_max": float(np.nanmax(xs) + eps),
            "y_min": float(np.nanmin(ys)),
            "y_max": float(np.nanmax(ys) + eps),
            "v_min": float(np.nanmin(vs)),
            "v_max": float(np.nanmax(vs) + eps),
        }

    def _kmeans_build_features(self, xutm, yutm, vwc, scales, coordinate_weight=1.0):
        """Min-max normalise and weight (x,y) then stack with normalised VWC."""
        x = (xutm - scales["x_min"]) / (scales["x_max"] - scales["x_min"])
        y = (yutm - scales["y_min"]) / (scales["y_max"] - scales["y_min"])
        v = (vwc - scales["v_min"]) / (scales["v_max"] - scales["v_min"])
        F = np.column_stack([x * float(coordinate_weight), y * float(coordinate_weight), v])
        m = np.isfinite(F).all(axis=1)
        return F[m], m

    def fit_kmeans_zones(
        self,
        n_clusters=2,
        coordinate_weight=1.0,
        random_state=0,
        verbose=True,
    ):
        """
        Fit ONE k-means model to ALL campaigns using features [x,y,VWC] (min-max normalised),
        with optional 'coordinate_weight' to encourage spatial coherence.
        Stores the model & scaling so you can call 'predict_kmeans_labels_for_campaign' and
        'plot_kmeans_zones_evolution' consistently.
        """
        try:
            from sklearn.cluster import KMeans
        except Exception as e:
            raise ImportError("scikit-learn is required for k-means zoning.") from e

        tables = self.import_data(show=False)
        scales = self._kmeans_prepare_global_scalers()

        # build a single training matrix across all campaigns
        X_all = []
        for df in tables:
            xutm, yutm = self.convert_to_utm(df["x"].values, df["y"].values)
            v = pd.to_numeric(df["vwc"], errors="coerce").to_numpy(float)
            F, m = self._kmeans_build_features(xutm, yutm, v, scales, coordinate_weight=coordinate_weight)
            if F.size:
                X_all.append(F)
        if not X_all:
            raise RuntimeError("Could not assemble training features for k-means.")
        X_all = np.vstack(X_all)

        # fit
        km = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=int(random_state))
        labels_all = km.fit_predict(X_all)

        # order clusters by overall (normalised) V component to have Dry→Wet = 0..k-1
        # (centroids are in the same normalised feature space; V is column 2)
        order = np.argsort(km.cluster_centers_[:, 2])
        remap = {int(old): int(new) for new, old in enumerate(order)}

        # stash
        self._kmeans_model = km
        self._kmeans_scales = scales
        self._kmeans_coord_w = float(coordinate_weight)
        self._kmeans_label_remap = remap

        if verbose:
            cc = km.cluster_centers_[order]
            print("[KMeans] Fitted zones (ordered dry→wet) on all campaigns.")
            for i, c in enumerate(cc):
                print(f"  Zone {i}: centroid ~ [x:{c[0]:.2f}, y:{c[1]:.2f}, v_norm:{c[2]:.2f}]")

        return {"model": km, "scales": scales, "remap": remap}

    def predict_kmeans_labels_for_campaign(self, sample_number=None):
        """
        Assign k-means zones to the points of a given campaign using the already-fitted model.
        Returns dict with fields: labels (int per point), order-stable (dry→wet), and mask of used rows.
        """
        if sample_number is None:
            sample_number = self.sample_number
        if not hasattr(self, "_kmeans_model"):
            raise RuntimeError("Call fit_kmeans_zones(...) before predicting labels.")

        df = self.import_data()[int(sample_number)]
        xutm, yutm = self.convert_to_utm(df["x"].values, df["y"].values)
        v = pd.to_numeric(df["vwc"], errors="coerce").to_numpy(float)

        F, m = self._kmeans_build_features(xutm, yutm, v, self._kmeans_scales, coordinate_weight=self._kmeans_coord_w)
        if F.size == 0:
            raise RuntimeError("No finite samples to assign k-means labels for this campaign.")

        labs = self._kmeans_model.predict(F)
        # remap to dry→wet order
        labs = np.vectorize(self._kmeans_label_remap.get)(labs)

        # build full-length label array with -1 for invalid rows
        full_labels = np.full(df.shape[0], -1, dtype=int)
        full_labels[m] = labs
        return {"labels": full_labels, "valid_mask": m, "n_clusters": self._kmeans_model.n_clusters}

    def plot_kmeans_zone_map(
        self,
        n_clusters=3,
        coordinate_weight=1.0,  # ignored (model already fitted)
        random_state=0,
        grid_step=1.0,
        cmap_name=None,
        show_points=True,
        annotate=True,
    ):
        # use fitted model
        if not hasattr(self, "_kmeans_model"):
            raise RuntimeError("Call fit_kmeans_zones(...) first.")

        # optional text outline
        try:
            from matplotlib import patheffects as pe
        except Exception:
            pe = None

        import matplotlib.path as mpath
        from matplotlib.colors import BoundaryNorm
        from matplotlib.colors import ListedColormap
        from scipy.interpolate import NearestNDInterpolator

        df = self.import_data()[self.sample_number].copy()
        v = pd.to_numeric(df["vwc"], errors="coerce").to_numpy(float)
        lon = pd.to_numeric(df["x"], errors="coerce")
        lat = pd.to_numeric(df["y"], errors="coerce")
        x, y = self.convert_to_utm(lon.values, lat.values)

        F, m = self._kmeans_build_features(x, y, v, self._kmeans_scales, coordinate_weight=self._kmeans_coord_w)
        if F.size == 0:
            raise RuntimeError("No finite samples to assign k-means labels for this campaign.")

        # predict with stored model + dry→wet remap
        labs_pts = np.full(df.shape[0], -1, dtype=int)
        labs_pts[m] = np.vectorize(self._kmeans_label_remap.get)(self._kmeans_model.predict(F))
        labs_core = labs_pts[m]  # <— labels aligned with x[m], y[m]
        K = self._kmeans_model.n_clusters

        # rasterize to grid (nearest)
        x_min, x_max, y_min, y_max, step = 0.0, 250.0, 0.0, 250.0, float(grid_step)
        gx = np.arange(x_min, x_max + step, step)
        gy = np.arange(y_min, y_max + step, step)
        GX, GY = np.meshgrid(gx, gy)

        nn = NearestNDInterpolator(np.column_stack([x[m], y[m]]), labs_core)
        Z = nn(GX, GY).astype(float)

        poly, xlim_max, ylim_max = self._field_polygon_utm()
        outside = ~mpath.Path(poly).contains_points(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        Z[outside] = np.nan

        # categorical colors
        if cmap_name is None:
            cmap_name = "tab10" if K <= 10 else "tab20"
        base = plt.get_cmap(cmap_name)
        colors = [base(i) for i in range(K)]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, K + 0.5, 1.0), cmap.N)

        # plot
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=300)
        im = ax.imshow(
            Z,
            extent=(gx.min(), gx.max(), gy.min(), gy.max()),
            origin="lower",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )

        poly_closed = np.vstack([poly, poly[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], "k-", linewidth=4.0, alpha=0.9)

        try:
            levels = np.arange(-0.5, K - 0.5 + 1e-9, 1.0)
            ax.contour(GX, GY, Z, levels=levels, colors="white", linewidths=1.0, alpha=0.9)
            ax.contour(GX, GY, Z, levels=levels, colors="k", linewidths=0.4, alpha=0.8)
        except Exception:
            pass

        if show_points:
            ax.scatter(x[m], y[m], s=12, c=labs_core, cmap=cmap, norm=norm, edgecolors="k", linewidths=0.3, alpha=0.85)

        if annotate:
            for k in range(K):
                mk = labs_core == k
                if not np.any(mk):
                    continue
                cx, cy = np.nanmedian(x[m][mk]), np.nanmedian(y[m][mk])
                txt = ax.text(cx, cy, f"Z{k+1}", ha="center", va="center", fontsize=9, weight="bold", color="white")
                if pe is not None:
                    txt.set_path_effects([pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()])

        ax.set_xlim(-5, xlim_max)
        ax.set_ylim(-5, ylim_max)
        self._apply_map_grid(ax, major=100, minor=50, labelsize=6)
        self._add_north_arrow(ax, loc="NE", size=0.09, inset=(0.03, 0.01), font_size=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

        import matplotlib.patches as mpatches

        handles = [mpatches.Patch(color=colors[i], label=f"Zone {i+1}") for i in range(K)]
        ax.legend(handles=handles, loc="lower left", frameon=True)

        date_str = self.extract_dates()[self.sample_number]
        ax.set_title(f"K-means VWC Zones • Field {self.field_letter} • {date_str}", pad=8)
        plt.tight_layout()
        plt.show()

    def plot_kmeans_zones_evolution(
        self,
        n_clusters=2,
        coordinate_weight=1.0,
        random_state=0,
        start=None,
        end=None,
        cmap_name="tab10",
    ):
        """
        Fit zones on *all campaigns* once, then plot the per-zone median VWC evolution with
        robust variance bars (MAD→σ) across campaigns. Zones are dry→wet consistent.
        """
        self.fit_kmeans_zones(
            n_clusters=n_clusters, coordinate_weight=coordinate_weight, random_state=random_state, verbose=False
        )

        tables = self.import_data(show=False)
        dates = pd.to_datetime(self.extract_dates(), format="%d/%m/%Y", errors="coerce")

        # optional clip
        if start is not None:
            start_ts = pd.to_datetime(start)
        else:
            start_ts = dates.min()
        if end is not None:
            end_ts = pd.to_datetime(end)
        else:
            end_ts = dates.max()
        sel_time = (dates >= start_ts) & (dates <= end_ts)

        K = self._kmeans_model.n_clusters
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i % 10) for i in range(K)]

        # collect medians and robust spread per zone per date
        med = {k: [] for k in range(K)}
        sig = {k: [] for k in range(K)}
        dts = []

        for i, use in enumerate(sel_time):
            if not use:
                continue
            asg = self.predict_kmeans_labels_for_campaign(sample_number=i)
            labs = asg["labels"]
            df = tables[i]
            v = pd.to_numeric(df["vwc"], errors="coerce").to_numpy(float)
            dts.append(dates[i])  # DatetimeIndex supports positional indexing via []
            for k in range(K):
                m = labs == k
                if m.any():
                    vv = v[m]
                    med[k].append(float(np.nanmedian(vv)))
                    sig[k].append(float(self._mad_std(vv)))
                else:
                    med[k].append(np.nan)
                    sig[k].append(np.nan)

        # plot
        fig, ax = plt.subplots(figsize=(10, 5))
        for k in range(K):
            y = np.array(med[k], dtype=float)
            e = np.array(sig[k], dtype=float)
            label = f"Zone {k+1}"  # 1-based, matches the map (Z1, Z2, …)
            (ln,) = ax.plot(dts, y, "o-", label=label, color=colors[k])
            ax.errorbar(dts, y, yerr=e, fmt="none", elinewidth=1, capsize=2, alpha=0.8, ecolor=ln.get_color())

        ax.set_xlabel("Date")
        ax.set_ylabel("VWC [/]")
        title = f"Zonal evolution (k-means, k={K}, coord_w={self._kmeans_coord_w:g}) • Field {self.field_letter}"
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return {
            "dates": np.array(dts),
            "median": {k: np.array(med[k]) for k in range(K)},
            "sigma": {k: np.array(sig[k]) for k in range(K)},
            "n_clusters": K,
            "coordinate_weight": float(self._kmeans_coord_w),
        }

    def plot_kmeans_model_selection(self, max_k=8, coordinate_weight=1.0, random_state=0):
        """Elbow (inertia) + silhouette (separation) across K=2..max_k."""

        tables = self.import_data(show=False)
        scales = self._kmeans_prepare_global_scalers()

        X_all = []
        for df in tables:
            xutm, yutm = self.convert_to_utm(df["x"].values, df["y"].values)
            v = pd.to_numeric(df["vwc"], errors="coerce").to_numpy(float)
            F, _ = self._kmeans_build_features(xutm, yutm, v, scales, coordinate_weight=coordinate_weight)
            if F.size:
                X_all.append(F)
        if not X_all:
            raise RuntimeError("No features to evaluate K.")
        X_all = np.vstack(X_all)

        Ks = list(range(2, int(max_k) + 1))
        inertias, sils = [], []
        for k in Ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=int(random_state))
            labs = km.fit_predict(X_all)
            inertias.append(float(km.inertia_))
            sils.append(float(silhouette_score(X_all, labs)))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(Ks, inertias, "o-", label="Inertia (elbow)")
        ax.set_xlabel("K")
        ax.set_ylabel("Inertia")
        ax2 = ax.twinx()
        ax2.plot(Ks, sils, "s--", label="Silhouette", alpha=0.8)
        ax2.set_ylabel("Mean silhouette")
        ax.set_title(f"K selection (coord_w={coordinate_weight:g}) • Field {self.field_letter}")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        return {"K": np.array(Ks), "inertia": np.array(inertias), "silhouette": np.array(sils)}

    def kmeans_vwc_anova(self, sample_number=None):
        """Report global variance, mean within-zone variance, and one-way ANOVA F/p."""

        if sample_number is None:
            sample_number = self.sample_number
        asg = self.predict_kmeans_labels_for_campaign(sample_number)
        labs, m, K = asg["labels"], asg["valid_mask"], asg["n_clusters"]
        df = self.import_data()[sample_number]
        v = pd.to_numeric(df["vwc"], errors="coerce").to_numpy(float)
        v = v[m]
        labs = labs[m]

        groups = [v[labs == k] for k in range(K) if np.isfinite(v[labs == k]).any()]
        if len(groups) < 2:
            raise RuntimeError("Need ≥2 non-empty zones for ANOVA.")

        global_var = float(np.nanvar(v))
        within_vars = [float(np.nanvar(g)) for g in groups]
        F = f_oneway(*groups)
        print(f"[ANOVA] global var={global_var:.4f} | mean within-zone var={np.nanmean(within_vars):.4f}")
        print(f"[ANOVA] F={float(F.statistic):.3f}, p={float(F.pvalue):.3g}")
        return {"global_var": global_var, "within_vars": within_vars, "F": float(F.statistic), "p": float(F.pvalue)}


# =========================
# Variogram
# =========================
class Variogram:
    """Experimental variogram creation and model fitting for GPR VWC."""

    def __init__(self, resolution=0.00002, field_letter="A", sample_number=0):
        self.resolution = float(resolution)
        self.field_letter = str(field_letter)
        self.sample_number = int(sample_number)

        if self.field_letter == "A":
            self.field_paths = GPR_A_PATHS
        elif self.field_letter == "B":
            self.field_paths = GPR_B_PATHS
        else:
            raise ValueError("field_letter must be either 'A' or 'B'")

        self.gpr_analysis = GprAnalysis(self.field_letter, self.sample_number)

    def determ_experimental_vario(self, maxlag=30, n_lags=50, solo_plot=True):
        """
        Build the isotropic experimental variogram on normal-score VWC.
        """
        tables = self.gpr_analysis.import_data()
        if not tables:
            raise RuntimeError("No GPR files found for the selected field.")
        if self.sample_number < 0 or self.sample_number >= len(tables):
            raise IndexError(f"sample_number={self.sample_number} is out of range (0..{len(tables)-1}).")

        df = tables[self.sample_number]

        # lon/lat -> UTM metres (normalized to 0 origin)
        utm_x, utm_y = self.gpr_analysis.convert_to_utm(df["x"].values, df["y"].values)

        # clean & normal-score
        grid = pd.DataFrame({"X": utm_x, "Y": utm_y, "Z": pd.to_numeric(df["vwc"].values, errors="coerce")}).dropna()

        if grid.shape[0] < 5:
            raise RuntimeError("Too few valid GPR points to compute a variogram.")

        data = grid["Z"].values.reshape(-1, 1)
        n_q = int(min(500, max(10, data.shape[0])))
        nst = QuantileTransformer(n_quantiles=n_q, output_distribution="normal", random_state=0, copy=True)
        grid["Z_nscore"] = nst.fit_transform(data).ravel()

        coords = grid[["X", "Y"]].to_numpy(dtype=float)
        values = grid["Z_nscore"].to_numpy(dtype=float)

        # Experimental variogram (isotropic)
        v_iso = skg.Variogram(
            coordinates=coords,  # <-- correct keyword
            values=values,
            bin_func="even",
            n_lags=int(n_lags),
            maxlag=float(maxlag),
            normalize=False,
        )

        xdata = v_iso.bins
        ydata = v_iso.experimental

        if solo_plot:
            date_str = self.gpr_analysis.extract_dates()[self.sample_number]
            plt.figure(figsize=(8, 6))
            plt.scatter(xdata, ydata, s=14)
            plt.title(f"Isotropic experimental variogram • Field {self.field_letter} • {date_str}")
            plt.xlabel("Lag (m)")
            plt.ylabel("Semivariance")
            plt.tight_layout()
            plt.show()

        return v_iso, xdata, ydata

    def fit_models(
        self,
        maxlag=30,
        n_lags=50,
        solo_plot=False,
        multi_plot=True,
        multi_zoom_plot=True,
        sample_number=None,
    ):
        """
        Fit Exponential, Gaussian, and Spherical models to the experimental variogram
        and report R² for each fit (plain and, if bin counts are available, weighted).
        """
        if sample_number is not None:
            self.sample_number = int(sample_number)

        # --- experimental variogram once ---
        v_iso, xdata, ydata = self.determ_experimental_vario(maxlag=maxlag, n_lags=n_lags, solo_plot=solo_plot)

        # try to get per-bin sample counts to support weighted R²
        w = getattr(v_iso, "bin_count", None)
        if isinstance(w, (list, np.ndarray)) and len(w) == len(xdata):
            w = np.asarray(w, dtype=float)
        else:
            w = None  # not available or mismatched

        # --- re-create inputs (robust to skgstat versions) ---
        tables = self.gpr_analysis.import_data()
        df = tables[self.sample_number]
        utm_x, utm_y = self.gpr_analysis.convert_to_utm(df["x"].values, df["y"].values)
        grid = pd.DataFrame({"X": utm_x, "Y": utm_y, "Z": pd.to_numeric(df["vwc"].values, errors="coerce")}).dropna()

        data = grid["Z"].values.reshape(-1, 1)
        n_q = int(min(500, max(10, data.shape[0])))
        nst = QuantileTransformer(n_quantiles=n_q, output_distribution="normal", random_state=0, copy=True)
        grid["Z_nscore"] = nst.fit_transform(data).ravel()

        coords = grid[["X", "Y"]].to_numpy(dtype=float)
        values = grid["Z_nscore"].to_numpy(dtype=float)

        # --- independent model fits ---
        v_exp = skg.Variogram(
            coords, values, bin_func="even", n_lags=n_lags, maxlag=maxlag, normalize=False, model="exponential"
        )
        v_gauss = skg.Variogram(
            coords, values, bin_func="even", n_lags=n_lags, maxlag=maxlag, normalize=False, model="gaussian"
        )
        v_sph = skg.Variogram(
            coords, values, bin_func="even", n_lags=n_lags, maxlag=maxlag, normalize=False, model="spherical"
        )

        def _eval_model(vobj, func, hvec):
            s, r, n = vobj.parameters  # (sill, range, nugget)
            return np.array([func(h, s, r, n) for h in hvec], dtype=float)

        y_fit_exp = _eval_model(v_exp, models.exponential, xdata)
        y_fit_gauss = _eval_model(v_gauss, models.gaussian, xdata)
        y_fit_sph = _eval_model(v_sph, models.spherical, xdata)

        # --- R² helpers ---
        def _r2(y_true, y_pred, weights=None):
            y_true = np.asarray(y_true, float)
            y_pred = np.asarray(y_pred, float)
            m = np.isfinite(y_true) & np.isfinite(y_pred)
            if m.sum() < 3:
                return np.nan
            yt, yp = y_true[m], y_pred[m]
            if weights is None:
                ss_res = np.sum((yt - yp) ** 2)
                ss_tot = np.sum((yt - yt.mean()) ** 2)
            else:
                ww = np.asarray(weights, float)
                ww = ww[m]
                if not np.isfinite(ww).any() or ww.sum() <= 0:
                    return _r2(yt, yp, None)
                ww = ww / ww.sum()
                ybar = np.sum(ww * yt)
                ss_res = np.sum(ww * (yt - yp) ** 2)
                ss_tot = np.sum(ww * (yt - ybar) ** 2)
            if ss_tot <= 0:
                return np.nan
            return 1.0 - ss_res / ss_tot

        R2 = {
            "exp": {"plain": _r2(ydata, y_fit_exp), "weighted": _r2(ydata, y_fit_exp, w)},
            "gauss": {"plain": _r2(ydata, y_fit_gauss), "weighted": _r2(ydata, y_fit_gauss, w)},
            "sph": {"plain": _r2(ydata, y_fit_sph), "weighted": _r2(ydata, y_fit_sph, w)},
        }

        # titles
        date_str = self.gpr_analysis.extract_dates()[self.sample_number]
        title = f"Isotropic variogram models • Field {self.field_letter} • {date_str}"

        # --- plots ---
        def _fmt(model_key):
            # prefer weighted R² when available; fall back to plain
            r = R2[model_key]["weighted"]
            if not np.isfinite(r):
                r = R2[model_key]["plain"]
            return f"R²={r:.2f}" if np.isfinite(r) else "R²=NA"

        if multi_plot:
            plt.figure(figsize=(8, 6))
            plt.plot(xdata, ydata, "o", label="Experimental")
            plt.plot(xdata, y_fit_gauss, "--", label=f"Gaussian ({_fmt('gauss')})")
            plt.plot(xdata, y_fit_exp, "-", label=f"Exponential ({_fmt('exp')})")
            plt.plot(xdata, y_fit_sph, "-.", label=f"Spherical ({_fmt('sph')})")
            plt.xlabel("Lag [m]")
            plt.ylabel("Semivariance")
            plt.title(title)
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.show()

        if multi_zoom_plot:
            plt.figure(figsize=(8, 6))
            plt.plot(xdata, ydata, "o", label="Experimental")
            plt.plot(xdata, y_fit_gauss, "--", label=f"Gaussian ({_fmt('gauss')})")
            plt.plot(xdata, y_fit_exp, "-", label=f"Exponential ({_fmt('exp')})")
            plt.plot(xdata, y_fit_sph, "-.", label=f"Spherical ({_fmt('sph')})")
            plt.xlim(0, min(5, float(maxlag)))
            plt.xlabel("Lag [m]")
            plt.ylabel("Semivariance")
            plt.title(title + " (zoom)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.show()

        return {
            "v_exp": v_exp,
            "v_gauss": v_gauss,
            "v_sph": v_sph,
            "x": xdata,
            "y_exp_scatter": ydata,
            "y_exp": y_fit_exp,
            "y_gauss": y_fit_gauss,
            "y_sph": y_fit_sph,
            "r2": R2,  # <- R² values here
            "weights_used": (w is not None),
        }


# =========================
# TDR paths (keep your global)
# =========================
# TDR_PATHS = sorted(glob.glob(os.path.join("data", "VWC verification", "*.xlsx")))


# =========================
# Rainfall
# =========================
class Rainfall:
    PATHS = sorted(glob.glob(os.path.join("Data", "Météo", "*.xlsx")))

    def __init__(self, paths=None):
        """Container for rainfall time series (per-file sheets with 'date' and 'prcp')."""
        self.paths = list(paths) if paths is not None else list(self.PATHS)

    def import_excel(self, show=False):
        """Read all rainfall Excel files into a list of DataFrames (tolerant to column order)."""
        out = []
        for p in self.paths:
            try:
                df = pd.read_excel(p)
                out.append(df)
            except Exception as e:
                print(f"[WARN] Could not read rainfall file: {p} ({e})")
        if show:
            print(out)
        return out

    def plot_data(self, plot=True, rolling_window=10):
        """
        Plot rainfall bars (daily 'prcp') and a centered rolling mean curve,
        restricted to 16 May 2023 → 02 Feb 2024.
        """
        frames = self.import_excel()
        if not frames:
            print("[INFO] No rainfall files found.")
            return None

        # Concatenate & coerce
        all_precip, all_dates = [], []
        for rf in frames:
            if "prcp" not in rf.columns or "date" not in rf.columns:
                cols = {c.lower(): c for c in rf.columns}
                if "prcp" in cols and "date" in cols:
                    prcp_col, date_col = cols["prcp"], cols["date"]
                else:
                    print("[WARN] Skipping rainfall file with missing columns.")
                    continue
            else:
                prcp_col, date_col = "prcp", "date"

            all_precip.extend(pd.to_numeric(rf[prcp_col], errors="coerce").tolist())
            all_dates.extend(pd.to_datetime(rf[date_col], errors="coerce").tolist())

        s = pd.Series(all_precip, index=pd.to_datetime(all_dates)).dropna()
        s = s.sort_index()
        s = s.groupby(s.index.normalize()).sum()  # daily totals

        # ---- clip to requested window ----
        start = pd.Timestamp("2023-05-16")
        end = pd.Timestamp("2024-02-02")
        s = s.loc[start:end]

        if s.empty:
            print("[INFO] No rainfall data in the requested window (2023-05-16 → 2024-02-02).")
            return s

        smoothed = s.rolling(window=int(max(1, rolling_window)), center=True, min_periods=1).mean()

        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(s.index, s.values, align="center", alpha=0.5)
            ax.plot(smoothed.index, smoothed.values, linewidth=2)
            ax.set_xlabel("Date")
            ax.set_ylabel("Precipitation [mm]")
            ax.set_title("Rainfall Precipitations (16 May 2023 → 02 Feb 2024)")
            ax.xaxis.set_major_locator(plt.MaxNLocator(12))
            ax.set_ylim(0, max(40, float(np.nanmax(s.values)) * 1.2))
            plt.xticks(rotation=45)
            ax.legend(["Rolling mean", "Daily precipitation"])
            fig.tight_layout()
            plt.show()

        return s


# =========================
# Teros
# =========================
class Teros:
    COORD_PATH = os.path.join("Data", "Teros Piezo", "coordonnees.xlsx")
    DATA_PATH = os.path.join("Data", "Teros Piezo", "teros_piezo.csv")

    def __init__(self, coord_path=None, data_path=None):
        """Loader & simple plots for Teros probe coordinates and VWC time series."""
        self.coord_path = coord_path or self.COORD_PATH
        self.data_path = data_path or self.DATA_PATH
        self.sampler_coords = self.import_coordinates()
        self.data = self.import_vwc_values()

    def import_coordinates(self):
        """Read sampler coordinates; index by 'Sampler' with ['North','East'] columns."""
        df = pd.read_excel(self.coord_path)
        if "Sampler" not in df.columns:
            raise RuntimeError("Coordinates file missing 'Sampler' column.")
        cols = {c.lower(): c for c in df.columns}
        north = cols.get("north", "North")
        east = cols.get("east", "East")
        sam = cols.get("sampler", "Sampler")
        out = df.set_index(sam)[[north, east]].copy()
        out.columns = ["North", "East"]
        return out

    def import_vwc_values(self):
        """Read Teros CSV with a 'Dates (hours)' datetime column and T_* probe columns."""
        df = pd.read_csv(self.data_path, parse_dates=["Dates (hours)"])
        return df

    def plot_vwc_evolution(self, plot=True, how="median"):
        """
        Plot median/mean daily VWC per probe, restricted to 16 May 2023 → 02 Feb 2024.
        """
        df = self.data.copy()
        if "Dates (hours)" not in df.columns:
            raise RuntimeError("Teros CSV missing 'Dates (hours)' column.")
        df = df.set_index("Dates (hours)").sort_index()

        vcols = [c for c in df.columns if c.startswith("T_")]
        if not vcols:
            raise RuntimeError("No Teros probe columns found (expected columns starting with 'T_').")

        # daily per-probe aggregation
        daily = df[vcols].resample("D").median() if how.lower() == "median" else df[vcols].resample("D").mean()

        # ---- clip to requested window ----
        start = pd.Timestamp("2023-05-16")
        end = pd.Timestamp("2024-02-02")
        daily = daily.loc[start:end]

        if daily.empty:
            print("[INFO] No data in the requested window (2023-05-16 → 2024-02-02).")
            return daily

        if plot:
            plt.figure(figsize=(12, 8))
            # plot only columns with at least one finite value in the window
            valid_cols = [c for c in daily.columns if np.isfinite(daily[c].to_numpy()).any()]
            for c in valid_cols:
                plt.plot(daily.index, daily[c], label=c)
            plt.xlabel("Date")
            plt.ylabel("VWC")
            plt.legend(loc="upper left", ncol=2)
            plt.title(f"Teros – daily {how} VWC (16 May 2023 → 02 Feb 2024)")
            plt.tight_layout()
            plt.show()

        return daily

    def plot_piezo_sampler_locations(self, field_suffix_regex=r"[AB]$", show_fields=True):
        """
        Plot sampler locations with distinctive colors.
        Optionally draw Field A and Field B polygons behind the points
        (grey fill, black outline) using the same geometry as MultispecAnalysis.
        """
        sc = self.sampler_coords
        if field_suffix_regex:
            mask = sc.index.to_series().str.contains(field_suffix_regex, regex=True)
            sc = sc[mask]

        if sc.empty:
            print("[INFO] No samplers match the given filter.")
            return

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # --- background: field polygons from MultispecAnalysis ---
        if show_fields:
            try:

                fields_to_draw = ("A", "B")
                all_poly_xy = []  # for autoscale later

                for fld in fields_to_draw:
                    # Build polygon in map coords exactly like MSA.apply_field_mask()
                    msa = MultispecAnalysis(field_letter=fld)
                    ref_T = msa._get_reference_transform()
                    ij = msa._field_polygon_indices()  # (col=x, row=y) pixel vertices
                    poly_xy = [rasterio.transform.xy(ref_T, int(r), int(c), offset="center") for c, r in ij]

                    # Draw polygon (grey fill, black outline)
                    ax.add_patch(
                        MplPolygon(
                            poly_xy,
                            closed=True,
                            facecolor=(0.7, 0.7, 0.7, 0.35),  # light grey, semi-transparent
                            edgecolor="black",
                            linewidth=1.5,
                            zorder=0,
                        )
                    )
                    # optional label at polygon centroid
                    cx = np.mean([p[0] for p in poly_xy])
                    cy = np.mean([p[1] for p in poly_xy])
                    ax.text(cx, cy, f"Field {fld}", ha="center", va="center", fontsize=9, color="k", zorder=1)
                    all_poly_xy.extend(poly_xy)
            except Exception as e:
                print(f"[WARN] Could not draw field polygons: {e}")

        # --- samplers ---
        colors = plt.cm.tab20(np.linspace(0, 1, len(sc)))
        for (name, row), col in zip(sc.iterrows(), colors):
            ax.scatter(row["East"], row["North"], color=col, marker="^", label=name, zorder=2)

        # --- axes / extent / styling ---
        xs = sc["East"].tolist()
        ys = sc["North"].tolist()
        if show_fields and len(all_poly_xy):
            xs += [p[0] for p in all_poly_xy]
            ys += [p[1] for p in all_poly_xy]
        if xs and ys:
            dx = max(1.0, (max(xs) - min(xs)) * 0.05)
            dy = max(1.0, (max(ys) - min(ys)) * 0.05)
            ax.set_xlim(min(xs) - dx, max(xs) + dx)
            ax.set_ylim(min(ys) - dy, max(ys) + dy)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_title("Piezo Sampler Locations")
        ax.grid(True)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        plt.show()


# =========================
# Water Table
# =========================
class WaterTable:
    def __init__(
        self,
        path=os.path.join("Data", "Water Table", "profondeur nappe-final.xlsx"),
        coord_path=os.path.join("Data", "Teros Piezo", "coordonnees.xlsx"),
    ):
        self.path = path
        self.coord_path = coord_path
        self.data, self.coord = self.import_data()
        self.sampler_coords = self.import_coordinates()

    def import_data(self):
        """Load WT time series (Excel) and coordinates (Excel)."""
        wt_data = pd.read_excel(self.path)
        wt_coord = pd.read_excel(self.coord_path)
        return wt_data, wt_coord

    def import_coordinates(self):
        """Sampler coordinates indexed by 'Sampler'."""
        df = pd.read_excel(self.coord_path)
        if "Sampler" not in df.columns:
            raise RuntimeError("Coordinates file missing 'Sampler' column.")
        cols = {c.lower(): c for c in df.columns}
        north = cols.get("north", "North")
        east = cols.get("east", "East")
        sam = cols.get("sampler", "Sampler")
        out = df.set_index(sam)[[north, east]].copy()
        out.columns = ["North", "East"]
        return out

    def plot_wt_evolution(self):
        """Plot daily median water-table depth for each sampler (16 May 2023 → 02 Feb 2024)."""
        df = self.data.copy()
        if "Time" not in df.columns:
            raise RuntimeError("Water table data missing 'Time' column.")
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df.set_index("Time").sort_index()

        # numeric-only columns (samplers)
        val_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
        if not val_cols:
            print("[INFO] No numeric sampler columns to plot.")
            return

        # daily median, then restrict period
        daily = df[val_cols].resample("D").median()

        start = pd.to_datetime("2023-05-16")
        end = pd.to_datetime("2024-02-02")
        daily = daily.loc[start:end]

        if daily.empty:
            print("[INFO] No data in the requested period (16 May 2023 → 02 Feb 2024).")
            return

        plt.figure(figsize=(12, 8))
        for c in val_cols:
            plt.plot(daily.index, daily[c], label=c)
        plt.xlabel("Date")
        plt.ylabel("Water table [cm]")
        plt.legend(loc="upper left", ncol=2)
        plt.title("Depth Water Table Evolution (daily median)\n16 May 2023 → 02 Feb 2024")
        plt.gca().set_xlim(start, end)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return daily

    def plot_wt_sampler_locations(self, show_fields=True):
        """Plot LS1..LS5 sampler locations (no A/B suffix), with fields in background."""
        sc = self.sampler_coords
        mask = sc.index.to_series().str.contains(r"LS[1-5]") & ~sc.index.to_series().str.contains(r"[AB]$")
        sel = sc[mask]
        if sel.empty:
            print("[INFO] No LS1–LS5 samplers found (without A/B suffix).")
            return

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # --- background: field polygons (A & B) from MultispecAnalysis ---
        all_poly_xy = []
        if show_fields:
            try:
                for fld in ("A", "B"):
                    msa = MultispecAnalysis(field_letter=fld)
                    ref_T = msa._get_reference_transform()
                    ij = msa._field_polygon_indices()  # (col=x, row=y) in pixel space
                    poly_xy = [rasterio.transform.xy(ref_T, int(r), int(c), offset="center") for c, r in ij]

                    ax.add_patch(
                        MplPolygon(
                            poly_xy,
                            closed=True,
                            facecolor=(0.7, 0.7, 0.7, 0.35),  # light grey, semi-transparent
                            edgecolor="black",
                            linewidth=1.5,
                            zorder=0,
                        )
                    )
                    # optional label at centroid
                    cx = np.mean([p[0] for p in poly_xy])
                    cy = np.mean([p[1] for p in poly_xy])
                    ax.text(cx, cy, f"Field {fld}", ha="center", va="center", fontsize=9, color="k", zorder=1)
                    all_poly_xy.extend(poly_xy)
            except Exception as e:
                print(f"[WARN] Could not draw field polygons: {e}")

        # --- samplers ---
        colors = plt.cm.tab20(np.linspace(0, 1, len(sel)))
        for (name, row), col in zip(sel.iterrows(), colors):
            ax.scatter(row["East"], row["North"], color=col, marker="^", label=name, zorder=2)

        # --- extent / styling ---
        xs = sel["East"].tolist()
        ys = sel["North"].tolist()
        if show_fields and all_poly_xy:
            xs += [p[0] for p in all_poly_xy]
            ys += [p[1] for p in all_poly_xy]
        if xs and ys:
            dx = max(1.0, (max(xs) - min(xs)) * 0.05)
            dy = max(1.0, (max(ys) - min(ys)) * 0.05)
            ax.set_xlim(min(xs) - dx, max(xs) + dx)
            ax.set_ylim(min(ys) - dy, max(ys) + dy)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_title("Water Table Sampler Locations (LS1–LS5)")
        ax.grid(True)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        plt.show()


def plot_hydro_soil_summary(
    rainfall: Rainfall,
    teros: Teros,
    watertable: WaterTable,
    rolling_window: int = 10,
    start="2023-05-16",
    end="2024-02-02",
):
    """
    Summary chart for 16-May-2023 → 02-Feb-2024:
      - Rainfall rolling mean (mm)
      - Teros daily median VWC by field:
          Field A = mean(LS3A, LS3B)
          Field B = mean(LS4A, LS4B, LS5A, LS5B)
      - Water-table depth by field:
          Field A = LS3
          Field B = mean(LS4, LS5)

    Three y-axes (shared time x-axis): VWC (left), WT depth (right), Rainfall (right, outward).
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # ---------- Rainfall (daily totals → rolling mean) ----------
    frames = rainfall.import_excel()
    if not frames:
        print("[INFO] No rainfall files found.")
        return None

    all_precip, all_dates = [], []
    for rf in frames:
        cols = {c.lower(): c for c in rf.columns}
        if "prcp" in cols and "date" in cols:
            prcp_col, date_col = cols["prcp"], cols["date"]
            all_precip.extend(pd.to_numeric(rf[prcp_col], errors="coerce").tolist())
            all_dates.extend(pd.to_datetime(rf[date_col], errors="coerce").tolist())

    rain = pd.Series(all_precip, index=pd.to_datetime(all_dates)).dropna()
    rain = rain.sort_index().groupby(rain.index.normalize()).sum()
    rain = rain.loc[start:end]
    rain_rm = rain.rolling(window=int(max(1, rolling_window)), center=True, min_periods=1).mean()

    # ---------- Teros VWC (daily median per probe, then aggregate by field) ----------
    td = teros.data.copy()
    if "Dates (hours)" not in td.columns:
        raise RuntimeError("Teros CSV missing 'Dates (hours)' column.")
    td = td.set_index("Dates (hours)").sort_index()
    vcols = [c for c in td.columns if c.startswith("T_")]
    if not vcols:
        raise RuntimeError("No Teros probe columns found (expected columns starting with 'T_').")
    teros_daily = td[vcols].resample("D").median().loc[start:end]

    def find_cols(df_cols, patterns):
        pats = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
        return [c for c in df_cols if any(p.search(c) for p in pats)]

    # Field A: LS3A & LS3B
    a_cols = find_cols(teros_daily.columns, [r"LS?3A", r"LS?3B"])
    # Field B: LS4A, LS4B, LS5A, LS5B
    b_cols = find_cols(teros_daily.columns, [r"LS?4A", r"LS?4B", r"LS?5A", r"LS?5B"])

    if not a_cols:
        print("[WARN] No Teros VWC columns matched LS3A/LS3B for Field A.")
    if not b_cols:
        print("[WARN] No Teros VWC columns matched LS4*/LS5* for Field B.")

    vwc_A = teros_daily[a_cols].mean(axis=1) if a_cols else pd.Series(index=teros_daily.index, dtype=float)
    vwc_B = teros_daily[b_cols].mean(axis=1) if b_cols else pd.Series(index=teros_daily.index, dtype=float)

    # ---------- Water table (daily median per sampler, then aggregate by field) ----------
    wtd = watertable.data.copy()
    if "Time" not in wtd.columns:
        raise RuntimeError("Water table data missing 'Time' column.")
    wtd["Time"] = pd.to_datetime(wtd["Time"], errors="coerce")
    wtd = wtd.set_index("Time").sort_index()

    # keep numeric sampler columns (LS3, LS4, LS5 etc.)
    wt_cols = [c for c in wtd.columns if wtd[c].dtype.kind in "fi"]
    wt_daily = wtd[wt_cols].resample("D").median().loc[start:end]

    # pick LS3, LS4, LS5 (allowing case/format variations)
    ls3_cols = find_cols(wt_daily.columns, [r"^LS?0*3\b", r"\bLS?0*3\b"])
    ls4_cols = find_cols(wt_daily.columns, [r"^LS?0*4\b", r"\bLS?0*4\b"])
    ls5_cols = find_cols(wt_daily.columns, [r"^LS?0*5\b", r"\bLS?0*5\b"])

    if not ls3_cols:
        print("[WARN] No WT columns matched LS3 for Field A.")
    if not (ls4_cols or ls5_cols):
        print("[WARN] No WT columns matched LS4/LS5 for Field B.")

    wt_A = wt_daily[ls3_cols].mean(axis=1) if ls3_cols else pd.Series(index=wt_daily.index, dtype=float)
    wt_B = pd.concat(
        [
            wt_daily[ls4_cols].mean(axis=1) if ls4_cols else pd.Series(index=wt_daily.index, dtype=float),
            wt_daily[ls5_cols].mean(axis=1) if ls5_cols else pd.Series(index=wt_daily.index, dtype=float),
        ],
        axis=1,
    ).mean(axis=1)

    # ---------- Align indices (outer join on time) ----------
    idx = pd.date_range(start, end, freq="D")
    rain_rm = rain_rm.reindex(idx)
    vwc_A = vwc_A.reindex(idx)
    vwc_B = vwc_B.reindex(idx)
    wt_A = wt_A.reindex(idx)
    wt_B = wt_B.reindex(idx)

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(12, 7))

    # consistent, colorblind-friendly palette
    COLORS = {
        "rain": "#1f78b4",  # blue
        "vwc_A": "#1b9e77",  # deep green
        "vwc_B": "#66a61e",  # olive
        "wt_A": "#d95f02",  # orange
        "wt_B": "#8c564b",  # brown
    }

    # Left Y: VWC (greens)
    (l1,) = ax.plot(
        vwc_A.index,
        vwc_A.values,
        color=COLORS["vwc_A"],
        lw=1.5,
        marker="o",
        markersize=4,
        markevery=14,
        label="VWC Field A (LS3A+LS3B mean)",
    )
    (l2,) = ax.plot(
        vwc_B.index,
        vwc_B.values,
        color=COLORS["vwc_B"],
        lw=1.5,
        marker="^",
        markersize=4,
        markevery=14,
        label="VWC Field B (LS4/LS5 mean)",
    )
    ax.set_ylabel("VWC [-]", color=COLORS["vwc_A"])
    ax.tick_params(axis="y", colors=COLORS["vwc_A"])
    ax.set_xlabel("Date")
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)

    # Right Y: Water table (orange/brown)
    ax2 = ax.twinx()
    (r1,) = ax2.plot(
        wt_A.index,
        wt_A.values,
        color=COLORS["wt_A"],
        lw=1.5,
        marker="s",
        markersize=4,
        markevery=21,
        label="WT depth Field A (LS3)",
    )
    (r2,) = ax2.plot(
        wt_B.index,
        wt_B.values,
        color=COLORS["wt_B"],
        lw=1.5,
        linestyle=":",
        marker="D",
        markersize=4,
        markevery=21,
        label="WT depth Field B (LS4/LS5 mean)",
    )
    ax2.set_ylabel("Water table depth [cm]", color=COLORS["wt_A"])
    ax2.tick_params(axis="y", colors=COLORS["wt_A"])
    ax2.spines["right"].set_color(COLORS["wt_A"])

    # Second Right Y (offset): Rainfall rolling mean (blue)
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 60))  # push third axis outward
    (rrm,) = ax3.plot(
        rain_rm.index,
        rain_rm.values,
        color=COLORS["rain"],
        linewidth=1,
        label=f"Rainfall (rolling mean, {rolling_window} d)",
    )
    ax3.fill_between(rain_rm.index, 0, rain_rm.values, color=COLORS["rain"], alpha=0.05, zorder=0)
    ax3.set_ylabel("Precipitation [mm]", color=COLORS["rain"])
    ax3.tick_params(axis="y", colors=COLORS["rain"])
    ax3.spines["right"].set_color(COLORS["rain"])

    # Unified legend (order: VWC A/B, WT A/B, Rain)
    handles = [l1, l2, r1, r2, rrm]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper left", ncol=1, frameon=True)

    ax.set_title("Hydro–Soil Summary • 16 May 2023 → 02 Feb 2024")
    fig.tight_layout()
    plt.show()


def analyze_rainfall_response(
    rainfall: Rainfall,
    teros: Teros,
    watertable: WaterTable,
    start="2023-05-16",
    end="2024-02-02",
    max_lag_days=30,
    rain_windows=(1, 3, 5, 7, 14, 30),
    min_pairs=10,
    plot=True,
):
    """
    Quantify lagged links between rainfall and (i) VWC (Field A/B), (ii) WT depth (Field A/B).
    - Uses daily rainfall totals (no smoothing).
    - Tests correlations at lags (0..max_lag_days).
    - Also tests accumulated rainfall windows vs *daily change* in response (ΔVWC, ΔWT).
    Returns dict with best lags/correlations and (optionally) plots:
      1) Cross-correlation (CCF) curves per response (rain vs response levels and rain vs Δresponse)
      2) Scatter at the best (window, lag) combo for each response with R and R².
    """

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # ---------- Rainfall (daily totals) ----------
    frames = rainfall.import_excel()
    if not frames:
        print("[INFO] No rainfall files found.")
        return None
    all_precip, all_dates = [], []
    for rf in frames:
        cols = {c.lower(): c for c in rf.columns}
        if "prcp" in cols and "date" in cols:
            all_precip.extend(pd.to_numeric(rf[cols["prcp"]], errors="coerce").tolist())
            all_dates.extend(pd.to_datetime(rf[cols["date"]], errors="coerce").tolist())
    rain = pd.Series(all_precip, index=pd.to_datetime(all_dates)).dropna().sort_index()
    rain = rain.groupby(rain.index.normalize()).sum().loc[start:end]

    # ---------- Teros VWC (daily median → fields) ----------
    td = teros.data.copy()
    td = td.set_index("Dates (hours)").sort_index()
    vcols = [c for c in td.columns if c.startswith("T_")]
    teros_daily = td[vcols].resample("D").median().loc[start:end]

    def find_cols(df_cols, patterns):
        pats = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
        return [c for c in df_cols if any(p.search(c) for p in pats)]

    a_cols = find_cols(teros_daily.columns, [r"LS?3A", r"LS?3B"])  # Field A
    b_cols = find_cols(teros_daily.columns, [r"LS?4A", r"LS?4B", r"LS?5A", r"LS?5B"])  # Field B
    vwc_A = teros_daily[a_cols].mean(axis=1) if a_cols else pd.Series(index=teros_daily.index, dtype=float)
    vwc_B = teros_daily[b_cols].mean(axis=1) if b_cols else pd.Series(index=teros_daily.index, dtype=float)

    # ---------- Water table (daily median → fields) ----------
    wt = watertable.data.copy()
    wt["Time"] = pd.to_datetime(wt["Time"], errors="coerce")
    wt = wt.set_index("Time").sort_index()
    wt_daily = wt[[c for c in wt.columns if wt[c].dtype.kind in "fi"]].resample("D").median().loc[start:end]
    ls3 = find_cols(wt_daily.columns, [r"^LS?0*3\b", r"\bLS?0*3\b"])  # Field A
    ls4 = find_cols(wt_daily.columns, [r"^LS?0*4\b", r"\bLS?0*4\b"])  # Field B
    ls5 = find_cols(wt_daily.columns, [r"^LS?0*5\b", r"\bLS?0*5\b"])
    wt_A = wt_daily[ls3].mean(axis=1) if ls3 else pd.Series(index=wt_daily.index, dtype=float)
    wt_B = pd.concat(
        [
            wt_daily[ls4].mean(axis=1) if ls4 else pd.Series(index=wt_daily.index, dtype=float),
            wt_daily[ls5].mean(axis=1) if ls5 else pd.Series(index=wt_daily.index, dtype=float),
        ],
        axis=1,
    ).mean(axis=1)

    # ---------- Reindex to common daily axis ----------
    idx = pd.date_range(start, end, freq="D")
    rain = rain.reindex(idx).fillna(0.0)  # missing days = 0 rain
    vwc_A = vwc_A.reindex(idx)
    vwc_B = vwc_B.reindex(idx)
    wt_A = wt_A.reindex(idx)
    wt_B = wt_B.reindex(idx)

    # Helpers
    def pearson(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < min_pairs:
            return np.nan
        return float(np.corrcoef(x[m], y[m])[0, 1])

    def ccf_series(x, y, max_lag, diff_y=False):
        """corr( rain_t , y_{t+lag} ), lag ≥ 0. If diff_y, correlate with Δy."""
        out = []
        yy = y.diff() if diff_y else y
        for L in range(0, max_lag + 1):
            xs = x.iloc[:-L] if L > 0 else x
            ys = yy.iloc[L:]
            out.append(pearson(xs, ys))
        return pd.Series(out, index=np.arange(0, max_lag + 1))

    def best_window_lag(rain_s, y, windows, max_lag, diff_y=True, min_pairs=10):
        """
        Grid-search accumulated rainfall window vs lag; returns best combo and the
        aligned series used for the scatter.
        """
        best = {"R": np.nan, "lag": None, "win": None, "series_x": None, "series_y": None}
        yy = y.diff() if diff_y else y

        for w in windows:
            acc = rain_s.rolling(w, min_periods=1).sum()
            for L in range(0, max_lag + 1):
                xs = acc.iloc[:-L] if L > 0 else acc  # window ending at t
                ys = yy.iloc[L:]  # response (or Δ) at t+L
                # align & drop NaNs
                xs_al, ys_al = xs.align(ys, join="inner")
                m = np.isfinite(xs_al) & np.isfinite(ys_al)
                if m.sum() < min_pairs:
                    continue
                r = float(np.corrcoef(xs_al[m], ys_al[m])[0, 1])
                if np.isfinite(r) and (not np.isfinite(best["R"]) or abs(r) > abs(best["R"])):
                    best.update({"R": r, "lag": L, "win": w, "series_x": xs_al[m], "series_y": ys_al[m]})
        return best

    responses = {"VWC_A": vwc_A, "VWC_B": vwc_B, "WT_A": wt_A, "WT_B": wt_B}

    results = {}

    # ---------- Plots: CCF curves ----------
    if plot:
        fig1, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=150)
        axes = axes.ravel()

    for i, (name, y) in enumerate(responses.items()):
        # CCF vs levels and vs deltas
        ccf_lvl = ccf_series(rain, y, max_lag_days, diff_y=False)
        ccf_dlt = ccf_series(rain, y, max_lag_days, diff_y=True)
        best_lvl_lag = int(ccf_lvl.abs().idxmax()) if np.isfinite(ccf_lvl).any() else None
        best_dlt_lag = int(ccf_dlt.abs().idxmax()) if np.isfinite(ccf_dlt).any() else None
        best_lvl_R = float(ccf_lvl.loc[best_lvl_lag]) if best_lvl_lag is not None else np.nan
        best_dlt_R = float(ccf_dlt.loc[best_dlt_lag]) if best_dlt_lag is not None else np.nan

        # Grid search: accumulated rainfall vs Δresponse
        best_grid = best_window_lag(rain, y, rain_windows, max_lag_days, diff_y=True)

        results[name] = {
            "ccf_levels": {"best_lag_days": best_lvl_lag, "R": best_lvl_R},
            "ccf_deltas": {"best_lag_days": best_dlt_lag, "R": best_dlt_R},
            "accum_vs_delta_best": {
                "window_days": int(best_grid["win"]) if best_grid["win"] is not None else None,
                "lag_days": int(best_grid["lag"]) if best_grid["lag"] is not None else None,
                "R": float(best_grid["R"]) if np.isfinite(best_grid["R"]) else np.nan,
            },
        }

        if plot:
            ax = axes[i]
            ax.plot(ccf_lvl.index, ccf_lvl.values, label="rain vs LEVEL")
            ax.plot(ccf_dlt.index, ccf_dlt.values, linestyle="--", label="rain vs Δ")
            # highlight bests
            if best_lvl_lag is not None:
                ax.axvline(best_lvl_lag, ymin=0, ymax=1, alpha=0.25)
            if best_dlt_lag is not None:
                ax.axvline(best_dlt_lag, ymin=0, ymax=1, alpha=0.25, linestyle=":")
            ax.set_title(
                f"{name} — CCF (0…{max_lag_days} d)\n"
                f"best level: lag={best_lvl_lag} d, R={best_lvl_R:.2f} | "
                f"best Δ: lag={best_dlt_lag} d, R={best_dlt_R:.2f}"
            )
            ax.set_xlabel("Lag (days)  [positive = response after rain]")
            ax.set_ylabel("Pearson R")
            ax.grid(True)
            ax.legend(fontsize=8)

    if plot:
        fig1.suptitle("Rainfall → Response Cross-Correlation", y=1.02)
        fig1.tight_layout()
        plt.show()

    # ---------- Plots: scatter at the best (window, lag) for each response ----------
    if plot:
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 6), dpi=150)
        axes2 = axes2.ravel()

    for i, (name, y) in enumerate(responses.items()):
        best = best_window_lag(rain, y, rain_windows, max_lag_days, diff_y=True, min_pairs=min_pairs)
        results[name]["scatter_best"] = {
            "window_days": int(best["win"]) if best["win"] is not None else None,
            "lag_days": int(best["lag"]) if best["lag"] is not None else None,
            "R": float(best["R"]) if np.isfinite(best["R"]) else np.nan,
            "R2": float(best["R"] ** 2) if np.isfinite(best["R"]) else np.nan,
        }
        if plot:
            ax = axes2[i]
            if best["series_x"] is not None:
                xs, ys = best["series_x"], best["series_y"]
                ax.scatter(xs, ys, s=12, alpha=0.6)
                if xs.size >= 2:
                    b, a = np.polyfit(xs.values, ys.values, 1)
                    xr = np.linspace(xs.min(), xs.max(), 100)
                    ax.plot(xr, a + b * xr)
                    ax.set_title(
                        f"{name}  best win={best['win']} d, lag={best['lag']} d\nR={best['R']:.2f}, R²={best['R']**2:.2f}"
                    )
            else:
                ax.text(0.5, 0.5, "No valid pairs", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Accumulated rain [mm]")
            ax.set_ylabel("Δ response per day")
            ax.grid(True)

    if plot:
        fig2.suptitle("Best Accumulated Rain (window) vs ΔResponse (scatter + fit)", y=1.02)
        fig2.tight_layout()
        plt.show()

    # ---------- Nicely print numeric summary ----------
    print("\n=== Rainfall → Soil/WT response (best lags & strengths) ===")
    for k, v in results.items():
        lvl = v["ccf_levels"]
        dlt = v["ccf_deltas"]
        bst = v["accum_vs_delta_best"]
        print(
            f"{k:6s} | level: lag={lvl['best_lag_days']} d, R={lvl['R']:.2f}  | "
            f"Δ: lag={dlt['best_lag_days']} d, R={dlt['R']:.2f}  | "
            f"accum_vs_Δ: win={bst['window_days']} d, lag={bst['lag_days']} d, R={bst['R']:.2f}"
        )

    return {"series": {"rain": rain, "vwc_A": vwc_A, "vwc_B": vwc_B, "wt_A": wt_A, "wt_B": wt_B}, "results": results}


# =========================
# Thermal rasters
# =========================
class Thermal:
    """Thin helper around thermal rasters."""

    TEMPERATURE_RASTER = sorted(glob.glob(os.path.join("Data", "Thermal", "**", "*.[tT][iI][fF]"), recursive=True))

    @staticmethod
    def read_band(path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            prof = src.profile.copy()
        return arr, prof

    @staticmethod
    def to_celsius(arr):
        """Convert Kelvin-like temperatures to °C when values look like >150K."""
        arr = np.array(arr, dtype=np.float32, copy=True)
        if np.isfinite(arr).any() and np.nanmedian(arr) > 150:
            arr = arr - 273.15
        arr[(arr < -100) | (arr > 80)] = np.nan
        return arr

    # ---- multispec-style thermal plotting ----
    def plot(
        self,
        msa,  # MultispecAnalysis instance (mask + renderer)
        sample_number=0,
        unit="C",  # "C" for Celsius, "K" to leave raw units
        use_mask=True,  # same default behavior as TVDI maps
        vmin=None,  # robust percentiles inside field if None
        vmax=None,  # robust percentiles inside field if None
        cmap="inferno",  # warmer default for temperature
    ):
        """
        Plot a thermal raster with the same styling used by MultispecAnalysis maps.
        Reuses msa._render_geo_map_like_tvdi for identical formatting.
        """
        # 1) load raster
        try:
            src_path = self.TEMPERATURE_RASTER[sample_number]
        except IndexError:
            raise IndexError(
                f"No thermal raster at index {sample_number} (found {len(self.TEMPERATURE_RASTER)} files)."
            )

        arr, prof = self.read_band(src_path)

        # 2) temperature units / cleaning
        if str(unit).upper().startswith("C"):
            arr = self.to_celsius(arr)
            cbar_label = "LST [°C]"
        else:
            a = arr.copy()
            a[(a < 150) | (a > 400)] = np.nan
            arr = a
            cbar_label = "LST [K]"

        # 3) title date, consistent with MSA
        date_obj = msa._parse_date_from_name(src_path)
        date_str = date_obj.strftime("%d/%m/%Y") if date_obj else "unknown date"
        title = f"Thermal • Field {msa.field_letter} • {date_str}"

        # 4) robust stretch inside field (for consistent look)
        arr_for_range = msa.apply_field_mask(arr, prof) if use_mask else arr
        if vmin is None or vmax is None:
            finite = arr_for_range[np.isfinite(arr_for_range)]
            if finite.size > 0:
                p2, p98 = np.nanpercentile(finite, [2, 98])
                if vmin is None:
                    vmin = p2
                if vmax is None:
                    vmax = p98

        # 5) render with the exact same helper
        msa._render_geo_map_like_tvdi(
            array=arr,
            profile=prof,
            title_text=title,
            cbar_label=cbar_label,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            use_mask=use_mask,
        )


def _sorted_glob(pattern: str):
    rec = "**" in pattern
    return sorted(glob.glob(pattern, recursive=rec), key=lambda s: s.lower())


NDVI_RASTER = _sorted_glob(os.path.join("Data", "Multispectral", "NDVI", "*.[tT][iI][fF]"))
BLUE_RASTER = _sorted_glob(os.path.join("Data", "Multispectral", "Blue", "*.[tT][iI][fF]"))
GREEN_RASTER = _sorted_glob(os.path.join("Data", "Multispectral", "Green", "*.[tT][iI][fF]"))
RED_RASTER = _sorted_glob(os.path.join("Data", "Multispectral", "Red", "*.[tT][iI][fF]"))
NIR_RASTER = _sorted_glob(os.path.join("Data", "Multispectral", "NIR", "*.[tT][iI][fF]"))
RED_EDGE_RASTER = _sorted_glob(os.path.join("Data", "Multispectral", "Rededge", "*.[tT][iI][fF]"))
TEMP_RASTER = Thermal.TEMPERATURE_RASTER


def _safe_list(global_name: str, fallback_pattern: str):
    """
    If a global like NDVI_RASTER exists and is non-empty, return it;
    otherwise glob the fallback pattern.
    """
    g = globals().get(global_name, None)
    if isinstance(g, (list, tuple)) and len(g) > 0:
        return list(g)
    return _sorted_glob(fallback_pattern)


class TopoPeatAnalysis:
    """
    Topography–peat–vegetation comparison for Field A vs B.

    Inputs (single-date rasters):
      - dtm_path: LiDAR DTM (m)
      - peat_path: modeled peat depth (units as in TIFF)
      - veg_path: vegetation classes (integer/categorical)

    If you pass an existing MultispecAnalysis instance (msa=...), we re-use its
    field polygons for perfect consistency. Otherwise, provide polygons yourself
    via 'field_polygons_xy' (dict {'A': [(x,y),...], 'B': [(x,y),...]}) and
    their CRS via 'polygons_crs'.
    """

    def __init__(
        self,
        dtm_path=r"D:\Coding\Python\Master-Thesis\Data\Peat Vege Elev\MR20230607_georeferenced_lidar_dtm.tif",
        peat_path=r"D:\Coding\Python\Master-Thesis\Data\Peat Vege Elev\peat_depth_mean_modelling.tif",
        veg_path=r"D:\Coding\Python\Master-Thesis\Data\Peat Vege Elev\vege_rf_new.tif",
        field_letter="A",
        msa: "MultispecAnalysis" = None,  # optional shortcut to reuse your polygons
        field_polygons_xy: dict | None = None,  # {'A': [(x,y),...], 'B': [(x,y),...]} in polygons_crs
        polygons_crs=None,
    ):

        self.paths = {"dtm": dtm_path, "peat": peat_path, "veg": veg_path}
        self.field_letter = str(field_letter).upper()
        self._msa = msa

        # lazy caches
        self._poly_xy = None  # dict {'A': [(x,y),...], 'B': [(x,y),...]} in self._poly_crs
        self._poly_crs = None  # rasterio CRS
        self._mask_cache = {}  # key=(field, crs_wkt, h, w, transform) -> mask ndarray

        # accept explicit polygons if provided
        if field_polygons_xy is not None:
            if not isinstance(field_polygons_xy, dict) or not {"A", "B"} <= set(field_polygons_xy.keys()):
                raise ValueError("field_polygons_xy must be a dict with keys {'A','B'} and list of (x,y).")
            if polygons_crs is None:
                raise ValueError("When passing field_polygons_xy you must also provide polygons_crs.")
            self._poly_xy = {k: list(map(tuple, field_polygons_xy[k])) for k in ("A", "B")}
            self._poly_crs = polygons_crs

        # probe rasters & keep DTM profile as our default reference
        with rasterio.open(self.paths["dtm"]) as src:
            self._ref_profile = src.profile.copy()
            self._ref_crs = src.crs

    # ---------- shared helpers (style-aligned to your code) ----------
    @staticmethod
    def _mad_std(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan
        med = np.nanmedian(a)
        mad = np.nanmedian(np.abs(a - med))
        return 1.4826 * mad

    @staticmethod
    def _auto_zoom_to_valid_coords(ax, array, transform, pad_pixels=20):
        yy, xx = np.where(np.isfinite(array))
        if yy.size == 0:
            return
        r0, r1 = int(yy.min()), int(yy.max())
        c0, c1 = int(xx.min()), int(xx.max())
        x_ul, y_ul = rasterio.transform.xy(transform, r0, c0, offset="ul")
        x_lr, y_lr = rasterio.transform.xy(transform, r1, c1, offset="lr")
        px_w = transform.a
        px_h = abs(transform.e)
        pad_x = pad_pixels * abs(px_w)
        pad_y = pad_pixels * abs(px_h)
        xmin, xmax = min(x_ul, x_lr) - pad_x, max(x_ul, x_lr) + pad_x
        ymin, ymax = min(y_ul, y_lr) - pad_y, max(y_ul, y_lr) + pad_y
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    def _apply_lambert72_grid(self, ax, major=100, minor=50, labelsize=7):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_autoscale_on(False)

        def _ticks_within(lo, hi, step):
            if step is None or step <= 0:
                return np.array([])
            start = np.ceil(min(lo, hi) / step) * step
            stop = np.floor(max(lo, hi) / step) * step
            if stop < start:
                return np.array([])
            return np.arange(start, stop + 0.5 * step, step)

        xt_major = _ticks_within(*xlim, major)
        yt_major = _ticks_within(*ylim, major)
        ax.set_xticks(xt_major)
        ax.set_yticks(yt_major)
        if minor and minor > 0:
            xt_minor = _ticks_within(*xlim, minor)
            yt_minor = _ticks_within(*ylim, minor)
            ax.set_xticks(xt_minor, minor=True)
            ax.set_yticks(yt_minor, minor=True)

        ax.grid(which="major", linewidth=0.5, alpha=0.35)
        ax.grid(which="minor", linewidth=0.4, alpha=0.15)
        ax.tick_params(top=True, right=True, direction="out", length=3, labelsize=labelsize)
        ax.tick_params(which="minor", length=2)

        from matplotlib.ticker import ScalarFormatter

        fmtx = ScalarFormatter(useMathText=False)
        fmty = ScalarFormatter(useMathText=False)
        for f in (fmtx, fmty):
            f.set_scientific(False)
            f.set_useOffset(False)
        ax.xaxis.set_major_formatter(fmtx)
        ax.yaxis.set_major_formatter(fmty)
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.margins(0)

    def _add_north_arrow(self, ax, loc="NE", size=0.09, inset=(0.03, 0.01), font_size=9):
        corners = {"NE": (1.0, 1.0), "NW": (0.0, 1.0), "SE": (1.0, 0.0), "SW": (0.0, 0.0)}
        cx, cy = corners.get(loc, corners["NE"])
        x = cx - inset[0] if cx == 1.0 else cx + inset[0]
        y_top = cy - inset[1] if cy == 1.0 else cy + inset[1]
        y_bot = y_top - size if cy == 1.0 else y_top + size
        arrow_kwargs = dict(facecolor="k", edgecolor="k", width=1.0, headwidth=8, headlength=10, shrinkA=0, shrinkB=0)
        ax.annotate(
            "N",
            xy=(x, y_top),
            xytext=(x, y_bot),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            ha="center",
            va="center",
            color="k",
            fontsize=font_size,
            arrowprops=arrow_kwargs,
            annotation_clip=False,
        )

    def _render_map(
        self,
        array,
        profile,
        title_text,
        cbar_label="Value",
        cmap="viridis",
        vmin=None,
        vmax=None,
        categorical=False,
        classes=None,
    ):
        fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=300)
        fig.subplots_adjust(bottom=0.10, top=0.92, right=0.95)

        extent = rasterio.plot.plotting_extent(array, profile["transform"])
        if categorical:
            from matplotlib.colors import BoundaryNorm
            from matplotlib.colors import ListedColormap

            if classes is None:
                classes = np.unique(array[np.isfinite(array)]).astype(int).tolist()
            base = plt.get_cmap("tab20")
            colors = [base(i % 20) for i in range(len(classes))]
            cmap = ListedColormap(colors)
            bounds = np.array(classes + [classes[-1] + 1]) - 0.5
            norm = BoundaryNorm(bounds, cmap.N)
            im = ax.imshow(array, origin="upper", extent=extent, cmap=cmap, norm=norm)
        else:
            im = ax.imshow(array, origin="upper", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)

        self._auto_zoom_to_valid_coords(ax, array, profile["transform"], pad_pixels=30)
        ax.set_aspect("equal", adjustable="box")

        finite_mask = np.isfinite(array).astype(int)
        ax.contour(finite_mask, levels=[0.5], colors="k", linewidths=1.5, origin="upper", extent=extent)
        self._add_north_arrow(ax, loc="NE", size=0.09, inset=(0.03, 0.01), font_size=9)
        self._apply_lambert72_grid(ax, major=100, minor=50, labelsize=7)

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.10)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label)

        ax.set_title(title_text, pad=10)
        plt.show()

    # ---------- polygon plumbing ----------
    def _ensure_polygons(self):
        import rasterio

        if self._poly_xy is not None:
            return
        if self._msa is None:
            raise RuntimeError(
                "No polygons yet. Pass an existing MultispecAnalysis instance as msa=... "
                "or provide field_polygons_xy + polygons_crs to the constructor."
            )
        # Build polygon XY in MSA's reference CRS
        ref_prof = self._msa._get_reference_profile()
        ref_T = ref_prof["transform"]
        ref_crs = ref_prof["crs"]

        def _poly_xy_for(letter):
            old = self._msa.field_letter
            try:
                self._msa.field_letter = letter
                ij = self._msa._field_polygon_indices()  # (col=x, row=y) pixel indices
            finally:
                self._msa.field_letter = old
            xy = [rasterio.transform.xy(ref_T, int(r), int(c), offset="center") for c, r in ij]
            return xy

        self._poly_xy = {"A": _poly_xy_for("A"), "B": _poly_xy_for("B")}
        self._poly_crs = ref_crs

    def _field_mask_for(self, profile, field="A"):
        """
        Rasterize field polygon (reprojecting if needed) into the given profile grid.
        Caches masks (fast).
        """
        import pyproj
        import rasterio.features

        key = (field, str(profile["crs"]), profile["height"], profile["width"], tuple(profile["transform"]))
        if key in self._mask_cache:
            return self._mask_cache[key]

        self._ensure_polygons()
        poly_xy = self._poly_xy[str(field).upper()]
        src_crs = self._poly_crs
        dst_crs = profile["crs"]

        if str(src_crs) != str(dst_crs):
            # reproject polygon coordinates
            tr = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            poly_xy = [tr.transform(x, y) for (x, y) in poly_xy]

        shape = {"type": "Polygon", "coordinates": [poly_xy]}
        mask = rasterio.features.rasterize(
            [shape],
            out_shape=(profile["height"], profile["width"]),
            transform=profile["transform"],
            fill=0,
            default_value=1,
            dtype="uint8",
        ).astype(bool)
        self._mask_cache[key] = mask
        return mask

    @staticmethod
    def _read_raster(path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            prof = src.profile.copy()
        return arr, prof

    @staticmethod
    def _resample_to_profile(src_array, src_profile, dst_profile, resampling=rasterio.enums.Resampling.bilinear):
        dst = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype=np.float32)
        rasterio.warp.reproject(
            source=src_array,
            destination=dst,
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            resampling=resampling,
        )
        return dst

    # ---------- terrain derivatives ----------
    @staticmethod
    def _slope_aspect_from_dtm(dtm, transform):
        # pixel sizes
        dx = abs(transform.a)
        dy = abs(transform.e)
        # gradients (rows=y, cols=x)
        dzdy, dzdx = np.gradient(dtm, dy, dx)
        slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))  # 0..90 deg
        aspect = np.degrees(np.arctan2(-dzdx, dzdy))  # 0=N
        aspect = np.where(aspect < 0, 360.0 + aspect, aspect)
        return slope.astype(np.float32), aspect.astype(np.float32)

    @staticmethod
    def _hillshade(slope_deg, aspect_deg, azimuth=315.0, altitude=45.0):
        az = np.radians(azimuth)
        alt = np.radians(altitude)
        sl = np.radians(slope_deg)
        asp = np.radians(aspect_deg)
        hs = np.cos(alt) * np.cos(sl) + np.sin(alt) * np.sin(sl) * np.cos(az - asp)
        hs = np.clip(hs, 0, 1)
        return (hs * 255).astype(np.uint8)

    # ---------- public: core maps ----------
    def plot_maps(self, field_letter=None):
        """Elevation (+ hillshade), slope, peat depth, and vegetation (categorical) for one field."""
        if field_letter is None:
            field_letter = self.field_letter

        # --- read rasters ---
        dtm, p_dtm = self._read_raster(self.paths["dtm"])
        peat, p_peat = self._read_raster(self.paths["peat"])
        veg, p_veg = self._read_raster(self.paths["veg"])

        # --- sanitize DTM nodata BEFORE any derivatives ---
        nod = p_dtm.get("nodata", None)
        if nod is not None:
            dtm[dtm == nod] = np.nan
        # common fallbacks seen in LiDAR tiles
        dtm[(dtm == 0) | (dtm <= -9990)] = np.nan

        # --- resample peat/veg to DTM grid (veg = nearest, peat = bilinear) ---
        peat_r = self._resample_to_profile(peat, p_peat, p_dtm, resampling=rasterio.enums.Resampling.bilinear)
        veg_r = self._resample_to_profile(veg, p_veg, p_dtm, resampling=rasterio.enums.Resampling.nearest)

        # --- field mask (in DTM grid) ---
        m_field = self._field_mask_for(p_dtm, field=field_letter)
        dtm_f = np.where(m_field & np.isfinite(dtm), dtm, np.nan)
        peat_f = np.where(m_field & np.isfinite(peat_r), peat_r, np.nan)
        veg_f = np.where(m_field & np.isfinite(veg_r), veg_r, np.nan)

        # --- terrain derivatives from (sanitized) DTM ---
        slope, aspect = self._slope_aspect_from_dtm(dtm, p_dtm["transform"])
        slope_f = np.where(m_field & np.isfinite(dtm), slope, np.nan)  # ensure outside field is NaN
        hs = self._hillshade(slope, aspect)

        # =========================
        # Maps (style coherent)
        # =========================

        # 1) Elevation only
        self._render_map(
            dtm_f, p_dtm, f"Elevation (m) • Field {field_letter}", cbar_label="Elevation (m)", cmap="terrain"
        )

        # 2) Elevation + hillshade overlay
        fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=300)
        extent = rasterio.plot.plotting_extent(dtm_f, p_dtm["transform"])
        im1 = ax.imshow(dtm_f, origin="upper", extent=extent, cmap="terrain")
        ax.imshow(np.where(np.isfinite(dtm_f), hs, np.nan), origin="upper", extent=extent, cmap="gray", alpha=0.30)
        self._auto_zoom_to_valid_coords(ax, dtm_f, p_dtm["transform"], 30)
        ax.set_aspect("equal")
        self._add_north_arrow(ax)
        self._apply_lambert72_grid(ax)
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.10)
        cb = plt.colorbar(im1, cax=cax)
        cb.set_label("Elevation (m)")
        ax.set_title(f"Elevation + Hillshade • Field {field_letter}", pad=10)
        plt.show()

        # 3) Slope (deg) — with proper NaN masking (transparent outside field / nodata)
        fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=300)
        extent = rasterio.plot.plotting_extent(slope_f, p_dtm["transform"])
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(alpha=0.0)  # hide NaNs entirely
        # robust upper limit (avoid extreme outliers)
        try:
            vmax = float(np.nanpercentile(slope_f, 98))
            if not np.isfinite(vmax) or vmax < 5:
                vmax = 35.0
        except Exception:
            vmax = 35.0
        im = ax.imshow(np.ma.masked_invalid(slope_f), origin="upper", extent=extent, cmap=cmap, vmin=0.0, vmax=vmax)
        self._auto_zoom_to_valid_coords(ax, slope_f, p_dtm["transform"], 30)
        ax.set_aspect("equal", adjustable="box")
        # field outline for crisp border
        finite_mask = np.isfinite(slope_f).astype(int)
        ax.contour(finite_mask, levels=[0.5], colors="k", linewidths=1.5, origin="upper", extent=extent)
        self._add_north_arrow(ax, loc="NE", size=0.09, inset=(0.03, 0.01), font_size=9)
        self._apply_lambert72_grid(ax, major=100, minor=50, labelsize=7)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.10)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Slope (°)")
        ax.set_title(f"Slope (deg) • Field {field_letter}", pad=10)
        plt.show()

        # 4) Peat depth (model)
        self._render_map(
            peat_f, p_dtm, f"Peat depth (model) • Field {field_letter}", cbar_label="Peat depth (model)", cmap="YlGnBu"
        )

        # 5) Vegetation (categorical)
        classes = np.unique(veg_f[np.isfinite(veg_f)]).astype(int).tolist() if np.isfinite(veg_f).any() else [0]
        self._render_map(
            veg_f,
            p_dtm,
            f"Vegetation (classes) • Field {field_letter}",
            cbar_label="Class",
            categorical=True,
            classes=classes,
        )

    # ---------- public: A vs B comparative analysis ----------
    def compare_fields(self, print_stats=True):
        """Side-by-side stats and plots contrasting Field A vs Field B."""
        # align everything on DTM grid
        dtm, p_dtm = self._read_raster(self.paths["dtm"])
        peat, p_peat = self._read_raster(self.paths["peat"])
        veg, p_veg = self._read_raster(self.paths["veg"])
        peat_r = self._resample_to_profile(peat, p_peat, p_dtm, resampling=rasterio.enums.Resampling.bilinear)
        veg_r = self._resample_to_profile(veg, p_veg, p_dtm, resampling=rasterio.enums.Resampling.nearest)

        slope, aspect = self._slope_aspect_from_dtm(dtm, p_dtm["transform"])

        masks = {L: self._field_mask_for(p_dtm, field=L) for L in ("A", "B")}
        stats = {}

        def _summ(name, arr):
            a = arr[np.isfinite(arr)]
            if a.size == 0:
                return {"n": 0}
            return {
                "n": int(a.size),
                "mean": float(np.nanmean(a)),
                "median": float(np.nanmedian(a)),
                "p05": float(np.nanpercentile(a, 5)),
                "p95": float(np.nanpercentile(a, 95)),
                "sigma_mad": float(self._mad_std(a)),
            }

        for L in ("A", "B"):
            m = masks[L]
            stats[L] = {
                "elev": _summ("elev", np.where(m, dtm, np.nan)),
                "slope": _summ("slope", np.where(m, slope, np.nan)),
                "peat": _summ("peat", np.where(m, peat_r, np.nan)),
                "veg_hist": None,
            }
            # veg composition
            vf = veg_r[m & np.isfinite(veg_r)]
            if vf.size:
                uq, cnt = np.unique(vf.astype(int), return_counts=True)
                comp = {int(k): float(c) / float(vf.size) for k, c in zip(uq, cnt)}
                stats[L]["veg_hist"] = comp

        if print_stats:
            print("=== Field comparison (robust stats) ===")
            for L in ("A", "B"):
                s = stats[L]
                print(
                    f"\nField {L} (n pixels: elev={s['elev']['n']}, peat={s['peat']['n']}, veg={sum((s['veg_hist'] or {}).values())})"
                )
                for k in ("elev", "slope", "peat"):
                    v = s[k]
                    print(
                        f"  {k:>5}: mean={v['mean']:.3f}  median={v['median']:.3f}  p05={v['p05']:.3f}  p95={v['p95']:.3f}  σ_MAD={v['sigma_mad']:.3f}"
                    )
                if s["veg_hist"]:
                    print(
                        "  veg comp (proportion by class):", {k: round(v, 3) for k, v in sorted(s["veg_hist"].items())}
                    )

        # ---- Plots (style-aligned) ----
        # 1) Hypsometric curves (elevation CDF)
        fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
        for L in ("A", "B"):
            m = masks[L]
            z = dtm[m & np.isfinite(dtm)]
            if z.size:
                z = np.sort(z)
                y = np.linspace(0, 1, z.size)
                ax.plot(z, y, label=f"Field {L}")
        ax.set_xlabel("Elevation (m)")
        ax.set_ylabel("Cumulative proportion")
        ax.set_title("Hypsometric curves (Elevation CDF) • Fields A vs B")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

        # 2) Slope distributions
        fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
        for L in ("A", "B"):
            m = masks[L]
            s = slope[m & np.isfinite(slope)]
            if s.size:
                ax.hist(s, bins=30, histtype="step", density=True, label=f"Field {L}")
        ax.set_xlabel("Slope (°)")
        ax.set_ylabel("Density")
        ax.set_title("Slope distributions • Fields A vs B")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

        # 3) Peat depth — boxplots
        data, labels = [], []
        for L in ("A", "B"):
            m = masks[L]
            p = peat_r[m & np.isfinite(peat_r)]
            if p.size:
                data.append(p)
                labels.append(f"Field {L}")
        if data:
            fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=150)
            ax.boxplot(data, labels=labels, showfliers=False)
            ax.set_ylabel("Peat depth (model)")
            ax.set_title("Peat depth by field")
            ax.grid(True, axis="y", alpha=0.4)
            plt.tight_layout()
            plt.show()

        # 4) Vegetation composition bars
        # unify class set
        all_classes = sorted(set((stats["A"]["veg_hist"] or {}).keys()) | set((stats["B"]["veg_hist"] or {}).keys()))
        if all_classes:
            fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
            width = 0.35
            x = np.arange(len(all_classes))

            def _vals(L):
                d = stats[L]["veg_hist"] or {}
                return [d.get(c, 0.0) for c in all_classes]

            ax.bar(x - width / 2, _vals("A"), width, label="Field A")
            ax.bar(x + width / 2, _vals("B"), width, label="Field B")
            ax.set_xticks(x)
            ax.set_xticklabels(all_classes)
            ax.set_ylabel("Proportion")
            ax.set_xlabel("Vegetation class")
            ax.set_title("Vegetation composition • Fields A vs B")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.show()

        # 5) Peat depth vs Elevation (per field) + Pearson r
        from sklearn.metrics import r2_score

        fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, sharex=True, sharey=True)
        for ax, L in zip(axs, ("A", "B")):
            m = masks[L]
            sel = m & np.isfinite(dtm) & np.isfinite(peat_r)
            X = dtm[sel]
            Y = peat_r[sel]
            if X.size >= 20:
                ax.scatter(X, Y, s=4, alpha=0.25)
                b, a = np.polyfit(X, Y, 1)
                xx = np.linspace(np.nanmin(X), np.nanmax(X), 100)
                yy = a + b * xx
                ax.plot(xx, yy, lw=1.5, label=f"fit: y={a:.2f}+{b:.2f}x")
                try:
                    r2 = r2_score(Y, a + b * X)
                    ax.text(0.05, 0.95, f"R²={r2:.2f}", transform=ax.transAxes, va="top")
                except Exception:
                    pass
            ax.set_title(f"Field {L}")
            ax.grid(True)
            ax.set_xlabel("Elevation (m)")
            ax.set_ylabel("Peat depth (model)")
        fig.suptitle("Peat depth vs Elevation")
        plt.tight_layout()
        plt.show()

        return stats


class MultispecAnalysis:
    """
    TVDI analysis using the *previous* method (thermal raster) as the primary path,
    plus the *new* Teros temperature workflow for validation & optional calibration.

    Key features:
    - Previous TVDI: NDVI + Thermal raster with *exact same* date (YYYYMMDD in filenames).
    - New (check only): Grid Teros soil temperatures (daily median of preferred depths)
      to the thermal grid; compare against thermal raster to get bias/RMSE/correlation.
    - Optional temperature correction applied to thermal raster before TVDI:
        * none        : use LST as-is
        * bias        : LST_adj = LST + b  (b from Teros vs LST difference)
        * affine      : LST_adj = a + b * LST  (from robust multi-date fit or per-date)
      You can choose to compute per-date correction (use_teros_for_date=True)
      or a global correction from many dates (pre-compute via calibrate_temperature_raster)
    """

    # ------------------------ construction ------------------------
    def __init__(
        self,
        sample_number=0,
        field_letter="A",
        out_dir="Data/Outputs",
        # NEW: choose where temperature comes from
        temperature_source="teros",  # "teros" | "raster"
        # Teros inputs
        teros_temp_csv=r"D:\Coding\Python\Master-Thesis\Data\Teros Piezo\data-final.csv",
        teros_coord_xlsx=r"Data/Teros Piezo/coordonnees.xlsx",
        teros_depths=(10, 30),
        teros_date_tolerance_days=0,
        teros_use_all_samplers=False,
        temp_interp_mode="linear",  # "linear" | "idw"
        temp_idw_k=12,  # k nearest points for IDW
        temp_idw_power=2.0,  # IDW power
        temp_smooth_sigma_px=2.0,
    ):  # masked Gaussian; 0 disables

        self.sample_number = int(sample_number)
        self.field_letter = str(field_letter).upper()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.temp_interp_mode = str(temp_interp_mode).lower()
        self.temp_idw_k = int(temp_idw_k)
        self.temp_idw_power = float(temp_idw_power)
        self.temp_smooth_sigma_px = float(temp_smooth_sigma_px)

        # keep all raster lists as before (NDVI, bands, thermal for legacy)
        self.raster_paths = {
            "ndvi": _safe_list("NDVI_RASTER", os.path.join("Data", "Multispectral", "NDVI", "*.[tT][iI][fF]")),
            "red": _safe_list("RED_RASTER", os.path.join("Data", "Multispectral", "Red", "*.[tT][iI][fF]")),
            "green": _safe_list("GREEN_RASTER", os.path.join("Data", "Multispectral", "Green", "*.[tT][iI][fF]")),
            "blue": _safe_list("BLUE_RASTER", os.path.join("Data", "Multispectral", "Blue", "*.[tT][iI][fF]")),
            "nir": _safe_list("NIR_RASTER", os.path.join("Data", "Multispectral", "NIR", "*.[tT][iI][fF]")),
            "red_edge": _safe_list(
                "RED_EDGE_RASTER", os.path.join("Data", "Multispectral", "Rededge", "*.[tT][iI][fF]")
            ),
            "temp": _safe_list("TEMP_RASTER", os.path.join("Data", "thermal", "**", "*.[tT][iI][fF]")),
        }

        # NEW
        self.temperature_source = str(temperature_source).lower()

        # caches
        self._poly_xy_cache = None
        self._ref_profile_cache = None

        # Teros config/caches
        self.teros_temp_csv = teros_temp_csv
        self.teros_coord_xlsx = teros_coord_xlsx
        self.teros_depths = tuple(teros_depths)
        self.teros_tol_days = int(teros_date_tolerance_days)
        self.teros_use_all_samplers = bool(teros_use_all_samplers)
        self._teros_cache = None

        self._global_temp_affine = None
        self._global_temp_bias = None

    # ------------------------ utilities ------------------------
    @staticmethod
    def _to_ts(d):
        ts = pd.to_datetime(d) if d is not None else None
        if ts is None:
            return None
        try:
            ts = ts.tz_localize(None)
        except Exception:
            pass
        return ts.normalize()

    @staticmethod
    def _mad_std(a):
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan
        med = np.nanmedian(a)
        mad = np.nanmedian(np.abs(a - med))
        return 1.4826 * mad

    @staticmethod
    def _parse_date_from_name(path_str):
        m = re.search(r"(\d{8})", os.path.basename(path_str))
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").date()
        except Exception:
            return None

    # ------------------------ IO helpers ------------------------
    def _get_reference_profile(self):
        if self._ref_profile_cache is None:
            if len(self.raster_paths["temp"]) > 0:
                ref_path = self.raster_paths["temp"][0]
                with rasterio.open(ref_path) as src:
                    self._ref_profile_cache = src.profile.copy()
            else:
                # fallback: first NDVI as reference
                if len(self.raster_paths["ndvi"]) == 0:
                    raise RuntimeError("No rasters found to build a reference profile.")
                ref_path = self.raster_paths["ndvi"][0]
                with rasterio.open(ref_path) as src:
                    self._ref_profile_cache = src.profile.copy()
        return self._ref_profile_cache

    def read_band(self, raster_path):
        with rasterio.open(raster_path) as src:
            arr = src.read(1).astype(np.float32)
            prof = src.profile.copy()
        return arr, prof

    def _resample_to_profile(self, src_array, src_profile, dst_profile, resampling=Resampling.nearest):
        dst = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=src_array,
            destination=dst,
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            resampling=resampling,
        )
        return dst

    # ------------------------ field mask ------------------------
    def _get_reference_transform(self):
        if self._ref_profile_cache is None:
            if len(self.raster_paths["temp"]) > 0:
                with rasterio.open(self.raster_paths["temp"][0]) as src:
                    self._ref_profile_cache = src.profile.copy()
            else:
                if len(self.raster_paths["ndvi"]) == 0:
                    raise RuntimeError("No rasters found to build a reference profile.")
                with rasterio.open(self.raster_paths["ndvi"][0]) as src:
                    self._ref_profile_cache = src.profile.copy()
        return self._ref_profile_cache

    def _field_polygon_indices(self):
        """
        Your tuned pixel polygon + offsets, in (col=x, row=y).
        """
        if self.field_letter == "A":
            polygon_coords = np.array(
                [[3100, 4800], [4550, 3400], [3550, 2350], [2530, 3400], [3100, 4800]],
                dtype=float,
            )
            x_offset, y_offset = 650, -10
        else:  # Field B
            polygon_coords = np.array(
                [[2530, 3400], [3550, 2350], [3350, 2150], [3550, 1950], [3000, 1000], [1700, 1300], [2530, 3400]],
                dtype=float,
            )
            x_offset, y_offset = 650, -150

        adjusted = polygon_coords.copy()
        adjusted[:, 0] += x_offset
        adjusted[:, 1] += y_offset
        return adjusted

    def _import_pair_for_tvdi_previous(self, sample_number):
        """
        Previous method: pair NDVI with the thermal raster that has the *exact same date*.
        Returns:
            temp_C, ndvi_on_temp_grid, temp_profile, ndvi_profile, temp_path, ndvi_path
        If no thermal match is found, returns (None, None, None, None, None, ndvi_path).
        If NDVI index is out of range, returns (None, None, None, None, None, None).
        """
        # --- NDVI path / bounds ---
        if sample_number < 0 or sample_number >= len(self.raster_paths.get("ndvi", [])):
            print("[TVDI] NDVI index out of range.")
            return (None, None, None, None, None, None)

        ndvi_path = self.raster_paths["ndvi"][sample_number]
        ndvi_date = self._parse_date_from_name(ndvi_path)
        if ndvi_date is None:
            raise ValueError(f"Could not parse date from NDVI filename: {ndvi_path}")

        # --- find thermal with the exact same date ---
        temp_list = self.raster_paths.get("temp", [])
        matches = [p for p in temp_list if self._parse_date_from_name(p) == ndvi_date]
        if not matches:
            print(f"[TVDI] No thermal raster for NDVI date {ndvi_date}.")
            return (None, None, None, None, None, ndvi_path)

        # stable pick in case there are multiple files for that date
        temp_path = sorted(matches)[0]

        # --- read & clean NDVI ---
        ndvi, ndvi_prof = self.read_band(ndvi_path)
        ndvi = ndvi.astype(np.float32, copy=False)
        ndvi[(ndvi < -1.0) | (ndvi > 1.0)] = np.nan

        # --- read thermal & convert to °C if in Kelvin ---
        temp, temp_prof = self.read_band(temp_path)
        temp = temp.astype(np.float32, copy=False)
        if np.isfinite(temp).any() and np.nanmedian(temp) > 150:  # likely Kelvin
            temp -= 273.15
        # sanity clamp
        temp[(temp < -100) | (temp > 80)] = np.nan

        # --- resample NDVI → thermal grid (bilinear) ---
        ndvi_r = self._resample_to_profile(ndvi, ndvi_prof, temp_prof, resampling=Resampling.bilinear)

        return temp, ndvi_r, temp_prof, ndvi_prof, temp_path, ndvi_path

    def apply_field_mask(self, array, profile):
        """
        Rasterize ONE canonical polygon (built once from the reference thermal transform)
        onto the given raster grid.
        """
        if self._poly_xy_cache is None:
            ref_prof = self._get_reference_profile()
            ref_T = ref_prof["transform"]  # <-- Affine
            ij = self._field_polygon_indices()  # (col=x, row=y)
            poly_xy = [rasterio.transform.xy(ref_T, int(r), int(c), offset="center") for c, r in ij]
            self._poly_xy_cache = poly_xy  # list[(X, Y)]

        T = profile["transform"]
        shape = {"type": "Polygon", "coordinates": [self._poly_xy_cache]}
        mask = rasterio.features.rasterize(
            [shape],
            out_shape=array.shape,
            transform=T,
            fill=0,
            default_value=1,
            dtype="uint8",
        )
        out = array.copy()
        out[mask == 0] = np.nan
        return out

    # ------------------------ NDVI/Thermal (previous method) pairing ------------------------
    def _import_pair_for_tvdi(self, sample_number):
        # NDVI
        if sample_number >= len(self.raster_paths["ndvi"]):
            return (None,) * 6
        ndvi_path = self.raster_paths["ndvi"][sample_number]
        ndvi, ndvi_prof = self.read_band(ndvi_path)
        ndvi[(ndvi < -1.0) | (ndvi > 1.0)] = np.nan

        # reference grid/profile (thermal if present; else NDVI)
        ref_prof = self._get_reference_profile()

        # NDVI on reference grid
        ndvi_r = self._resample_to_profile(ndvi.astype(np.float32), ndvi_prof, ref_prof, resampling=Resampling.bilinear)

        # temperature source
        ndvi_date = self._parse_date_from_name(ndvi_path)
        if self.temperature_source == "raster":
            # legacy: pick thermal with same date
            matches = [p for p in self.raster_paths["temp"] if self._parse_date_from_name(p) == ndvi_date]
            if not matches:
                return None, None, None, None, None, ndvi_path
            t_path = matches[0]
            temp, temp_prof = self.read_band(t_path)
            if np.isfinite(temp).any() and np.nanmedian(temp) > 150:  # Kelvin -> °C
                temp = temp - 273.15
            temp[(temp < -100) | (temp > 80)] = np.nan
            # put thermal on the *same* reference grid (usually already the same)
            temp_r = self._resample_to_profile(
                temp.astype(np.float32), temp_prof, ref_prof, resampling=Resampling.bilinear
            )
            return temp_r, ndvi_r, ref_prof, ndvi_prof, t_path, ndvi_path
        else:
            # NEW: Teros soil temperature grid for that day, interpolated onto ref grid
            if ndvi_date is None:
                raise RuntimeError(f"Could not parse date from NDVI name: {ndvi_path}")

            temp_grid = self._soil_temp_grid_for_date(ndvi_date, ref_prof)
            if temp_grid is None:
                return (None,) * 6

            return temp_grid.astype(np.float32), ndvi_r, ref_prof, ndvi_prof, None, ndvi_path

    # ------------------------ Teros (new workflow) for checking/calibration ------------------------
    def _load_teros_cache(self):
        if self._teros_cache is None:
            df = pd.read_csv(self.teros_temp_csv, parse_dates=["Time"])
            df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
            df.columns = [c.strip() for c in df.columns]
            df["date"] = df["Time"].dt.normalize()
            self._teros_cache = df
        return self._teros_cache

    @staticmethod
    def _parse_thousand_num(x):
        return float(str(x).replace(",", "").strip())

    def _pick_depth_columns(self, sampler, df):
        """Return only the configured depths for this sampler (e.g., 10 & 30 cm)."""
        cols = []
        for d in self.teros_depths:
            c = f"{sampler}{int(d)}.Temp"
            if c in df.columns:
                cols.append(c)
        return cols  # if empty, this sampler is skipped for that day

    def _allowed_teros_samplers(self):
        """
        Return sampler bases like 'LS3A', 'LS4B', ...
        If teros_use_all_samplers=True, return *all* samplers found in coords (preferred)
        or, if that fails, infer from CSV headers that have 10/30 cm temps.
        """
        if getattr(self, "teros_use_all_samplers", False):
            # Prefer the coordinates file so we only include samplers with known East/North.
            try:
                coords = pd.read_excel(self.teros_coord_xlsx)
                samps = sorted({str(s).strip() for s in coords["Sampler"].dropna()})
                if samps:
                    return samps
            except Exception:
                pass
            # Fallback: infer from CSV headers
            ter = self._load_teros_cache()
            pat = re.compile(r"^(LS\d+[AB])(10|30)\.Temp$")
            bases = set()
            for c in ter.columns:
                m = pat.match(c)
                if m:
                    bases.add(m.group(1))
            return sorted(bases)

        # Original behaviour (keep field-specific default)
        f = self.field_letter.upper()
        return ["LS3A", "LS3B"] if f == "A" else ["LS4A", "LS4B", "LS5A", "LS5B"]

    def _soil_temp_per_day_for_field(self, target_date: pd.Timestamp):
        ter = self._load_teros_cache()
        t = self._to_ts(target_date)
        days = ter["date"].drop_duplicates().sort_values()
        day = None
        if self.teros_tol_days == 0:
            if (days == t).any():
                day = t
        else:
            if not days.empty:
                idx = np.argmin(np.abs((days - t).values.astype("timedelta64[D]").astype(int)))
                if abs((days.iloc[idx] - t).days) <= self.teros_tol_days:
                    day = days.iloc[idx]
        if day is None:
            return None

        day_df = ter[ter["date"] == day].copy()

        coords = pd.read_excel(self.teros_coord_xlsx)
        coords["North"] = coords["North"].apply(self._parse_thousand_num)
        coords["East"] = coords["East"].apply(self._parse_thousand_num)

        # NEW: restrict to requested samplers for this field
        allowed = set(self._allowed_teros_samplers())
        if not self.teros_use_all_samplers:
            coords = coords[coords["Sampler"].astype(str).str.strip().isin(allowed)].copy()
        else:
            # keep all rows; still nice to warn if coords has none
            if coords.empty:
                raise RuntimeError(f"No Teros coordinates found in {self.teros_coord_xlsx}")

        rows = []
        for _, r in coords.iterrows():
            sampler = str(r["Sampler"]).strip()
            # strictly prefer 10 & 30 cm (fallbacks handled by _pick_depth_columns)
            cols = self._pick_depth_columns(sampler, day_df)
            if not cols:
                continue
            # per time: mean(10cm,30cm), then daily median per sampler
            per_time = day_df[cols].mean(axis=1, skipna=True)
            temp_val = float(np.nanmedian(per_time.values))
            rows.append({"Sampler": sampler, "East": r["East"], "North": r["North"], "Temp": temp_val})

        if not rows:
            return None
        return pd.DataFrame(rows)

    def debug_tvdi_pipeline(self, sample_number=0):
        out = {}
        # NDVI & date
        if sample_number >= len(self.raster_paths["ndvi"]):
            raise RuntimeError("sample_number out of range for NDVI list.")
        ndvi_path = self.raster_paths["ndvi"][sample_number]
        d = self._parse_date_from_name(ndvi_path)
        out["ndvi_path"] = ndvi_path
        out["ndvi_date"] = d

        # CSV + columns
        df = self._load_teros_cache().copy()
        out["csv_rows_total"] = len(df)
        out["csv_has_Time"] = "Time" in df.columns
        day = None
        days = df["date"].drop_duplicates().sort_values()
        if d is not None and (days == pd.Timestamp(d)).any():
            day = pd.Timestamp(d)
        elif self.teros_tol_days > 0 and not days.empty:
            idx = np.argmin(np.abs((days - pd.Timestamp(d)).values.astype("timedelta64[D]").astype(int)))
            day = days.iloc[idx]
        out["matched_day"] = None if day is None else str(day.date())

        if day is not None:
            day_df = df[df["date"] == day]
            out["csv_rows_for_day"] = len(day_df)
            # sensor columns present for this field & depths
            needed = []
            for s in self._allowed_teros_samplers():
                needed += [f"{s}10.Temp", f"{s}30.Temp"]
            present = [c for c in needed if c in day_df.columns]
            missing = sorted(set(needed) - set(present))
            out["needed_cols"] = needed
            out["present_needed_cols"] = present
            out["missing_needed_cols"] = missing

        # coordinates file
        out["coord_xlsx_exists"] = os.path.exists(self.teros_coord_xlsx)

        # points used
        ref_prof = self._get_reference_profile()
        try:
            pts = self._soil_temp_per_day_for_field(pd.Timestamp(d))
            out["n_points"] = None if pts is None else len(pts)
            if pts is not None:
                out["samplers_used"] = pts["Sampler"].tolist()
                out["temp_median_used"] = float(np.nanmedian(pts["Temp"].values))
        except Exception as e:
            out["points_error"] = str(e)

        # grid stats
        try:
            grid = self._soil_temp_grid_for_date(d, ref_prof)
            if grid is None:
                out["grid"] = "None"
            else:
                out["grid"] = {
                    "shape": tuple(grid.shape),
                    "finite_pixels": int(np.isfinite(grid).sum()),
                    "median_temp": float(np.nanmedian(grid)),
                }
        except Exception as e:
            out["grid_error"] = str(e)

        print("=== TVDI DEBUG ===")
        for k, v in out.items():
            print(f"{k}: {v}")
        return out

    def _soil_temp_grid_for_date(self, date_obj, profile):
        """
        Interpolate Teros point temperatures to the provided raster profile grid (°C),
        using tiled evaluation to avoid huge allocations.
        - Linear interpolation where possible (>=3 unique points), filled by nearest.
        - Nearest-only with mild Gaussian smoothing if linear is not feasible.
        - Computes only inside the field's bounding window to reduce work, keeps NaN elsewhere.
        """
        pts = self._soil_temp_per_day_for_field(pd.Timestamp(date_obj))
        if pts is None or pts.empty:
            return None

        T = profile["transform"]
        h, w = profile["height"], profile["width"]

        # map coords → pixel indices (float row/col, then dedup integer pixels)
        rows, cols = rasterio.transform.rowcol(T, pts["East"].values, pts["North"].values)
        rows = np.asarray(rows, dtype=float)
        cols = np.asarray(cols, dtype=float)
        vals = pts["Temp"].values.astype(float)

        # de-duplicate identical pixel locations: median per pixel
        if rows.size == 0:
            return None
        rc_int = np.column_stack([rows.astype(int), cols.astype(int)])
        uniq_keys, inv, first_idx = np.unique(rc_int, axis=0, return_inverse=True, return_index=True)
        if uniq_keys.shape[0] != rows.shape[0]:
            med_vals = np.zeros(uniq_keys.shape[0], dtype=float)
            for k in range(uniq_keys.shape[0]):
                med_vals[k] = np.nanmedian(vals[inv == k])
            rows = rows[first_idx]
            cols = cols[first_idx]
            vals = med_vals

        n_unique = len(vals)

        # compute bounding window of the field polygon on this profile
        r0, c0, r1, c1 = 0, 0, h, w
        try:
            if self._poly_xy_cache is None:
                ref_prof = self._get_reference_profile()
                ref_T = ref_prof["transform"]
                ij = self._field_polygon_indices()
                self._poly_xy_cache = [rasterio.transform.xy(ref_T, int(r), int(c), offset="center") for c, r in ij]
            xs, ys = zip(*self._poly_xy_cache)
            rr, cc = rasterio.transform.rowcol(T, xs, ys)
            pad = 32
            r0 = max(0, int(np.floor(np.nanmin(rr))) - pad)
            r1 = min(h, int(np.ceil(np.nanmax(rr))) + pad)
            c0 = max(0, int(np.floor(np.nanmin(cc))) - pad)
            c1 = min(w, int(np.ceil(np.nanmax(cc))) + pad)
        except Exception:
            pass

        out = np.full((h, w), np.nan, dtype=np.float32)

        # build interpolators
        tree = cKDTree(np.column_stack([cols, rows]))
        interp_lin = None
        if n_unique >= 3 and self.temp_interp_mode == "linear":
            try:
                interp_lin = LinearNDInterpolator(np.column_stack([cols, rows]), vals)
            except QhullError:
                interp_lin = None

        tile = 1024
        for rstart in range(r0, r1, tile):
            rstop = min(r1, rstart + tile)
            for cstart in range(c0, c1, tile):
                cstop = min(c1, cstart + tile)

                gc, gr = np.meshgrid(np.arange(cstart, cstop), np.arange(rstart, rstop), indexing="xy")

                if self.temp_interp_mode == "idw":
                    # --- IDW on this tile (kNN from the KDTree) ---
                    q = np.column_stack([gc.ravel(), gr.ravel()])
                    dist, idx = tree.query(q, k=min(self.temp_idw_k, n_unique))
                    if dist.ndim == 1:  # k=1 safety
                        dist = dist[:, None]
                        idx = idx[:, None]
                    w = 1.0 / np.maximum(dist, 1e-3) ** self.temp_idw_power
                    z = (w * vals[idx]).sum(axis=1) / w.sum(axis=1)
                    block = z.reshape(gr.shape)
                else:
                    # --- linear, filled with nearest where NaN ---
                    block = None
                    if interp_lin is not None:
                        try:
                            block = interp_lin(gc, gr)
                        except Exception:
                            block = None
                    if block is None:
                        q = np.column_stack([gc.ravel(), gr.ravel()])
                        _, idx = tree.query(q, k=1)
                        block = vals[idx].reshape(gr.shape)
                    else:
                        mask = ~np.isfinite(block)
                        if mask.any():
                            q = np.column_stack([gc[mask], gr[mask]])
                            _, idx = tree.query(q, k=1)
                            block = np.asarray(block)
                            block[mask] = vals[idx]

                out[rstart:rstop, cstart:cstop] = block.astype(np.float32)

        # --- masked Gaussian smoothing (optional, works with NaNs) ---
        sig = float(getattr(self, "temp_smooth_sigma_px", 0.0) or 0.0)
        if sig > 0:
            blk = out[r0:r1, c0:c1].copy()
            m = np.isfinite(blk)
            if m.any():
                w = m.astype(np.float32)
                blk[~m] = 0.0
                num = gaussian_filter(blk, sigma=sig, mode="nearest")
                den = gaussian_filter(w, sigma=sig, mode="nearest")
                sm = np.divide(num, np.maximum(den, 1e-6), out=np.full_like(num, np.nan), where=den > 1e-6)
                out[r0:r1, c0:c1] = sm.astype(np.float32)

        out[(out < -40) | (out > 60)] = np.nan
        return out

    # ------------------------ Comparison & calibration ------------------------
    def compare_thermal_vs_teros(self, sample_number=None, plot=True):
        """
        For a date (by NDVI index), load thermal LST (previous method) and Teros soil-temp grid
        and compute diagnostics over the in-field area.
        Returns dict with bias, rmse, slope, intercept, r2, n_pixels, plus arrays (optional).
        """
        if sample_number is None:
            sample_number = self.sample_number

        # Previous method pair
        result = self._import_pair_for_tvdi_previous(sample_number)
        if result[0] is None:
            raise RuntimeError("No thermal raster for this NDVI date.")
        temp, ndvi_r, temp_prof, _ndvi_prof, temp_path, ndvi_path = result

        # Teros grid for the same day (or within tolerance)
        date_obj = self._parse_date_from_name(ndvi_path)
        teros_grid = self._soil_temp_grid_for_date(date_obj, temp_prof)
        if teros_grid is None:
            raise RuntimeError("No Teros temperature available for this date (within tolerance).")

        # in-field only
        temp_f = self.apply_field_mask(temp, temp_prof)
        teros_f = self.apply_field_mask(teros_grid, temp_prof)

        sel = np.isfinite(temp_f) & np.isfinite(teros_f)
        if sel.sum() < 200:
            raise RuntimeError("Too few overlapping pixels for comparison.")

        X = temp_f[sel]  # raster LST (°C)
        Y = teros_f[sel]  # Teros soil temp (°C)

        bias = float(np.nanmean(Y - X))
        rmse = float(np.sqrt(np.nanmean((Y - X) ** 2)))

        # affine fit: Y ≈ a + b X
        b, a = np.polyfit(X, Y, 1)
        r2 = float(r2_score(Y, a + b * X))

        out = {
            "bias": bias,
            "rmse": rmse,
            "slope": float(b),
            "intercept": float(a),
            "r2": r2,
            "n_pixels": int(sel.sum()),
        }

        if plot:
            # Scatter
            plt.figure(figsize=(5, 4), dpi=200)
            plt.scatter(X, Y, s=4, alpha=0.25, label=f"n={sel.sum()}")
            xx = np.linspace(np.nanmin(X), np.nanmax(X), 100)
            plt.plot(xx, a + b * xx, label=f"Y≈{a:.2f}+{b:.2f}X\nR²={r2:.2f}")
            plt.plot(xx, xx, "--", lw=1, label="1:1")
            plt.xlabel("Thermal raster LST (°C)")
            plt.ylabel("Teros soil temp (°C)")
            ds = date_obj.strftime("%d/%m/%Y") if date_obj else "unknown date"
            plt.title(f"LST vs Teros • Field {self.field_letter} • {ds}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

            # Quick maps
            for arr, name, vmin, vmax in [
                (temp_f, "LST (°C)", None, None),
                (teros_f, "Teros soil temp (°C)", None, None),
                (teros_f - temp_f, "Teros - LST (°C)", -5, 5),
            ]:
                fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
                extent = plotting_extent(arr, temp_prof["transform"])
                im = ax.imshow(arr, origin="upper", extent=extent, vmin=vmin, vmax=vmax)
                ax.set_aspect("equal")
                cb = fig.colorbar(im, ax=ax)
                cb.set_label(name)
                ax.set_title(f"{name} • {ds}")
                plt.tight_layout()
                plt.show()

        return out

    def calibrate_temperature_raster(
        self,
        dates_limit=None,
        min_pixels=200,
        robust="median",  # "median" | "mean"
        verbose=True,
    ):
        """
        Build a *global* affine correction LST_adj = a + b*LST using all dates that have:
        - NDVI raster
        - exact-date thermal raster
        - Teros soil-temp available within tolerance

        Also computes a global bias-only correction as the robust central tendency
        of per-date pixelwise (Teros - LST).
        """
        records = []
        n = len(self.raster_paths["ndvi"])
        for i in range(n):
            try:
                temp, ndvi_r, temp_prof, _ndvi_prof, _tpath, ndvi_path = self._import_pair_for_tvdi_previous(i)
                if temp_prof is None:
                    continue
                d = self._parse_date_from_name(ndvi_path)
                if dates_limit is not None and callable(dates_limit):
                    if not dates_limit(d):
                        continue

                teros_grid = self._soil_temp_grid_for_date(d, temp_prof)
                if teros_grid is None:
                    continue

                temp_f = self.apply_field_mask(temp, temp_prof)
                teros_f = self.apply_field_mask(teros_grid, temp_prof)
                sel = np.isfinite(temp_f) & np.isfinite(teros_f)
                if sel.sum() < min_pixels:
                    continue

                X = temp_f[sel]  # LST
                Y = teros_f[sel]  # Teros
                b, a = np.polyfit(X, Y, 1)  # Y ≈ a + b*X
                bias = float(np.nanmedian(Y - X))  # bias-only per date
                records.append({"date": d, "a": a, "b": b, "bias": bias, "n": int(sel.sum())})
            except Exception:
                continue

        if not records:
            raise RuntimeError("No suitable dates to calibrate a global temperature correction.")

        df = pd.DataFrame(records).sort_values("date")
        if robust == "median":
            a_glob = float(df["a"].median())
            b_glob = float(df["b"].median())
            bias_glob = float(df["bias"].median())
        else:
            a_glob = float(df["a"].mean())
            b_glob = float(df["b"].mean())
            bias_glob = float(df["bias"].mean())

        self._global_temp_affine = {"a": a_glob, "b": b_glob}
        self._global_temp_bias = bias_glob

        if verbose:
            print(
                f"[Calibration] Global affine LST_adj = {a_glob:.3f} + {b_glob:.3f} * LST  "
                f"(from {len(df)} dates); global bias = {bias_glob:+.3f} °C"
            )

        return {"affine": self._global_temp_affine, "bias": self._global_temp_bias, "per_date": df}

    # ------------------------ temperature correction application ------------------------
    @staticmethod
    def _apply_temp_correction_array(temp, mode, a=None, b=None, bias=None):
        if mode is None or mode == "none":
            return temp
        if mode == "bias":
            bb = 0.0 if bias is None else float(bias)
            return temp + bb
        if mode == "affine":
            aa = 0.0 if a is None else float(a)
            bb = 1.0 if b is None else float(b)
            return aa + bb * temp
        raise ValueError("temp_correction must be one of: 'none','bias','affine'.")

    # ------------------------ TVDI core ------------------------
    def _compute_tvdi_from_arrays(
        self,
        temp,
        ndvi_r,
        ndvi_bin_width=0.01,
        pct_wet=5,
        pct_dry=95,
        min_pixels_per_bin=50,
        binning="equal_count",  # "equal_count" | "fixed_width"
        target_bins=20,
        wet_edge="linear",  # "linear" | "constant_mean"
        water_mask=None,
        shadow_mask=None,
        aggregate_factor=None,
    ):
        # 1) pre-mask (optionally water/shadow) for edge estimation
        base_mask = np.isfinite(temp) & np.isfinite(ndvi_r)
        if water_mask is not None:
            base_mask &= ~water_mask
        if shadow_mask is not None:
            base_mask &= ~shadow_mask
        if base_mask.sum() < 1000:
            raise RuntimeError("Too few valid pixels (after masks) to build TVDI.")

        ts = np.where(base_mask, temp, np.nan)
        vi = np.where(base_mask, ndvi_r, np.nan)

        if aggregate_factor and aggregate_factor > 1:

            def _block_reduce(arr, factor, func=np.nanmedian):
                h, w = arr.shape
                nh, nw = h // factor, w // factor
                if nh == 0 or nw == 0:
                    return arr
                cropped = arr[: nh * factor, : nw * factor]
                tmp = func(cropped.reshape(nh, factor, nw, factor), axis=3)
                return func(tmp, axis=1)

            ts_use = _block_reduce(ts, aggregate_factor)
            vi_use = _block_reduce(vi, aggregate_factor)
        else:
            ts_use, vi_use = ts, vi

        sel = np.isfinite(ts_use) & np.isfinite(vi_use)
        ts_u = ts_use[sel].astype(np.float32)
        vi_u = vi_use[sel].astype(np.float32)

        centers, wet_vals, dry_vals, counts = [], [], [], []

        # 2) binning
        if binning == "equal_count":
            order = np.argsort(vi_u)
            vi_s = vi_u[order]
            ts_s = ts_u[order]
            bins_idx = np.array_split(np.arange(vi_s.size), max(1, target_bins))
            merged, cur = [], []
            for b_ in bins_idx:
                cur.extend(b_.tolist())
                if len(cur) >= min_pixels_per_bin:
                    merged.append(np.array(cur, dtype=int))
                    cur = []
            if len(cur) >= min_pixels_per_bin // 2:
                merged.append(np.array(cur, dtype=int))
            for ind in merged:
                if ind.size < min_pixels_per_bin:
                    continue
                vbin = vi_s[ind]
                tbin = ts_s[ind]
                centers.append(np.nanmedian(vbin))
                counts.append(ind.size)
                wet_vals.append(np.nanpercentile(tbin, pct_wet))
                dry_vals.append(np.nanpercentile(tbin, pct_dry))
        else:
            vi_min, vi_max = np.nanmin(vi_u), np.nanmax(vi_u)
            edges = np.arange(
                np.floor(vi_min / ndvi_bin_width) * ndvi_bin_width,
                np.ceil(vi_max / ndvi_bin_width) * ndvi_bin_width + ndvi_bin_width,
                ndvi_bin_width,
            )
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                selb = (vi_u >= lo) & (vi_u < hi)
                if selb.sum() >= min_pixels_per_bin:
                    centers.append((lo + hi) / 2.0)
                    counts.append(selb.sum())
                    wet_vals.append(np.nanpercentile(ts_u[selb], pct_wet))
                    dry_vals.append(np.nanpercentile(ts_u[selb], pct_dry))

        centers = np.asarray(centers, np.float32)
        wet_vals = np.asarray(wet_vals, np.float32)
        dry_vals = np.asarray(dry_vals, np.float32)
        counts = np.asarray(counts, np.float32)

        if centers.size < 3:
            raise RuntimeError("Not enough NDVI bins to fit edges.")

        # 3) edge fits
        W = np.sqrt(counts / (counts.max() + 1e-9))
        a_dry, b_dry = np.polyfit(centers, dry_vals, 1, w=W)

        if wet_edge == "constant_mean":
            tsmin_scalar = float(np.nanmean(wet_vals))
            tsmin = np.full_like(ndvi_r, tsmin_scalar, dtype=np.float32)
            a_wet, b_wet = 0.0, tsmin_scalar
        else:
            a_wet, b_wet = np.polyfit(centers, wet_vals, 1, w=W)
            tsmin = a_wet * ndvi_r + b_wet

        tsmax = a_dry * ndvi_r + b_dry
        den = tsmax - tsmin
        den = np.where(np.abs(den) < 1e-6, np.nan, den)

        tvdi = (temp - tsmin) / den
        tvdi[~np.isfinite(tvdi)] = np.nan
        tvdi = np.clip(tvdi, 0, 1).astype(np.float32)

        return tvdi, (centers, (a_wet, b_wet), (a_dry, b_dry))

    # ------------------------ masks (optional for edges) ------------------------
    def _import_green_nir_pair_by_date(self, sample_number, anchor="green"):
        if anchor == "green":
            g_path = self.raster_paths["green"][sample_number]
            g_date = self._parse_date_from_name(g_path)
            if g_date is None:
                return (None,) * 6
            n_matches = [p for p in self.raster_paths["nir"] if self._parse_date_from_name(p) == g_date]
            if not n_matches:
                return (None,) * 6
            n_path = n_matches[0]
        else:
            n_path = self.raster_paths["nir"][sample_number]
            n_date = self._parse_date_from_name(n_path)
            if n_date is None:
                return (None,) * 6
            g_matches = [p for p in self.raster_paths["green"] if self._parse_date_from_name(p) == n_date]
            if not g_matches:
                return (None,) * 6
            g_path = g_matches[0]
        g, gprof = self.read_band(g_path)
        n, nprof = self.read_band(n_path)
        return g, gprof, n, nprof, g_path, n_path

    def _make_masks_for_date(self, sample_number, temp_prof, shadow_pct=5, water_ndwi_thresh=-0.10):
        gpn = self._import_green_nir_pair_by_date(sample_number, anchor="nir")
        masks = {"water": None, "shadow": None}
        if gpn[0] is not None:
            g, gprof, n, _nprof, _gpath, _npath = gpn
            ndwi = -(g - n) / (g + n)
            ndwi[(ndwi < -1) | (ndwi > 1)] = np.nan
            ndwi_r = self._resample_to_profile(ndwi.astype(np.float32), gprof, temp_prof, resampling=Resampling.nearest)
            nir_r = self._resample_to_profile(n.astype(np.float32), gprof, temp_prof, resampling=Resampling.nearest)
            masks["water"] = ndwi_r < water_ndwi_thresh
            try:
                p = np.nanpercentile(nir_r, shadow_pct)
                masks["shadow"] = nir_r <= p
            except Exception:
                masks["shadow"] = None
        return masks

    # ------------------------ Public: compute TVDI (previous method + optional correction) ------------------------
    def calculate_tvdi(
        self,
        sample_number=None,
        save=True,
        plot=True,
        ndvi_bin_width="auto",
        pct_wet=5,
        pct_dry=95,
        min_pixels_per_bin=50,
        binning="equal_count",
        target_bins=20,
        wet_edge="linear",
        use_water_mask=True,
        use_shadow_mask=True,
        aggregate_factor=None,
        shadow_pct=5,
        water_ndwi_thresh=-0.10,
        # temperature correction controls
        temp_correction="none",
        use_teros_for_date=False,
        save_corrected_temp=False,
        field_letter=None,
    ):
        _old_field, _old_poly = self.field_letter, self._poly_xy_cache
        if field_letter is not None:
            fl = str(field_letter).upper()
            if fl not in ("A", "B"):
                raise ValueError("field_letter must be 'A' or 'B'")
            if fl != self.field_letter:
                self.field_letter = fl
                self._poly_xy_cache = None
        try:
            if sample_number is None:
                sample_number = self.sample_number

            # unified importer
            result = self._import_pair_for_tvdi(sample_number)
            if result[0] is None:
                return None, None
            temp, ndvi_r, ref_prof, _ndvi_prof, _tpath, npath = result

            # in-field mask
            temp_f = self.apply_field_mask(temp, ref_prof)
            ndvi_f = self.apply_field_mask(ndvi_r, ref_prof)

            # temperature correction: skip for Teros (not meaningful)
            if self.temperature_source == "raster" and temp_correction != "none":
                a = b = bias = None
                if use_teros_for_date:
                    date_obj = self._parse_date_from_name(npath)
                    teros_grid = self._soil_temp_grid_for_date(date_obj, ref_prof)
                    if teros_grid is None:
                        raise RuntimeError("No Teros temperature for this date to compute correction.")
                    teros_f = self.apply_field_mask(teros_grid, ref_prof)
                    sel = np.isfinite(temp_f) & np.isfinite(teros_f)
                    if sel.sum() < 200:
                        raise RuntimeError("Too few overlapping pixels for per-date correction.")
                    X, Y = temp_f[sel], teros_f[sel]
                    bb, aa = np.polyfit(X, Y, 1)
                    bias = float(np.nanmean(Y - X))
                    a, b = float(aa), float(bb)
                else:
                    if temp_correction == "bias" and self._global_temp_bias is not None:
                        bias = float(self._global_temp_bias)
                    elif temp_correction == "affine" and self._global_temp_affine is not None:
                        a = float(self._global_temp_affine["a"])
                        b = float(self._global_temp_affine["b"])
                    else:
                        raise RuntimeError("Requested temp correction but no global calibration set.")
                temp_adj = self._apply_temp_correction_array(temp, temp_correction, a=a, b=b, bias=bias)
                temp_f = self.apply_field_mask(temp_adj, ref_prof)
                if save_corrected_temp:
                    date_obj = self._parse_date_from_name(npath)
                    date_str = date_obj.strftime("%Y%m%d") if date_obj else "unknown"
                    outp = self.out_dir / f"LST_corrected_Field{self.field_letter}_{date_str}.tif"
                    prof = ref_prof.copy()
                    prof.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
                    with rasterio.open(outp, "w", **prof) as dst:
                        dst.write(temp_adj.astype(np.float32), 1)

            # --- light vegetation mask to avoid roads/shadows/bare soil in edges & TVDI
            ndvi_min, ndvi_max = 0.15, 0.95  # tweak if needed
            ndvi_f = ndvi_f.copy()
            ndvi_f[(ndvi_f < ndvi_min) | (ndvi_f > ndvi_max)] = np.nan

            valid = np.isfinite(temp_f) & np.isfinite(ndvi_f)
            if valid.sum() < 1000:
                raise RuntimeError("Too few valid in-field pixels to build TVDI.")

            # --- build optional masks (on the same reference grid) ---
            water_mask = shadow_mask = None
            if use_water_mask or use_shadow_mask:
                try:
                    masks = self._make_masks_for_date(
                        sample_number,
                        ref_prof,
                        shadow_pct=shadow_pct,
                        water_ndwi_thresh=water_ndwi_thresh,
                    )
                except Exception:
                    masks = {"water": None, "shadow": None}
                if use_water_mask:
                    water_mask = masks.get("water", None)
                if use_shadow_mask:
                    shadow_mask = masks.get("shadow", None)

            # --- effective NDVI bin width for fixed-width mode ---
            if isinstance(ndvi_bin_width, str) and ndvi_bin_width.lower() == "auto":
                vi = ndvi_f[np.isfinite(ndvi_f)]
                if vi.size >= 10:
                    # robust span (avoid outliers)
                    vi_min, vi_max = np.nanpercentile(vi, 1), np.nanpercentile(vi, 99)
                else:
                    vi_min, vi_max = np.nanmin(vi), np.nanmax(vi)
                span = float(vi_max - vi_min)
                eff_width = max(0.005, span / max(1, int(target_bins)))
            else:
                eff_width = float(ndvi_bin_width)

            # Optional water/shadow masks built from Green+NIR of the same date
            water_mask = shadow_mask = None
            if use_water_mask or use_shadow_mask:
                _m = self._make_masks_for_date(
                    sample_number, ref_prof, shadow_pct=shadow_pct, water_ndwi_thresh=water_ndwi_thresh
                )
                water_mask = _m["water"] if use_water_mask else None
                shadow_mask = _m["shadow"] if use_shadow_mask else None

            # (masks, auto bin width) — unchanged ...
            tvdi, edges = self._compute_tvdi_from_arrays(
                temp=temp_f,
                ndvi_r=ndvi_f,  # unchanged downstream
                ndvi_bin_width=(eff_width if binning == "fixed_width" else 0.01),
                pct_wet=pct_wet,
                pct_dry=pct_dry,
                min_pixels_per_bin=min_pixels_per_bin,
                binning=binning,
                target_bins=target_bins,
                wet_edge=wet_edge,
                water_mask=water_mask,
                shadow_mask=shadow_mask,
                aggregate_factor=aggregate_factor,
            )

            # save/plot — unchanged, but tweak Y label depending on source
            date_obj = self._parse_date_from_name(npath)
            date_str = date_obj.strftime("%Y%m%d") if date_obj else "unknown"
            out_path = self.out_dir / f"TVDI_Field{self.field_letter}_{date_str}.tif"

            if save:
                prof = ref_prof.copy()
                prof.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
                with rasterio.open(out_path, "w", **prof) as dst:
                    dst.write(tvdi.astype(np.float32), 1)

            if plot:
                (centers, (a_wet, b_wet), (a_dry, b_dry)) = edges
                ts = temp_f[valid]
                vi = ndvi_f[valid]

                plt.figure(figsize=(5, 4), dpi=200)
                idx = np.random.choice(ts.size, size=min(5000, ts.size), replace=False)

                # samples (keep bluish points)
                plt.scatter(vi[idx], ts[idx], s=6, alpha=0.25, color="tab:blue", linewidths=0)

                x_line = np.linspace(np.nanmin(centers), np.nanmax(centers), 100)
                # dry edge = orange
                plt.plot(x_line, a_dry * x_line + b_dry, lw=2.5, color="tab:orange", label=f"Dry edge ({pct_dry}th)")
                # wet edge = light blue
                if wet_edge == "constant_mean":
                    plt.axhline(b_wet, lw=2.5, color="#9ecae1", label=f"Wet edge (flat, {pct_wet}th)")
                else:
                    plt.plot(x_line, a_wet * x_line + b_wet, lw=2.5, color="#9ecae1", label=f"Wet edge ({pct_wet}th)")

                ttl_date = date_obj.strftime("%d/%m/%Y") if date_obj else "unknown date"
                ylab = "Soil Temperature [°C]" if self.temperature_source == "teros" else "LST [°C]"
                plt.xlabel("NDVI [-]")
                plt.ylabel(ylab)
                plt.title(f"NDVI–Temperature Triangle • Field {self.field_letter} • {ttl_date}")
                plt.legend()
                plt.tight_layout()
                plt.show()

                self._render_geo_map_like_tvdi(
                    array=tvdi,
                    profile=ref_prof,
                    title_text=f"TVDI [-] • Field {self.field_letter} • {ttl_date}",
                    cbar_label="TVDI [-]",
                    use_mask=False,
                    fill_outline_holes=True,  # <— only here
                )

            return tvdi, (str(out_path) if save else None)
        finally:
            self.field_letter, self._poly_xy_cache = _old_field, _old_poly

    # ------------------------ Convenience plots ------------------------
    def extract_dates(self, raster_type):
        out = []
        for p in self.raster_paths[raster_type]:
            d = self._parse_date_from_name(p)
            out.append(d.strftime("%d/%m/%Y") if d else "unknown date")
        return out

    def calculate_ndwi(self, sample_number):
        g_path = self.raster_paths["green"][sample_number]
        n_path = self.raster_paths["nir"][sample_number]
        g, gprof = self.read_band(g_path)
        n, _ = self.read_band(n_path)
        ndwi = -(g - n) / (g + n)
        ndwi[(ndwi < -1) | (ndwi > 1) | (ndwi == 0)] = np.nan
        return ndwi

    # ---------- pretty map helpers (identical look to your old maps) ----------
    def _auto_zoom_to_valid_coords(self, ax, array, transform, pad_pixels=20):
        yy, xx = np.where(np.isfinite(array))
        if yy.size == 0:
            return
        r0, r1 = int(yy.min()), int(yy.max())
        c0, c1 = int(xx.min()), int(xx.max())
        x_ul, y_ul = rasterio.transform.xy(transform, r0, c0, offset="ul")
        x_lr, y_lr = rasterio.transform.xy(transform, r1, c1, offset="lr")
        px_w = transform.a
        px_h = -transform.e if transform.e < 0 else transform.e
        pad_x = pad_pixels * abs(px_w)
        pad_y = pad_pixels * abs(px_h)
        xmin, xmax = min(x_ul, x_lr) - pad_x, max(x_ul, x_lr) + pad_x
        ymin, ymax = min(y_ul, y_lr) - pad_y, max(y_ul, y_lr) + pad_y
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    def _apply_lambert72_grid(self, ax, major=100, minor=50, labelsize=4):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_autoscale_on(False)

        def _ticks_within(lo, hi, step):
            if step is None or step <= 0:
                return np.array([])
            start = np.ceil(min(lo, hi) / step) * step
            stop = np.floor(max(lo, hi) / step) * step
            if stop < start:
                return np.array([])
            return np.arange(start, stop + 0.5 * step, step)

        xt_major = _ticks_within(*xlim, major)
        yt_major = _ticks_within(*ylim, major)
        ax.set_xticks(xt_major)
        ax.set_yticks(yt_major)
        if minor and minor > 0:
            xt_minor = _ticks_within(*xlim, minor)
            yt_minor = _ticks_within(*ylim, minor)
            ax.set_xticks(xt_minor, minor=True)
            ax.set_yticks(yt_minor, minor=True)

        ax.grid(which="major", linewidth=0.5, alpha=0.35)
        ax.grid(which="minor", linewidth=0.4, alpha=0.15)
        ax.tick_params(top=True, right=True, direction="out", length=3, labelsize=labelsize)
        ax.tick_params(which="minor", length=2)

        from matplotlib.ticker import ScalarFormatter

        fmtx = ScalarFormatter(useMathText=False)
        fmty = ScalarFormatter(useMathText=False)
        fmtx.set_scientific(False)
        fmty.set_scientific(False)
        fmtx.set_useOffset(False)
        fmty.set_useOffset(False)
        ax.xaxis.set_major_formatter(fmtx)
        ax.yaxis.set_major_formatter(fmty)
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.margins(0)

    def _add_north_arrow(self, ax, loc="NE", size=0.04, color="k", style="simple", inset=(0.05, 0.01), font_size=9):
        corners = {"NE": (1.0, 1.0), "NW": (0.0, 1.0), "SE": (1.0, 0.0), "SW": (0.0, 0.0)}
        cx, cy = corners.get(loc, corners["NE"])
        x = cx - inset[0] if cx == 1.0 else cx + inset[0]
        y_top = cy - inset[1] if cy == 1.0 else cy + inset[1]
        y_bot = y_top - size if cy == 1.0 else y_top + size
        arrow_kwargs = dict(
            facecolor=color, edgecolor=color, width=1.0, headwidth=8, headlength=10, shrinkA=0, shrinkB=0
        )
        ax.annotate(
            "N",
            xy=(x, y_top),
            xytext=(x, y_bot),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            ha="center",
            va="center",
            color=color,
            fontsize=font_size,
            arrowprops=arrow_kwargs,
            annotation_clip=False,
        )

    def _render_geo_map_like_tvdi(
        self,
        array,
        profile,
        title_text,
        cbar_label="Value",
        cmap="viridis",
        vmin=None,
        vmax=None,
        use_mask=True,
        fill_outline_holes=False,  # <— NEW (default False so others unchanged)
    ):
        arr = self.apply_field_mask(array, profile) if use_mask else array.copy()
        fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=300)

        # leave room for a separate colorbar axis
        fig.subplots_adjust(bottom=0.10, top=0.92, right=0.95)

        # --- enforce 0..1 + yellow→blue for NDVI/NDWI/TVDI ---
        label_l = (cbar_label or "").lower()
        if any(k in label_l for k in ("ndvi", "ndwi", "tvdi")):
            vmin, vmax = 0.0, 1.0  # fixed scale
            # low=yellow, high=blue
            cmap = "YlGnBu"

        transform = profile["transform"]
        extent = plotting_extent(arr, transform)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", extent=extent)

        # --- outline (optionally hole-filled) ---
        from scipy.ndimage import binary_fill_holes

        mask_for_outline = np.isfinite(arr)
        if fill_outline_holes:
            mask_for_outline = binary_fill_holes(mask_for_outline)

        ax.contour(
            mask_for_outline.astype(int),
            levels=[0.5],
            colors="k",
            linewidths=1.5,
            origin="upper",
            extent=extent,
        )

        self._auto_zoom_to_valid_coords(ax, arr, transform, pad_pixels=30)
        ax.set_aspect("equal", adjustable="box")

        # finite_mask = np.isfinite(arr).astype(int)
        # ax.contour(finite_mask, levels=[0.5], colors="k", linewidths=1.5, origin="upper", extent=extent)
        self._add_north_arrow(ax, loc="NE", size=0.09, inset=(0.03, 0.01), font_size=9)
        self._apply_lambert72_grid(ax, major=100, minor=50, labelsize=7)
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)

        # colorbar axis that matches the map height exactly
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.10)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label)

        ax.set_title(title_text, pad=10)
        plt.show()

    # ---------- memory-safe plot_rasters (brings back your old entry point) ----------
    def plot_rasters(
        self,
        sample_number=0,
        ndvi=False,
        red=False,
        green=False,
        blue=False,
        nir=False,
        red_edge=False,
        ndwi=False,
        tvdi=False,
        downscale=8,  # adjust if you need smaller/faster previews
        apply_mask=True,
        field_letter=None,  # <— NEW
    ):
        # --- minimal field override (with polygon-cache safety) ---
        _old_field = self.field_letter
        _old_poly = self._poly_xy_cache
        if field_letter is not None:
            fl = str(field_letter).upper()
            if fl not in ("A", "B"):
                raise ValueError("field_letter must be 'A' or 'B'")
            if fl != self.field_letter:
                self.field_letter = fl
                self._poly_xy_cache = None
        try:

            def _read_downscaled(path):
                with rasterio.open(path) as src:
                    out_h = max(1, src.height // downscale)
                    out_w = max(1, src.width // downscale)
                    arr = src.read(1, out_shape=(1, out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
                    prof = src.profile.copy()
                    prof.update(height=out_h, width=out_w)
                    prof["transform"] = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
                return arr, prof

            # Use NDVI↔Thermal pairing so dates line up with TVDI
            result = self._import_pair_for_tvdi_previous(sample_number)
            has_thermal = result[2] is not None
            date_obj = self._parse_date_from_name(result[5]) if result[5] else None
            date_str = date_obj.strftime("%d/%m/%Y") if date_obj else "unknown date"

            layers = []

            # NDVI
            if ndvi and len(self.raster_paths["ndvi"]) > sample_number:
                if has_thermal and result[1] is not None:
                    ndvi_on_temp = result[1].astype(np.float32)
                    ndvi_on_temp[(ndvi_on_temp < -1) | (ndvi_on_temp > 1)] = np.nan
                    prof = result[2]
                    arr = self.apply_field_mask(ndvi_on_temp, prof) if apply_mask else ndvi_on_temp
                    layers.append((f"NDVI [-] • {date_str}", arr, prof))
                else:
                    arr, prof = _read_downscaled(self.raster_paths["ndvi"][sample_number])
                    arr[(arr < -1) | (arr > 1)] = np.nan
                    if apply_mask:
                        arr = self.apply_field_mask(arr, prof)
                    layers.append((f"NDVI [-] • {date_str}", arr, prof))

            # Single bands
            for label, key, flag in [
                ("Red Band", "red", red),
                ("Green Band", "green", green),
                ("Blue Band", "blue", blue),
                ("NIR Band", "nir", nir),
                ("Red Edge Band", "red_edge", red_edge),
            ]:
                if flag and len(self.raster_paths[key]) > sample_number:
                    arr, prof = _read_downscaled(self.raster_paths[key][sample_number])
                    arr[arr < 0] = np.nan
                    if apply_mask:
                        arr = self.apply_field_mask(arr, prof)
                    layers.append((label, arr, prof))

            # NDWI (green↔nir exact-date)
            if ndwi:
                gpn = self._import_green_nir_pair_by_date(sample_number, anchor="green")
                if gpn[0] is not None:
                    g_path, n_path = gpn[4], gpn[5]
                    g_arr, g_prof = _read_downscaled(g_path)
                    n_arr, n_prof = _read_downscaled(n_path)
                    n_on_g = self._resample_to_profile(n_arr, n_prof, g_prof, resampling=Resampling.bilinear)
                    ndwi_arr = -(g_arr - n_on_g) / (g_arr + n_on_g)
                    ndwi_arr[(ndwi_arr < -1) | (ndwi_arr > 1) | (ndwi_arr == 0)] = np.nan
                    if apply_mask:
                        ndwi_arr = self.apply_field_mask(ndwi_arr, g_prof)
                    layers.append((f"NDWI [-] • {date_str}", ndwi_arr, g_prof))
                else:
                    print("[NDWI] Skipping — no matching Green/NIR pair for this index.")

            # Quick-look TVDI on downscaled grid
            if tvdi and has_thermal:
                temp_full, ndvi_full, temp_prof_full, _, _, _ = result

                def _block_reduce(arr, factor, func=np.nanmedian):
                    h, w = arr.shape
                    f = max(2, int(factor))
                    nh, nw = max(1, h // f), max(1, w // f)
                    arr = arr[: nh * f, : nw * f]
                    tmp = func(arr.reshape(nh, f, nw, f), axis=3)
                    return func(tmp, axis=1)

                f = max(2, downscale)
                temp_ds = _block_reduce(temp_full.astype(np.float32), f)
                ndvi_ds = _block_reduce(ndvi_full.astype(np.float32), f)

                T = temp_prof_full["transform"]
                new_transform = Affine(T.a * f, T.b, T.c, T.d, T.e * f, T.f)
                temp_prof_ds = temp_prof_full.copy()
                temp_prof_ds.update(height=temp_ds.shape[0], width=temp_ds.shape[1], transform=new_transform)

                temp_f = self.apply_field_mask(temp_ds, temp_prof_ds) if apply_mask else temp_ds
                ndvi_f = self.apply_field_mask(ndvi_ds, temp_prof_ds) if apply_mask else ndvi_ds

                tvdi_arr, _ = self._compute_tvdi_from_arrays(
                    temp=temp_f, ndvi_r=ndvi_f, binning="equal_count", target_bins=20, min_pixels_per_bin=25
                )
                layers.append((f"TVDI [-] (quick-look) • {date_str}", tvdi_arr, temp_prof_ds))
            elif tvdi and not has_thermal:
                print("[TVDI] Skipping — no matching thermal raster for this NDVI date.")

            # Plot each layer
            for title, arr, prof in layers:
                self._render_geo_map_like_tvdi(
                    array=arr,
                    profile=prof,
                    title_text=f"{title} • Field {self.field_letter}",
                    cbar_label=title.split(" (")[0],
                )
        finally:
            # restore original field + polygon cache
            self.field_letter = _old_field
            self._poly_xy_cache = _old_poly

    # ---------- Evolution plots + correlations (old feature, adapted) ----------
    def plot_indices_evolution(
        self,
        plot=True,
        include_teros=True,
        include_gpr=True,
        teros_csv="Data/Teros Piezo/teros_piezo.csv",
        teros_daily_agg="median",
        teros_sensor_agg="median",
        start="2023-04-15",
        end="2024-02-15",
        field_letter=None,  # <— NEW
    ):
        # --- minimal field override (with polygon-cache safety) ---
        _old_field = self.field_letter
        _old_poly = self._poly_xy_cache
        if field_letter is not None:
            fl = str(field_letter).upper()
            if fl not in ("A", "B"):
                raise ValueError("field_letter must be 'A' or 'B'")
            if fl != self.field_letter:
                self.field_letter = fl
                self._poly_xy_cache = None
        try:

            def _sort_by_date(dates, vals1, vals2):
                filt = [(i, self._to_ts(d)) for i, d in enumerate(dates) if d is not None]
                if not filt:
                    return [], [], []
                idx, dts = zip(*filt)
                order = np.argsort(np.array(dts, dtype="datetime64[ns]"))
                return [dts[i] for i in order], [vals1[idx[i]] for i in order], [vals2[idx[i]] for i in order]

            def _clip_range(dates, vals1, vals2, lo, hi):
                out_d, out_v1, out_v2 = [], [], []
                for d, v1, v2 in zip(dates, vals1, vals2):
                    if d is not None and lo <= d <= hi:
                        out_d.append(d)
                        out_v1.append(v1)
                        out_v2.append(v2)
                return out_d, out_v1, out_v2

            # normalize window
            start_ts = self._to_ts(pd.to_datetime(start))
            end_ts = self._to_ts(pd.to_datetime(end))

            # ---------- NDVI ----------
            ndvi_dates, ndvi_med, ndvi_sig = [], [], []
            for ndvi_path in self.raster_paths["ndvi"]:
                ndvi, prof = self.read_band(ndvi_path)
                ndvi[(ndvi < -1.0) | (ndvi > 1.0)] = np.nan
                ndvi_m = self.apply_field_mask(ndvi, prof)
                ndvi_dates.append(self._to_ts(self._parse_date_from_name(ndvi_path)))
                ndvi_med.append(np.nanmedian(ndvi_m))
                ndvi_sig.append(self._mad_std(ndvi_m))

            # ---------- NDWI ----------
            ndwi_dates, ndwi_med, ndwi_sig = [], [], []
            green_by_date = {
                self._to_ts(self._parse_date_from_name(p)): p
                for p in self.raster_paths["green"]
                if self._parse_date_from_name(p) is not None
            }
            nir_by_date = {
                self._to_ts(self._parse_date_from_name(p)): p
                for p in self.raster_paths["nir"]
                if self._parse_date_from_name(p) is not None
            }
            common_dates = sorted(set(green_by_date) & set(nir_by_date))
            for d in common_dates:
                g_path, n_path = green_by_date[d], nir_by_date[d]
                g, gprof = self.read_band(g_path)
                n, _ = self.read_band(n_path)
                ndwi = -(g - n) / (g + n)
                ndwi[(ndwi < -1) | (ndwi > 1) | (ndwi == 0)] = np.nan
                ndwi_m = self.apply_field_mask(ndwi, gprof)
                ndwi_dates.append(d)
                ndwi_med.append(np.nanmedian(ndwi_m))
                ndwi_sig.append(self._mad_std(ndwi_m))

            # --- TVDI (delegate to the same pipeline used elsewhere) ---
            tvdi_dates, tvdi_med, tvdi_sig = [], [], []
            for i in range(len(self.raster_paths["ndvi"])):
                try:
                    # compute TVDI exactly like the per-date tool (no saving, no plotting)
                    tvdi_arr, _ = self.calculate_tvdi(
                        sample_number=i,
                        save=False,
                        plot=False,
                        # keep defaults so we match your main pipeline:
                        binning="equal_count",  # with safe fallback inside calculate_tvdi
                        target_bins=20,
                        min_pixels_per_bin=50,
                        use_water_mask=True,
                        use_shadow_mask=True,
                        aggregate_factor=2,
                    )
                    if tvdi_arr is None:
                        continue
                except Exception:
                    # e.g., no temp for that date or too few pixels
                    continue

                # use the same reference profile used by calculate_tvdi()
                ref_prof = self._get_reference_profile()
                tvdi_m = self.apply_field_mask(tvdi_arr, ref_prof)

                ndvi_path = self.raster_paths["ndvi"][i]
                tvdi_dates.append(self._to_ts(self._parse_date_from_name(ndvi_path)))
                tvdi_med.append(float(np.nanmedian(tvdi_m)))
                tvdi_sig.append(float(self._mad_std(tvdi_m)))

            # sort + clip
            ndvi_dates, ndvi_med, ndvi_sig = _sort_by_date(ndvi_dates, ndvi_med, ndvi_sig)
            ndwi_dates, ndwi_med, ndwi_sig = _sort_by_date(ndwi_dates, ndwi_med, ndwi_sig)
            tvdi_dates, tvdi_med, tvdi_sig = _sort_by_date(tvdi_dates, tvdi_med, tvdi_sig)

            ndvi_dates, ndvi_med, ndvi_sig = _clip_range(ndvi_dates, ndvi_med, ndvi_sig, start_ts, end_ts)
            ndwi_dates, ndwi_med, ndwi_sig = _clip_range(ndwi_dates, ndwi_med, ndwi_sig, start_ts, end_ts)
            tvdi_dates, tvdi_med, tvdi_sig = _clip_range(tvdi_dates, tvdi_med, tvdi_sig, start_ts, end_ts)

            # holders for *daily* Teros series used in correlations
            teros_raw_dates, teros_raw_vals = [], []

            # ---------- TEROS ----------
            teros_dates, teros_med, teros_sig = [], [], []
            if include_teros:
                try:
                    ter = pd.read_csv(teros_csv, parse_dates=["Dates (hours)"]).rename(
                        columns={"Dates (hours)": "datetime"}
                    )
                    fld = self.field_letter.upper()
                    fld_cols = [c for c in ter.columns if c.startswith("T_") and c.endswith(fld)] or [
                        c for c in ter.columns if c.startswith("T_")
                    ]
                    if not fld_cols:
                        raise RuntimeError("No Teros VWC columns (T_*) found.")
                    ter = ter[["datetime"] + fld_cols].set_index("datetime").sort_index()
                    daily = (
                        ter.resample("D").mean() if teros_daily_agg.lower() == "mean" else ter.resample("D").median()
                    )
                    agg = daily.mean(axis=1) if teros_sensor_agg.lower() == "mean" else daily.median(axis=1)
                    # raw daily series (for correlations)
                    _raw_dates = [self._to_ts(d) for d in agg.index]
                    _raw_vals = agg.values.astype(float).tolist()
                    # sort + clip (reuse existing helpers)
                    _raw_dates, _raw_vals, _ = _sort_by_date(_raw_dates, _raw_vals, _raw_vals)
                    _raw_dates, _raw_vals, _ = _clip_range(_raw_dates, _raw_vals, _raw_vals, start_ts, end_ts)
                    teros_raw_dates, teros_raw_vals = _raw_dates, _raw_vals

                    win = "15D"
                    teros_15d = agg.resample(win).median()

                    def _mad_std_series(s):
                        s = np.asarray(s, float)
                        s = s[np.isfinite(s)]
                        if s.size == 0:
                            return np.nan
                        med = np.nanmedian(s)
                        mad = np.nanmedian(np.abs(s - med))
                        return 1.4826 * mad

                    teros_15d_sigma = agg.resample(win).apply(_mad_std_series)
                    teros_dates = [self._to_ts(d) for d in teros_15d.index]
                    teros_med = teros_15d.values.astype(float).tolist()
                    teros_sig = teros_15d_sigma.values.astype(float).tolist()

                    # sort + clip
                    def _s(dates, v1, v2):
                        return _clip_range(*_sort_by_date(dates, v1, v2), start_ts, end_ts)

                    teros_dates, teros_med, teros_sig = _s(teros_dates, teros_med, teros_sig)
                except Exception as e:
                    print(f"[WARN] Teros series skipped: {e}")

            # ---------- GPR ----------
            gpr_dates, gpr_med, gpr_sig = [], [], []
            if include_gpr:
                try:
                    gpr = GprAnalysis(field_letter=self.field_letter)  # noqa: F821
                    gpr_tables = gpr.import_data(show=False)
                    gpr_dates_pd = pd.to_datetime(gpr.extract_dates(), format="%d/%m/%Y").map(self._to_ts)
                    for df in gpr_tables:
                        v = np.asarray(df["vwc"].values, dtype=float)
                        v[~np.isfinite(v)] = np.nan
                        gpr_med.append(np.nanmedian(v))
                        gpr_sig.append(self._mad_std(v))
                    n = min(len(gpr_med), len(gpr_dates_pd))
                    pairs = sorted(zip(list(gpr_dates_pd[:n]), gpr_med[:n], gpr_sig[:n]), key=lambda t: t[0])
                    if pairs:
                        gpr_dates, gpr_med, gpr_sig = map(list, zip(*pairs))
                    gpr_dates, gpr_med, gpr_sig = _clip_range(gpr_dates, gpr_med, gpr_sig, start_ts, end_ts)
                except Exception as e:
                    print(f"[WARN] GPR series skipped: {e}")

            if plot:
                title = (
                    f"Evolution of NDVI, NDWI, TVDI, Teros, GPR • Field {self.field_letter} "
                    f"({start_ts.strftime('%d %b %Y')} – {end_ts.strftime('%d %b %Y')})"
                )
                fig, ax = plt.subplots(figsize=(10, 5))
                if ndvi_dates:
                    (ln_ndvi,) = ax.plot(ndvi_dates, ndvi_med, "o-", label="NDVI (median)")
                    ax.errorbar(
                        ndvi_dates,
                        ndvi_med,
                        yerr=ndvi_sig,
                        fmt="none",
                        elinewidth=1,
                        capsize=2,
                        alpha=0.7,
                        ecolor=ln_ndvi.get_color(),
                    )
                if ndwi_dates:
                    (ln_ndwi,) = ax.plot(ndwi_dates, ndwi_med, "s-", label="NDWI (median)")
                    ax.errorbar(
                        ndwi_dates,
                        ndwi_med,
                        yerr=ndwi_sig,
                        fmt="none",
                        elinewidth=1,
                        capsize=2,
                        alpha=0.7,
                        ecolor=ln_ndwi.get_color(),
                    )
                if tvdi_dates:
                    (ln_tvdi,) = ax.plot(tvdi_dates, tvdi_med, "^-", label="TVDI (median)")
                    ax.errorbar(
                        tvdi_dates,
                        tvdi_med,
                        yerr=tvdi_sig,
                        fmt="none",
                        elinewidth=1,
                        capsize=2,
                        alpha=0.7,
                        ecolor=ln_tvdi.get_color(),
                    )
                if teros_dates:
                    (ln_teros,) = ax.plot(teros_dates, teros_med, "d-", label="Teros (15-day median)")
                    ax.errorbar(
                        teros_dates,
                        teros_med,
                        yerr=teros_sig,
                        fmt="none",
                        elinewidth=1,
                        capsize=2,
                        alpha=0.7,
                        ecolor=ln_teros.get_color(),
                    )
                if gpr_dates:
                    (ln_gpr,) = ax.plot(gpr_dates, gpr_med, "x-", label="GPR (median)")
                    ax.errorbar(
                        gpr_dates,
                        gpr_med,
                        yerr=gpr_sig,
                        fmt="none",
                        elinewidth=1,
                        capsize=2,
                        alpha=0.7,
                        ecolor=ln_gpr.get_color(),
                    )
                ax.set_xlabel("Date")
                ax.set_ylabel("Value [-] (VWC for Teros/GPR)")
                ax.set_title(title)
                ax.grid(True)
                ax.legend()
                ax.set_xlim(start_ts, end_ts)
                ax.xaxis.set_major_locator(plt.MaxNLocator(12))
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

                # cross-correlations (unchanged helpers, now with RMSE/MAE too)
                def _pair_and_plot(name1, dates1, vals1, name2, dates2, vals2):
                    A = pd.Series(vals1, index=pd.to_datetime(dates1))
                    B = pd.Series(vals2, index=pd.to_datetime(dates2))
                    common = A.index.intersection(B.index)
                    if len(common) < 3:
                        return

                    # Drop non-finite/degenerate pairs to avoid np.polyfit SVD errors
                    df = (
                        pd.DataFrame(
                            {
                                "x": A.loc[common].values.astype(float),
                                "y": B.loc[common].values.astype(float),
                            }
                        )
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                    )
                    if len(df) < 3:
                        return

                    x = df["x"].to_numpy()
                    y = df["y"].to_numpy()
                    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
                        return

                    try:
                        beta, alpha = np.polyfit(x, y, 1)
                    except np.linalg.LinAlgError:
                        return

                    from sklearn.metrics import mean_absolute_error
                    from sklearn.metrics import mean_squared_error
                    from sklearn.metrics import r2_score

                    yhat = alpha + beta * x
                    r2 = r2_score(y, yhat)
                    rmse = mean_squared_error(y, yhat, squared=False)
                    mae = mean_absolute_error(y, yhat)

                    plt.figure(figsize=(5, 4), dpi=150)
                    plt.scatter(x, y, s=20, alpha=0.7)
                    xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                    plt.plot(xx, alpha + beta * xx, label=f"R²={r2:.2f} • RMSE={rmse:.3f} • MAE={mae:.3f}")
                    plt.xlabel(name1)
                    plt.ylabel(name2)
                    plt.legend()
                    plt.grid(True)
                    plt.title(f"{name1} vs {name2} • Field {self.field_letter}")
                    plt.tight_layout()
                    plt.show()

                if ndvi_dates and ndwi_dates:
                    _pair_and_plot("NDVI", ndvi_dates, ndvi_med, "NDWI", ndwi_dates, ndwi_med)
                if ndvi_dates and tvdi_dates:
                    _pair_and_plot("NDVI", ndvi_dates, ndvi_med, "TVDI", tvdi_dates, tvdi_med)
                if ndwi_dates and tvdi_dates:
                    _pair_and_plot("NDWI", ndwi_dates, ndwi_med, "TVDI", tvdi_dates, tvdi_med)
                if teros_dates:
                    if ndvi_dates:
                        _pair_and_plot("NDVI", ndvi_dates, ndvi_med, "Teros VWC", teros_raw_dates, teros_raw_vals)
                    if ndwi_dates:
                        _pair_and_plot("NDWI", ndwi_dates, ndwi_med, "Teros VWC", teros_raw_dates, teros_raw_vals)
                    if tvdi_dates:
                        _pair_and_plot("TVDI", tvdi_dates, tvdi_med, "Teros VWC", teros_raw_dates, teros_raw_vals)

                if gpr_dates and teros_dates:
                    _pair_and_plot("GPR VWC", gpr_dates, gpr_med, "Teros VWC", teros_raw_dates, teros_raw_vals)

                if gpr_dates:
                    if ndvi_dates:
                        _pair_and_plot("NDVI", ndvi_dates, ndvi_med, "GPR VWC", gpr_dates, gpr_med)
                    if ndwi_dates:
                        _pair_and_plot("NDWI", ndwi_dates, ndwi_med, "GPR VWC", gpr_dates, gpr_med)
                    if tvdi_dates:
                        _pair_and_plot("TVDI", tvdi_dates, tvdi_med, "GPR VWC", gpr_dates, gpr_med)
                if gpr_dates and teros_dates:
                    _pair_and_plot("GPR VWC", gpr_dates, gpr_med, "Teros VWC", teros_dates, teros_med)

            out = {
                "ndvi": {"dates": np.array(ndvi_dates), "mean": np.array(ndvi_med), "variance": np.array(ndvi_sig)},
                "ndwi": {"dates": np.array(ndwi_dates), "mean": np.array(ndwi_med), "variance": np.array(ndwi_sig)},
                "tvdi": {"dates": np.array(tvdi_dates), "mean": np.array(tvdi_med), "variance": np.array(tvdi_sig)},
            }
            if include_teros and len(teros_dates) > 0:
                out["teros"] = {
                    "dates": np.array(teros_dates),
                    "mean": np.array(teros_med),
                    "variance": np.array(teros_sig),
                }
            if include_gpr and len(gpr_dates) > 0:
                out["gpr"] = {"dates": np.array(gpr_dates), "mean": np.array(gpr_med), "variance": np.array(gpr_sig)}
            return out
        finally:
            # restore original field + polygon cache
            self.field_letter = _old_field
            self._poly_xy_cache = _old_poly

    # ---------- TVDI quantile tuner (old feature, adapted) ----------
    def grid_search_tvdi_quantiles(
        self,
        target="teros",  # "teros" | "gpr"
        date_tolerance_days=1,
        wet_list=(2, 5, 10),
        dry_list=(90, 95, 98),
        binning="equal_count",
        target_bins=20,
        teros_csv="Data/Teros Piezo/teros_piezo.csv",
        teros_daily_agg="median",
        teros_sensor_agg="median",
        verbose=True,
        plot_heatmap=True,
    ):
        _to_ts = self._to_ts

        # collect all date items with matching thermal
        date_items = []
        for i in range(len(self.raster_paths["ndvi"])):
            temp, ndvi_r, temp_prof, _ndvi_prof, _tpath, npath = self._import_pair_for_tvdi_previous(i)
            if temp_prof is None:
                continue
            date_items.append((_to_ts(self._parse_date_from_name(npath)), temp, ndvi_r, temp_prof))
        date_items.sort(key=lambda x: x[0])
        if len(date_items) < 3:
            raise RuntimeError("Not enough dates with thermal matches for tuning.")

        # target series (Teros or GPR)
        if target.lower() == "gpr":
            g = GprAnalysis(field_letter=self.field_letter)  # noqa: F821 if not present
            g_tables = g.import_data(show=False)
            g_dates = pd.to_datetime(g.extract_dates(), format="%d/%m/%Y").map(_to_ts)
            g_meds = [np.nanmedian(np.asarray(df["vwc"], dtype=float)) for df in g_tables]
            target_df = pd.DataFrame({"date": g_dates, "VWC": g_meds}).dropna().sort_values("date")
        else:
            ter = pd.read_csv(teros_csv, parse_dates=["Dates (hours)"]).rename(columns={"Dates (hours)": "datetime"})
            fld = self.field_letter.upper()
            fld_cols = [c for c in ter.columns if c.startswith("T_") and c.endswith(fld)] or [
                c for c in ter.columns if c.startswith("T_")
            ]
            ter = ter[["datetime"] + fld_cols].set_index("datetime").sort_index()
            daily = ter.resample("D").mean() if teros_daily_agg == "mean" else ter.resample("D").median()
            agg = daily.mean(axis=1) if teros_sensor_agg == "mean" else daily.median(axis=1)
            target_df = pd.DataFrame({"date": agg.index.map(_to_ts), "VWC": agg.values}).dropna().sort_values("date")

        def _pair_series(tvdi_ser):
            A = tvdi_ser.sort_values("date").copy()
            B = target_df.copy()
            if date_tolerance_days == 0:
                pairs = pd.merge(A, B, on="date", how="inner")
            else:
                pairs = pd.merge_asof(
                    A.sort_values("date"),
                    B.sort_values("date"),
                    on="date",
                    direction="nearest",
                    tolerance=pd.Timedelta(days=date_tolerance_days),
                ).dropna(subset=["VWC"])
            return pairs

        def _r2_loocv(X, Y):
            X = np.asarray(X, dtype=float)
            Y = np.asarray(Y, dtype=float)
            n = X.size
            if n < 3 or not np.isfinite(X).all() or not np.isfinite(Y).all():
                return np.nan
            ybar = np.mean(Y)
            tss = np.sum((Y - ybar) ** 2)
            if tss <= 0:
                return np.nan
            sse_cv = 0.0
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                beta, alpha = np.polyfit(X[mask], Y[mask], 1)
                yhat_i = alpha + beta * X[i]
                sse_cv += (Y[i] - yhat_i) ** 2
            return 1.0 - (sse_cv / tss)

        rows = []
        for wet in wet_list:
            for dry in dry_list:
                if wet >= dry:
                    continue
                tvdi_med_dates, tvdi_med_vals = [], []
                for d, temp, ndvi_r, temp_prof in date_items:
                    temp_f = self.apply_field_mask(temp, temp_prof)
                    ndvi_f = self.apply_field_mask(ndvi_r, temp_prof)
                    tvdi_arr, _ = self._compute_tvdi_from_arrays(
                        temp=temp_f,
                        ndvi_r=ndvi_f,
                        ndvi_bin_width=0.01,
                        pct_wet=wet,
                        pct_dry=dry,
                        min_pixels_per_bin=50,
                        binning=binning,
                        target_bins=target_bins,
                    )
                    tvdi_med_dates.append(d)
                    tvdi_med_vals.append(np.nanmedian(tvdi_arr))
                tvdi_ser = pd.DataFrame({"date": tvdi_med_dates, "TVDI": tvdi_med_vals}).dropna()
                pairs = _pair_series(tvdi_ser)
                X = pairs["TVDI"].to_numpy()
                Y = pairs["VWC"].to_numpy()
                n_pairs = X.size
                if n_pairs < 3:
                    r2_plain = np.nan
                    r2_cv = np.nan
                else:
                    from sklearn.metrics import r2_score

                    beta, alpha = np.polyfit(X, Y, 1)
                    r2_plain = r2_score(Y, alpha + beta * X)
                    r2_cv = _r2_loocv(X, Y)
                rows.append({"wet": wet, "dry": dry, "r2_plain": r2_plain, "r2_cv": r2_cv, "n_pairs": n_pairs})

        df = pd.DataFrame(rows).sort_values(["r2_cv", "r2_plain"], ascending=False)
        best = df.iloc[0].to_dict()
        if verbose:
            print(
                f"[Tuning] Target={target} Best: wet={int(best['wet'])}, dry={int(best['dry'])}, "
                f"CV R²={best['r2_cv']:.3f} (n_pairs={int(best['n_pairs'])})"
            )

        if plot_heatmap:
            wets = sorted(set(wet_list))
            drys = sorted(set(dry_list))
            Z = np.full((len(wets), len(drys)), np.nan, dtype=float)
            for _, r in df.iterrows():
                wi = wets.index(int(r["wet"]))
                di = drys.index(int(r["dry"]))
                Z[wi, di] = r["r2_cv"]
            fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
            im = ax.imshow(Z, origin="lower", aspect="auto")
            ax.set_xticks(range(len(drys)))
            ax.set_xticklabels(drys)
            ax.set_yticks(range(len(wets)))
            ax.set_yticklabels(wets)
            ax.set_xlabel("dry percentile")
            ax.set_ylabel("wet percentile")
            ax.set_title(f"LOO CV R² vs {target.upper()}")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("CV R²")
            plt.tight_layout()
            plt.show()

        return {
            "best": {
                "wet": int(best["wet"]),
                "dry": int(best["dry"]),
                "r2_cv": float(best["r2_cv"]),
                "r2_plain": float(best["r2_plain"]),
                "n_pairs": int(best["n_pairs"]),
            },
            "table": df,
        }

    # ===== K-MEANS ZONAL TENDENCIES (MULTISPECTRAL) =====
    # --- small helpers (vectorized coords, robust σ) ---
    def _ms__robust_std(self, a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return 1.0
        med = np.nanmedian(a)
        mad = np.nanmedian(np.abs(a - med))
        return max(1e-6, 1.4826 * mad)

    def _ms__grid_coords_from_profile(self, rows, cols, profile):
        # Vectorized map coords (center of pixels)
        T = profile["transform"]
        # Affine: x = a*col + b*row + c ; y = d*col + e*row + f
        a, b, c, d, e, f = T.a, T.b, T.c, T.d, T.e, T.f
        cc = cols.astype(float) + 0.5
        rr = rows.astype(float) + 0.5
        X = a * cc + b * rr + c
        Y = d * cc + e * rr + f
        return X, Y

    def _ms__index_on_ref(self, sample_number, index_name):
        """
        Return (arr_on_ref, ref_profile, date_ts) for the requested index and campaign.
        Uses field mask. Index options: 'tvdi' (preferred), 'ndvi', 'ndwi'.
        """
        index_name = str(index_name).lower()
        if index_name == "tvdi":
            temp, ndvi_r, temp_prof, _ndvi_prof, _tpath, npath = self._import_pair_for_tvdi_previous(sample_number)
            if temp_prof is None:
                return None, None, None
            temp_f = self.apply_field_mask(temp, temp_prof)
            ndvi_f = self.apply_field_mask(ndvi_r, temp_prof)
            tvdi_arr, _ = self._compute_tvdi_from_arrays(
                temp=temp_f,
                ndvi_r=ndvi_f,
                pct_wet=5,
                pct_dry=95,
                min_pixels_per_bin=50,
                binning="equal_count",
                target_bins=20,
            )
            d = self._parse_date_from_name(npath)
            return tvdi_arr.astype(np.float32), temp_prof, self._to_ts(pd.to_datetime(d))
        elif index_name == "ndvi":
            # read NDVI, clamp, resample to reference thermal profile
            if sample_number >= len(self.raster_paths["ndvi"]):
                return None, None, None
            arr, prof = self.read_band(self.raster_paths["ndvi"][sample_number])
            arr[(arr < -1) | (arr > 1)] = np.nan
            ref_prof = self._get_reference_profile()
            arr_r = self._resample_to_profile(arr.astype(np.float32), prof, ref_prof, resampling=Resampling.bilinear)
            arr_f = self.apply_field_mask(arr_r, ref_prof)
            d = self._parse_date_from_name(self.raster_paths["ndvi"][sample_number])
            return arr_f.astype(np.float32), ref_prof, self._to_ts(pd.to_datetime(d))
        elif index_name == "ndwi":
            gpn = self._import_green_nir_pair_by_date(sample_number, anchor="green")
            if gpn[0] is None:
                return None, None, None
            g, gprof, n, _nprof, g_path, _n_path = gpn
            ndwi = -(g - n) / (g + n)
            ndwi[(ndwi < -1) | (ndwi > 1) | (ndwi == 0)] = np.nan
            ref_prof = self._get_reference_profile()
            ndwi_r = self._resample_to_profile(ndwi.astype(np.float32), gprof, ref_prof, resampling=Resampling.bilinear)
            ndwi_f = self.apply_field_mask(ndwi_r, ref_prof)
            d = self._parse_date_from_name(g_path)
            return ndwi_f.astype(np.float32), ref_prof, self._to_ts(pd.to_datetime(d))
        else:
            raise ValueError("index must be 'tvdi', 'ndvi' or 'ndwi'")

    def _ms__iter_index_arrays(self, index_name):
        """Yield (arr_on_ref, ref_profile, date_ts) across all campaigns for the chosen index."""
        if index_name == "ndvi":
            N = len(self.raster_paths["ndvi"])
            rng = range(N)
        elif index_name == "ndwi":
            # follow Green count; we will skip dates without NIR match in _import_green_nir_pair_by_date
            N = len(self.raster_paths["green"])
            rng = range(N)
        else:  # tvdi follows NDVI dates that have a thermal match
            N = len(self.raster_paths["ndvi"])
            rng = range(N)
        for i in rng:
            arr, prof, d = self._ms__index_on_ref(i, index_name)
            if arr is not None and prof is not None and d is not None:
                yield i, arr, prof, d

    def _ms__prepare_global_scalers(self, index="tvdi", sample_step=8):
        """Collect global mins/maxs to min-max normalise x,y,VAL across all dates (subsampled)."""
        xs, ys, vs = [], [], []
        for _, arr, prof, _ in self._ms__iter_index_arrays(index):
            h, w = arr.shape
            rr = np.arange(0, h, int(sample_step))
            cc = np.arange(0, w, int(sample_step))
            if rr.size == 0 or cc.size == 0:
                continue
            sub = arr[np.ix_(rr, cc)]
            m = np.isfinite(sub)
            if not m.any():
                continue
            R, C = np.meshgrid(rr, cc, indexing="ij")
            x, y = self._ms__grid_coords_from_profile(R[m], C[m], prof)
            xs.append(x)
            ys.append(y)
            vs.append(sub[m].ravel())
        if not xs:
            raise RuntimeError("No finite pixels available to build k-means scalers.")
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        vs = np.concatenate(vs)
        eps = 1e-9
        return {
            "x_min": float(np.nanmin(xs)),
            "x_max": float(np.nanmax(xs) + eps),
            "y_min": float(np.nanmin(ys)),
            "y_max": float(np.nanmax(ys) + eps),
            "v_min": float(np.nanmin(vs)),
            "v_max": float(np.nanmax(vs) + eps),
        }

    def _ms__build_features(self, x, y, v, scales, coordinate_weight=1.0):
        """Min-max normalise and weight (x,y) then stack with normalised value v."""
        xx = (x - scales["x_min"]) / (scales["x_max"] - scales["x_min"])
        yy = (y - scales["y_min"]) / (scales["y_max"] - scales["y_min"])
        vv = (v - scales["v_min"]) / (scales["v_max"] - scales["v_min"])
        F = np.column_stack([xx * float(coordinate_weight), yy * float(coordinate_weight), vv])
        m = np.isfinite(F).all(axis=1)
        return F[m], m

    # --- PUBLIC API (mirrors GPR names) ---

    def fit_kmeans_zones(
        self,
        n_clusters=3,
        index="tvdi",
        coordinate_weight=1.0,
        random_state=0,
        sample_step=16,  # coarser grid sampling
        max_train_samples=120_000,  # cap the number of pixels used to fit
        use_minibatch=True,  # use MiniBatchKMeans for speed
        minibatch_batch_size=10_000,
        verbose=True,
    ):
        """
        Fit ONE k-means on ALL campaigns using features [x,y,VAL] (min-max normalised).
        VAL is TVDI/NDVI/NDWI depending on 'index'. Uses strong subsampling + MiniBatch option.
        NOTE: force float64 + contiguous arrays to avoid sklearn dtype mismatch on some builds.
        """
        from sklearn.cluster import KMeans
        from sklearn.cluster import MiniBatchKMeans

        rng = np.random.default_rng(int(random_state))

        index = str(index).lower()
        scales = self._ms__prepare_global_scalers(index=index, sample_step=sample_step)

        # assemble subsampled training matrix
        X_all = []
        for _, arr, prof, _ in self._ms__iter_index_arrays(index):
            h, w = arr.shape
            rr = np.arange(0, h, int(sample_step))
            cc = np.arange(0, w, int(sample_step))
            if rr.size == 0 or cc.size == 0:
                continue
            sub = arr[np.ix_(rr, cc)]
            m = np.isfinite(sub)
            if not m.any():
                continue
            R, C = np.meshgrid(rr, cc, indexing="ij")
            x, y = self._ms__grid_coords_from_profile(R[m], C[m], prof)
            v = sub[m].ravel()
            F, _ = self._ms__build_features(x, y, v, scales, coordinate_weight=coordinate_weight)
            if F.size:
                # >>> critical: float64 + C-contiguous
                X_all.append(np.ascontiguousarray(F, dtype=np.float64))
        if not X_all:
            raise RuntimeError("Could not assemble training pixels for k-means.")
        X_all = np.vstack(X_all)

        # cap training size
        if X_all.shape[0] > max_train_samples:
            idx = rng.choice(X_all.shape[0], size=int(max_train_samples), replace=False)
            X_all = X_all[idx]

        # fit
        if use_minibatch:
            km = MiniBatchKMeans(
                n_clusters=int(n_clusters),
                random_state=int(random_state),
                batch_size=int(minibatch_batch_size),
                n_init=3,
                reassignment_ratio=0.01,
            )
        else:
            km = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=int(random_state))

        km.fit(X_all)

        # order clusters by VAL component (dry→wet)
        order = np.argsort(km.cluster_centers_[:, 2])
        remap = {int(old): int(new) for new, old in enumerate(order)}

        # stash
        self._kmeans_ms_model = km
        self._kmeans_ms_scales = scales
        self._kmeans_ms_coord_w = float(coordinate_weight)
        self._kmeans_ms_label_remap = remap
        self._kmeans_ms_index = index
        self._kmeans_ms_sample_step = int(sample_step)

        if verbose:
            cc = km.cluster_centers_[order]
            print(f"[MS-KMeans] Fitted zones on index='{index}' (ordered dry→wet).")
            for i, c in enumerate(cc):
                print(f"  Zone {i}: centroid ~ [x:{c[0]:.2f}, y:{c[1]:.2f}, val_norm:{c[2]:.2f}]")

        return {"model": km, "scales": scales, "remap": remap, "index": index}

    def predict_kmeans_labels_for_campaign(self, sample_number=None, index=None):
        """
        Assign k-means zones to ALL pixels of one campaign; returns a label raster (-1 for invalid).
        Uses the fitted model. Forces float64 contiguous features for sklearn predict().
        """
        if sample_number is None:
            sample_number = self.sample_number
        if not hasattr(self, "_kmeans_ms_model"):
            raise RuntimeError("Call fit_kmeans_zones(...) first.")

        index = self._kmeans_ms_index if index is None else str(index).lower()
        if index != self._kmeans_ms_index:
            raise RuntimeError(f"Model was trained on index='{self._kmeans_ms_index}', got '{index}'.")

        arr, prof, _ = self._ms__index_on_ref(sample_number, index)
        if arr is None:
            raise RuntimeError("No array available for this campaign/index.")

        h, w = arr.shape
        R, C = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        m = np.isfinite(arr)
        if not m.any():
            raise RuntimeError("No finite pixels to assign labels.")

        x, y = self._ms__grid_coords_from_profile(R[m], C[m], prof)
        v = arr[m].ravel()
        F, _ = self._ms__build_features(x, y, v, self._kmeans_ms_scales, coordinate_weight=self._kmeans_ms_coord_w)

        # >>> critical: float64 + C-contiguous for predict()
        F = np.ascontiguousarray(F, dtype=np.float64)

        labs = self._kmeans_ms_model.predict(F)
        labs = np.vectorize(self._kmeans_ms_label_remap.get)(labs)

        out = np.full((h, w), -1, dtype=int)
        out[m] = labs
        return {"labels_raster": out, "valid_mask": m, "n_clusters": self._kmeans_ms_model.n_clusters, "profile": prof}

    def plot_kmeans_zone_map(self, sample_number=None, cmap_name=None, show_points=False, annotate=True):
        """
        Plot categorical zone map (crisp boundaries) for the chosen campaign using
        the already-fitted k-means model (fit_kmeans_zones must have been called).
        """
        if sample_number is None:
            sample_number = self.sample_number
        if not hasattr(self, "_kmeans_ms_model"):
            raise RuntimeError("Call fit_kmeans_zones(...) first.")

        # local imports
        from rasterio.plot import plotting_extent

        try:
            from matplotlib import patheffects as pe
        except Exception:
            pe = None  # optional

        # predict labels for this campaign
        pred = self.predict_kmeans_labels_for_campaign(sample_number)
        Z = pred["labels_raster"].astype(float)
        prof = pred["profile"]
        K = pred["n_clusters"]

        # --- key fix: mask out-of-field pixels (labels < 0) ---
        Z[Z < 0] = np.nan

        # categorical colormap
        if cmap_name is None:
            cmap_name = "tab10" if K <= 10 else "tab20"
        base = plt.get_cmap(cmap_name)
        colors = [base(i % base.N) for i in range(K)]
        from matplotlib.colors import BoundaryNorm
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(colors)
        cmap.set_bad(alpha=0)  # transparent outside the field
        norm = BoundaryNorm(np.arange(-0.5, K + 0.5, 1.0), cmap.N)

        extent = plotting_extent(Z, prof["transform"])
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=300)
        im = ax.imshow(Z, origin="upper", extent=extent, cmap=cmap, norm=norm)

        # field outline from finite mask (now correct because invalid -> NaN)
        arr_mask = np.isfinite(Z).astype(int)
        ax.contour(arr_mask, levels=[0.5], colors="k", linewidths=2, origin="upper", extent=extent)

        # zone boundaries
        try:
            levels = np.arange(-0.5, K - 0.5 + 1e-9, 1.0)
            ax.contour(Z, levels=levels, colors="white", linewidths=1.0, origin="upper", extent=extent)
            ax.contour(Z, levels=levels, colors="k", linewidths=0.4, origin="upper", extent=extent)
        except Exception:
            pass

        # annotate zones at spatial medians (only where zone exists)
        if annotate:
            h, w = Z.shape
            R, C = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            for k in range(K):
                mk = Z == k
                if not np.any(mk):
                    continue
                x, y = self._ms__grid_coords_from_profile(R[mk], C[mk], prof)
                cx, cy = np.nanmedian(x), np.nanmedian(y)
                txt = ax.text(cx, cy, f"Z{k+1}", ha="center", va="center", fontsize=9, weight="bold", color="white")
                if pe is not None:
                    txt.set_path_effects([pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()])

        # optional point overlay (very light)
        if show_points:
            yy, xx = np.where(np.isfinite(Z))
            if yy.size:
                X, Y = self._ms__grid_coords_from_profile(yy, xx, prof)
                ax.plot(X[::200], Y[::200], ".", ms=1, alpha=0.25)

        # cosmetics to match your map style
        self._auto_zoom_to_valid_coords(ax, Z, prof["transform"], pad_pixels=30)
        self._apply_lambert72_grid(ax, major=100, minor=50, labelsize=7)
        self._add_north_arrow(ax, loc="NE", size=0.09, inset=(0.03, 0.01), font_size=9)

        # legend (only for zones that exist)
        import matplotlib.patches as mpatches

        present = sorted({int(v) for v in np.unique(Z[np.isfinite(Z)])})
        handles = [mpatches.Patch(color=colors[i], label=f"Zone {i+1}") for i in present]
        if handles:
            ax.legend(handles=handles, loc="lower left", frameon=True)

        # title with date
        _, _, dts = self._ms__index_on_ref(sample_number, self._kmeans_ms_index)
        ds = dts.strftime("%d/%m/%Y") if dts is not None else "unknown date"
        ax.set_title(f"K-means Zones ({self._kmeans_ms_index.upper()}) • Field {self.field_letter} • {ds}", pad=8)

        plt.tight_layout()
        plt.show()

    def plot_kmeans_zones_evolution(
        self,
        n_clusters=3,
        index="tvdi",
        coordinate_weight=1.0,
        random_state=0,
        start=None,
        end=None,
        cmap_name="tab10",
        sample_step=8,
    ):
        """
        Fit zones once (on all dates), then plot per-zone median of the index through time
        with robust σ bars (MAD→σ). Zones are ordered dry→wet.
        """
        self.fit_kmeans_zones(
            n_clusters=n_clusters,
            index=index,
            coordinate_weight=coordinate_weight,
            random_state=random_state,
            sample_step=sample_step,
            verbose=False,
        )

        # Collect series
        dates_all, med, sig = [], {}, {}
        K = self._kmeans_ms_model.n_clusters
        for k in range(K):
            med[k], sig[k] = [], []

        for i, arr, prof, d in self._ms__iter_index_arrays(self._kmeans_ms_index):
            dates_all.append(d)
            pred = self.predict_kmeans_labels_for_campaign(sample_number=i, index=self._kmeans_ms_index)
            Z = pred["labels_raster"]
            m = np.isfinite(arr)
            for k in range(K):
                mk = (Z == k) & m
                if np.any(mk):
                    vv = arr[mk]
                    med[k].append(float(np.nanmedian(vv)))
                    sig[k].append(float(self._mad_std(vv)))
                else:
                    med[k].append(np.nan)
                    sig[k].append(np.nan)

        # clip by date if requested
        dates = pd.to_datetime(dates_all)
        if start is not None:
            dates_mask_lo = dates >= pd.to_datetime(start)
        else:
            dates_mask_lo = np.ones_like(dates, dtype=bool)
        if end is not None:
            dates_mask_hi = dates <= pd.to_datetime(end)
        else:
            dates_mask_hi = np.ones_like(dates, dtype=bool)
        sel = dates_mask_lo & dates_mask_hi
        dts = list(dates[sel])

        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i % 10) for i in range(K)]
        for k in range(K):
            y = np.array(med[k])[sel]
            e = np.array(sig[k])[sel]
            label = f"Zone {k+1}"  # 1-based, matches the map (Z1, Z2, …)
            (ln,) = ax.plot(dts, y, "o-", label=label, color=colors[k])
            ax.errorbar(dts, y, yerr=e, fmt="none", elinewidth=1, capsize=2, alpha=0.8, ecolor=ln.get_color())

        ax.set_xlabel("Date")
        ax.set_ylabel(f"{self._kmeans_ms_index.upper()} [-]")
        title = f"Zonal evolution (k-means, k={K}, coord_w={self._kmeans_ms_coord_w:g}, index={self._kmeans_ms_index}) • Field {self.field_letter}"
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return {
            "dates": np.array(dts),
            "median": {k: np.array(med[k])[sel] for k in range(K)},
            "sigma": {k: np.array(sig[k])[sel] for k in range(K)},
            "n_clusters": K,
            "coordinate_weight": float(self._kmeans_ms_coord_w),
            "index": self._kmeans_ms_index,
        }

    def plot_kmeans_model_selection(
        self,
        index="tvdi",
        max_k=8,
        coordinate_weight=1.0,
        sample_step=16,  # coarser grid sampling (↑ to 24–40 for more speed)
        max_train_samples=80_000,  # cap the number of pixels used to fit models
        silhouette_max_samples=15_000,  # compute silhouette on a small subset (<=0 to skip)
        use_minibatch=True,  # much faster for big n
        minibatch_batch_size=10_000,
        random_state=0,
    ):
        """
        Elbow (inertia) + silhouette across K=2..max_k for the chosen index,
        using heavy subsampling + optional MiniBatchKMeans for speed.
        Forces float64 contiguous arrays to avoid dtype mismatch.
        """
        from sklearn.cluster import KMeans
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import silhouette_score

        rng = np.random.default_rng(int(random_state))

        index = str(index).lower()
        scales = self._ms__prepare_global_scalers(index=index, sample_step=sample_step)

        # assemble a training matrix with subsampling
        X_all = []
        for _, arr, prof, _ in self._ms__iter_index_arrays(index):
            h, w = arr.shape
            rr = np.arange(0, h, int(sample_step))
            cc = np.arange(0, w, int(sample_step))
            if rr.size == 0 or cc.size == 0:
                continue
            sub = arr[np.ix_(rr, cc)]
            m = np.isfinite(sub)
            if not m.any():
                continue
            R, C = np.meshgrid(rr, cc, indexing="ij")
            x, y = self._ms__grid_coords_from_profile(R[m], C[m], prof)
            v = sub[m].ravel()
            F, _ = self._ms__build_features(x, y, v, scales, coordinate_weight=coordinate_weight)
            if F.size:
                # >>> critical: float64 + C-contiguous
                X_all.append(np.ascontiguousarray(F, dtype=np.float64))
        if not X_all:
            raise RuntimeError("No features to evaluate K.")
        X_all = np.vstack(X_all)

        if X_all.shape[0] > max_train_samples:
            idx = rng.choice(X_all.shape[0], size=int(max_train_samples), replace=False)
            X_all = X_all[idx]

        Ks = list(range(2, int(max_k) + 1))
        inertias, sils = [], []
        for k in Ks:
            if use_minibatch:
                km = MiniBatchKMeans(
                    n_clusters=int(k),
                    random_state=int(random_state),
                    batch_size=int(minibatch_batch_size),
                    n_init=3,
                    reassignment_ratio=0.01,
                )
            else:
                km = KMeans(n_clusters=int(k), n_init=10, random_state=int(random_state))

            km.fit(X_all)
            inertias.append(float(km.inertia_))

            if silhouette_max_samples and silhouette_max_samples > 0:
                ns = min(int(silhouette_max_samples), X_all.shape[0])
                if ns < X_all.shape[0]:
                    sel = rng.choice(X_all.shape[0], size=ns, replace=False)
                    X_sil = X_all[sel]
                    labs_sil = km.labels_[sel]
                else:
                    X_sil = X_all
                    labs_sil = km.labels_
                try:
                    sil = float(silhouette_score(X_sil, labs_sil))
                except Exception:
                    sil = np.nan
            else:
                sil = np.nan
            sils.append(sil)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(Ks, inertias, "o-", label="Inertia (elbow)")
        ax.set_xlabel("K")
        ax.set_ylabel("Inertia")
        ax2 = ax.twinx()
        ax2.plot(Ks, sils, "s--", label="Silhouette", alpha=0.8)
        ax2.set_ylabel("Mean silhouette (subset)")
        ax.set_title(f"K selection (coord_w={coordinate_weight:g}, index={index}) • Field {self.field_letter}")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        return {"K": np.array(Ks), "inertia": np.array(inertias), "silhouette": np.array(sils)}

    def kmeans_index_anova(self, sample_number=None):
        """Report global variance, mean within-zone variance, and one-way ANOVA F/p for the chosen campaign."""
        from scipy.stats import f_oneway

        if sample_number is None:
            sample_number = self.sample_number
        if not hasattr(self, "_kmeans_ms_model"):
            raise RuntimeError("Call fit_kmeans_zones(...) first.")

        arr, _prof, _ = self._ms__index_on_ref(sample_number, self._kmeans_ms_index)
        pred = self.predict_kmeans_labels_for_campaign(sample_number)
        Z, m, K = pred["labels_raster"], pred["valid_mask"], pred["n_clusters"]
        v = arr[m]
        labs = Z[m]

        groups = [v[labs == k] for k in range(K) if np.isfinite(v[labs == k]).any()]
        if len(groups) < 2:
            raise RuntimeError("Need ≥2 non-empty zones for ANOVA.")

        global_var = float(np.nanvar(v))
        within_vars = [float(np.nanvar(g)) for g in groups]
        F = f_oneway(*groups)
        print(f"[MS-ANOVA] global var={global_var:.4f} | mean within-zone var={np.nanmean(within_vars):.4f}")
        print(f"[MS-ANOVA] F={float(F.statistic):.3f}, p={float(F.pvalue):.3g}")
        return {
            "global_var": global_var,
            "within_vars": within_vars,
            "F": float(F.statistic),
            "p": float(F.pvalue),
            "index": self._kmeans_ms_index,
        }

    def compare_tvdi_vs_gpr_methods(
        self,
        field_letter=None,
        date_tolerance_days=2,
        # TVDI settings (defaults match what worked for you)
        tvdi_params=None,  # e.g., dict(binning="equal_count", target_bins=30, min_pixels_per_bin=25, aggregate_factor=2,
        #            use_water_mask=True, use_shadow_mask=True)
        # Deming
        deming_lambda=1.0,  # error-variance ratio λ = σ_x^2 / σ_y^2 (set to 1 if unknown)
        # Quantile regression
        do_quantile=True,
        quantiles=(0.25, 0.5, 0.75),
        # Bootstrap CIs
        bootstrap=2000,
        random_state=0,
        plot=True,
    ):
        """
        Method-comparison analysis between field-median TVDI and field-median GPR VWC.

        Produces:
        - Passing–Bablok and Deming regression (with bootstrap CI)
        - Lin’s CCC (+ bootstrap CI)
        - Spearman rho
        - Bland–Altman plot (VWC vs TVDI->VWC via Deming line)
        - (optional) Quantile regression lines

        Returns a dict with metrics + paired arrays.
        """
        import warnings

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy.stats import spearmanr
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import r2_score

        # ---------- helpers ----------
        def _pair_by_date(dates1, vals1, dates2, vals2, tol_days=1):
            """
            Pair two time series by exact date or nearest date within a tolerance.
            Returns (X, Y, pairs_df) with X from series1 and Y from series2.
            - Deduplicates same-day entries via median.
            - Drops NaT/NaNs.
            """
            A = pd.DataFrame(
                {
                    "date": pd.to_datetime(dates1),
                    "x": np.asarray(vals1, dtype=float),
                }
            ).dropna(subset=["date"])

            B = pd.DataFrame(
                {
                    "date": pd.to_datetime(dates2),
                    "y": np.asarray(vals2, dtype=float),
                }
            ).dropna(subset=["date"])

            # Deduplicate same-day entries to avoid merge_asof warnings
            A = A.groupby("date", as_index=False).agg(x=("x", "median")).sort_values("date")
            B = B.groupby("date", as_index=False).agg(y=("y", "median")).sort_values("date")

            if int(tol_days) == 0:
                pairs = pd.merge(A, B, on="date", how="inner")
            else:
                pairs = pd.merge_asof(A, B, on="date", direction="nearest", tolerance=pd.Timedelta(days=int(tol_days)))

            pairs = pairs.dropna(subset=["x", "y"])
            return pairs["x"].to_numpy(dtype=float), pairs["y"].to_numpy(dtype=float), pairs

        def _lin_ccc(x, y):
            x, y = np.asarray(x, float), np.asarray(y, float)
            mx, my = np.mean(x), np.mean(y)
            vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
            sxy = np.cov(x, y, ddof=1)[0, 1]
            ccc = 2 * sxy / (vx + vy + (mx - my) ** 2)
            return float(ccc)

        def _deming(x, y, lam=1.0):
            x, y = np.asarray(x, float), np.asarray(y, float)
            sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
            sxy = np.cov(x, y, ddof=1)[0, 1]
            if np.isclose(sxy, 0.0):
                return np.nan, np.nan
            num = sy2 - lam * sx2 + np.sqrt((sy2 - lam * sx2) ** 2 + 4 * lam * (sxy**2))
            slope = num / (2 * sxy)
            intercept = np.mean(y) - slope * np.mean(x)
            return float(slope), float(intercept)

        def _passing_bablok(x, y):
            x, y = np.asarray(x, float), np.asarray(y, float)
            n = x.size
            S = []
            for i in range(n - 1):
                dx = x[i + 1 :] - x[i]
                dy = y[i + 1 :] - y[i]
                valid = dx != 0
                if np.any(valid):
                    S.extend((dy[valid] / dx[valid]).tolist())
            if not S:
                return np.nan, np.nan
            slope = np.median(S)
            intercept = np.median(y - slope * x)
            return float(slope), float(intercept)

        def _bootstrap_ci(estimator, data_xy, B=1000, rng=None):
            if B <= 0:
                return None
            rng = np.random.default_rng(rng)
            x, y = data_xy
            n = len(x)
            vals = []
            for _ in range(B):
                idx = rng.integers(0, n, size=n)
                vals.append(estimator(x[idx], y[idx]))
            vals = np.asarray(vals, float)
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            return float(lo), float(hi)

        # ---------- collect field-median TVDI & GPR series (like your evolution plot) ----------
        if field_letter is not None:
            old_f, old_poly = self.field_letter, self._poly_xy_cache
            self.field_letter, self._poly_xy_cache = str(field_letter).upper(), None

        # TVDI per NDVI campaign (median within field)
        tvdi_dates, tvdi_med = [], []
        for i in range(len(self.raster_paths["ndvi"])):
            try:
                tvdi_arr, _ = self.calculate_tvdi(
                    sample_number=i,
                    save=False,
                    plot=False,
                    **(
                        tvdi_params
                        or dict(
                            binning="equal_count",
                            target_bins=30,
                            min_pixels_per_bin=25,
                            aggregate_factor=2,
                            use_water_mask=True,
                            use_shadow_mask=True,
                        )
                    ),
                )
                if tvdi_arr is None:
                    continue
                ref_prof = self._get_reference_profile()
                tvdi_m = self.apply_field_mask(tvdi_arr, ref_prof)
                ndvi_path = self.raster_paths["ndvi"][i]
                tvdi_dates.append(self._to_ts(self._parse_date_from_name(ndvi_path)))
                tvdi_med.append(float(np.nanmedian(tvdi_m)))
            except Exception:
                pass

        # GPR per campaign (median of VWC)
        gpr_dates, gpr_med = [], []
        try:
            gpr = GprAnalysis(field_letter=self.field_letter)  # relies on your existing class
            gpr_tables = gpr.import_data(show=False)
            gpr_dates_pd = pd.to_datetime(gpr.extract_dates(), format="%d/%m/%Y").map(self._to_ts)
            for df, d in zip(gpr_tables, gpr_dates_pd):
                v = np.asarray(df["vwc"].values, float)
                v[~np.isfinite(v)] = np.nan
                if d is not None and np.isfinite(v).any():
                    gpr_dates.append(d)
                    gpr_med.append(float(np.nanmedian(v)))
        except Exception:
            pass

        # Pair by nearest date
        X, Y, P = _pair_by_date(tvdi_dates, tvdi_med, gpr_dates, gpr_med, tol_days=date_tolerance_days)
        if X.size < 3:
            raise RuntimeError("Too few paired dates for method-comparison analysis.")

        # ---------- statistics ----------
        # Ordinary least squares (for reference)
        beta_ols = np.polyfit(X, Y, 1)  # slope, intercept
        Yhat_ols = beta_ols[0] * X + beta_ols[1]
        r2 = r2_score(Y, Yhat_ols)
        rmse = mean_squared_error(Y, Yhat_ols, squared=False)
        mae = mean_absolute_error(Y, Yhat_ols)

        # Deming (errors-in-variables)
        slope_d, intercept_d = _deming(X, Y, lam=float(deming_lambda))
        Yhat_d = slope_d * X + intercept_d if np.isfinite(slope_d) else np.full_like(X, np.nan)

        # Passing–Bablok
        slope_pb, intercept_pb = _passing_bablok(X, Y)

        # Concordance correlation (Lin)
        ccc = _lin_ccc(X, Y)

        # Spearman (monotonic)
        rho, p_rho = spearmanr(X, Y)

        # Bootstrap CIs
        rng = np.random.default_rng(int(random_state))
        ci_ccc = _bootstrap_ci(lambda a, b: _lin_ccc(a, b), (X, Y), B=int(bootstrap), rng=rng)
        ci_d_slope = _bootstrap_ci(
            lambda a, b: _deming(a, b, lam=float(deming_lambda))[0], (X, Y), B=int(bootstrap), rng=rng
        )
        ci_pb_slope = _bootstrap_ci(lambda a, b: _passing_bablok(a, b)[0], (X, Y), B=int(bootstrap), rng=rng)

        # Bland–Altman: compare observed VWC to calibrated TVDI->VWC (use Deming line)
        # NOTE: BA requires same units; so we transform TVDI via Deming to VWC.
        y1 = Y
        y2 = Yhat_d
        ba_mean = np.nanmean(y1 - y2)
        ba_sd = np.nanstd(y1 - y2, ddof=1)
        loa_low = ba_mean - 1.96 * ba_sd
        loa_high = ba_mean + 1.96 * ba_sd

        # ---------- plotting ----------
        out = {
            "n_pairs": int(X.size),
            "ols": {
                "slope": float(beta_ols[0]),
                "intercept": float(beta_ols[1]),
                "r2": float(r2),
                "rmse": float(rmse),
                "mae": float(mae),
            },
            "deming": {"slope": float(slope_d), "intercept": float(intercept_d), "slope_CI95": ci_d_slope},
            "passing_bablok": {"slope": float(slope_pb), "intercept": float(intercept_pb), "slope_CI95": ci_pb_slope},
            "ccc": {"value": float(ccc), "CI95": ci_ccc},
            "spearman": {"rho": float(rho), "p": float(p_rho)},
            "bland_altman": {
                "bias": float(ba_mean),
                "sd": float(ba_sd),
                "loa_low": float(loa_low),
                "loa_high": float(loa_high),
            },
            "paired": {"tvdi": X, "gpr_vwc": Y, "dates": P},
        }

        if not plot:
            if field_letter is not None:
                self.field_letter, self._poly_xy_cache = old_f, old_poly
            return out

        # (1) Method-comparison scatter with fits (same look/feel as your current scatter)
        plt.figure(figsize=(6, 4.5), dpi=150)
        plt.scatter(X, Y, s=30, alpha=0.8)
        xx = np.linspace(np.nanmin(X), np.nanmax(X), 200)
        plt.plot(xx, beta_ols[0] * xx + beta_ols[1], label=f"OLS  R²={r2:.2f}", linewidth=2)
        if np.isfinite(slope_d):
            lbl = f"Deming λ={deming_lambda:g}  slope={slope_d:.2f}"
            if ci_d_slope:
                lbl += f" [{ci_d_slope[0]:.2f},{ci_d_slope[1]:.2f}]"
            plt.plot(xx, slope_d * xx + intercept_d, linestyle="--", linewidth=2, label=lbl)
        if np.isfinite(slope_pb):
            lbl = f"Passing–Bablok slope={slope_pb:.2f}"
            if ci_pb_slope:
                lbl += f" [{ci_pb_slope[0]:.2f},{ci_pb_slope[1]:.2f}]"
            plt.plot(xx, slope_pb * xx + intercept_pb, linestyle=":", linewidth=2, label=lbl)
        plt.xlabel("TVDI (median)")
        plt.ylabel("GPR VWC (median)")
        ttl_field = self.field_letter
        plt.title(f"TVDI vs GPR VWC • Field {ttl_field}")
        box = f"CCC={ccc:.2f}"
        if ci_ccc:
            box += f" [{ci_ccc[0]:.2f},{ci_ccc[1]:.2f}]"
        box += f" • Spearman ρ={rho:.2f}"
        plt.legend(title=box, loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Optional: quantile regression overlays (if enough points and statsmodels available)
        if do_quantile and X.size >= 10:
            try:
                import statsmodels.api as sm

                plt.figure(figsize=(6, 4.5), dpi=150)
                plt.scatter(X, Y, s=30, alpha=0.8)
                X1 = sm.add_constant(X)
                for q in quantiles:
                    mod = sm.QuantReg(Y, X1).fit(q=q)
                    plt.plot(xx, mod.params[0] + mod.params[1] * xx, linewidth=2, label=f"τ={q:g}")
                plt.xlabel("TVDI (median)")
                plt.ylabel("GPR VWC (median)")
                plt.title(f"Quantile regression • Field {ttl_field}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        # (2) Bland–Altman plot (VWC vs TVDI→VWC using Deming line)
        if np.isfinite(Yhat_d).any():
            mean_pair = (y1 + y2) / 2.0
            diff_pair = y1 - y2
            plt.figure(figsize=(6, 4.5), dpi=150)
            plt.scatter(mean_pair, diff_pair, s=30, alpha=0.8)
            plt.axhline(ba_mean, linewidth=2, label=f"Bias={ba_mean:.3f}")
            plt.axhline(loa_low, linestyle="--", label=f"LoA low={loa_low:.3f}")
            plt.axhline(loa_high, linestyle="--", label=f"LoA high={loa_high:.3f}")
            plt.xlabel("Mean of (GPR VWC, TVDI→VWC)")
            plt.ylabel("Difference (GPR VWC − TVDI→VWC)")
            plt.title(f"Bland–Altman • Field {ttl_field}")
            plt.legend(loc="best")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        if field_letter is not None:
            self.field_letter, self._poly_xy_cache = old_f, old_poly
        return out
