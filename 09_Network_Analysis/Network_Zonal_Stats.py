#!/usr/bin/env python3
"""
UTCI zonal means onto a road network, robust version.

Key changes vs. the original:
- No merge(on="id"): exact_extract preserves input order, so columns are
  assigned by position. This removes all NaN caused by non-unique/mismatched ids.
- Network is reprojected to the raster CRS ONCE, on a stable frame; geometry
  for OUTPUT is kept in the original CRS (no round-trip drift).
- null/empty geometries are dropped up front (they always produce NaN).
- buffer distance is sized to the raster resolution, not a blind 1.
- NaN are reported per-time; rows are only dropped once at the end (and only
  if NaN for ALL times), instead of compounding the drop every iteration.
"""
import os
import numpy as np
import rasterio
import geopandas as gpd
from exactextract import exact_extract

NET_PATH            = "/storage/project/r-rbasu31-0/hyu483/City_Atlanta/networks/Atlanta-062226_selected.geojson"
RASTER_DIR          = "/storage/project/r-rbasu31-0/hyu483/City_Atlanta/Output_UTCI"
UTCI_VECTOR_OUTPUT  = "/storage/project/r-rbasu31-0/hyu483/City_Atlanta/networks/Atlanta-062226_UTCI_3.geojson"

paths = {}
time_list = ["11"]

# Dictionary comprehension to format each hour
time_lists = {
    hour: f"{int(hour) if int(hour) <= 12 else int(hour) - 12}{'am' if int(hour) < 12 else 'pm'}"
    for hour in time_list
}
print(time_lists)

BUFFER_DIST = None   # in raster-CRS units (meters if projected). None -> auto (~1 pixel)


def min_max(gdf, col):
    mn, mx = gdf[col].min(), gdf[col].max()
    rng = mx - mn
    if rng == 0 or np.isnan(rng):
        gdf[col + "_scale"] = 0.0
        print(f"{col}: min == max (or all NaN); scale set to 0")
    else:
        gdf[col + "_scale"] = (gdf[col] - mn) / rng
    print(f"{col}: min={mn:.3f}  max={mx:.3f}")
    return gdf


# --- raster CRS + resolution from a sample raster ---
sample = os.path.join(RASTER_DIR, f"UTCI_{next(iter(time_lists.values()))}.tif")
with rasterio.open(sample) as ds:
    rcrs, rres = ds.crs, ds.res
if rcrs is None:
    raise ValueError("Sample raster has no CRS; cannot align the network to it.")

# --- load network, drop unusable geometries up front ---
net = gpd.read_file(NET_PATH)
orig_crs = net.crs
bad = net.geometry.isna() | net.geometry.is_empty
if bad.any():
    print(f"Dropping {bad.sum()} null/empty geometries")
net = net[~bad].reset_index(drop=True)

# --- reprojected + buffered copy used ONLY for extraction ---
net_r = net.to_crs(rcrs)
if net_r.crs.is_geographic:
    print("WARNING: raster CRS is geographic; BUFFER_DIST is in DEGREES, not meters.")
if BUFFER_DIST is None:
    BUFFER_DIST = max(abs(rres[0]), abs(rres[1]))   # ~one pixel
print(f"Using buffer distance: {BUFFER_DIST} ({rcrs.to_string()})")
buf = net_r.copy()
buf["geometry"] = buf.geometry.buffer(BUFFER_DIST)

# --- extract each time onto the original-CRS network, by position ---
val_cols = []
for time_int, time in time_lists.items():
    raster_path = os.path.join(RASTER_DIR, f"UTCI_{time}.tif")
    if not os.path.exists(raster_path):
        print(f"!! missing raster: {raster_path}")
        continue
    res = exact_extract(raster_path, buf, ["mean"], output="pandas")
    col = f"UTCI_{time}"
    net[col] = res["mean"].to_numpy()          # order-preserving; no merge
    val_cols.append(col)

    # report, then fill NaN with the column average (skips NaN by default)
    n_nan    = net[col].isna().sum()
    col_mean = net[col].mean()
    print(f"{col}: {n_nan}/{len(net)} NaN ({n_nan/len(net):.1%})  -> filling with mean {col_mean:.3f}")
    net[col] = net[col].fillna(col_mean)

    net = min_max(net, col)   # now NaN-free, so *_scale is also complete

# --- summary ---
print(f"\nDone. {len(net)} features, no rows dropped (NaN filled with per-time mean).")

# --- GeoJSON should be EPSG:4326 ---
if net.crs is not None and not net.crs.equals("EPSG:4326"):
    net = net.to_crs(4326)
net.to_file(UTCI_VECTOR_OUTPUT, driver="GeoJSON")
print(f"Wrote {len(net)} features -> {UTCI_VECTOR_OUTPUT}")