#!/usr/bin/env python3
"""
Diagnose why exactextract zonal means come back NaN.
Run this once; it tells you whether the NaN are from edges falling OUTSIDE
the raster extent vs. edges sitting over NODATA, plus flags CRS/geometry issues.
"""
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
import geopandas as gpd
from exactextract import exact_extract

NET_PATH   = "/storage/project/r-rbasu31-0/hyu483/City_Atlanta/networks/Atlanta-062226_selected.geojson"
RASTER_DIR = "/storage/project/r-rbasu31-0/hyu483/City_Atlanta/Output_UTCI"
SAMPLE_TIME = "9am"
raster_path = os.path.join(RASTER_DIR, f"UTCI_{SAMPLE_TIME}.tif")

print("=== 1. Network ===")
net = gpd.read_file(NET_PATH)
print(f"features: {len(net)}   CRS: {net.crs}")
print(f"geom types: {net.geom_type.value_counts().to_dict()}")
n_null  = net.geometry.isna().sum()
n_empty = (~net.geometry.isna() & net.geometry.is_empty).sum()
print(f"null geoms: {n_null}   empty geoms: {n_empty}  (these ALWAYS give NaN)")
if "id" in net.columns:
    print(f"'id' unique? {net['id'].is_unique}   dups: {net['id'].duplicated().sum()}   "
          f"nulls: {net['id'].isna().sum()}   <-- non-unique/null id breaks merge(on='id')")
else:
    print("!! no 'id' COLUMN -> include_cols='id' and merge(on='id') would fail")

print("\n=== 2. Raster ===")
with rasterio.open(raster_path) as ds:
    rcrs, rbounds, nd = ds.crs, ds.bounds, ds.nodata
    print(f"CRS: {rcrs}")
    print(f"bounds: {rbounds}")
    print(f"res: {ds.res}   size: {ds.width}x{ds.height}   nodata: {nd}")
    factor = max(1, max(ds.height, ds.width) // 1500)
    arr = ds.read(1,
                  out_shape=(max(1, ds.height // factor), max(1, ds.width // factor)),
                  resampling=Resampling.nearest).astype("float64")
total = arr.size
inval = np.isnan(arr)
if nd is not None:
    inval |= (arr == nd)
print(f"approx nodata/NaN fraction of raster: {inval.mean():.1%}   "
      f"<-- if high, most edges may sit on nodata")
if (~inval).any():
    v = arr[~inval]
    print(f"valid value range: {v.min():.2f} .. {v.max():.2f}")

print("\n=== 3. CRS / extent overlap ===")
if rcrs is None:
    print("!! raster has NO CRS -> to_crs(rcrs) fails / geometries won't align")
net_r = net.to_crs(rcrs)
nb = net_r.total_bounds  # minx, miny, maxx, maxy
print(f"network bounds (raster CRS): {nb}")
print(f"raster  bounds:              {tuple(rbounds)}")
overlap = not (nb[2] < rbounds.left or nb[0] > rbounds.right or
               nb[3] < rbounds.bottom or nb[1] > rbounds.top)
print(f"bounding boxes overlap at all: {overlap}")
cent = net_r.geometry.centroid
inside = ((cent.x >= rbounds.left) & (cent.x <= rbounds.right) &
          (cent.y >= rbounds.bottom) & (cent.y <= rbounds.top))
print(f"edges whose centroid is OUTSIDE the raster bbox: "
      f"{(~inside).sum()} / {len(net_r)} ({(~inside).mean():.1%})")
if net_r.crs.is_geographic:
    print("!! raster CRS is GEOGRAPHIC -> buffer(1) = 1 DEGREE (~111 km). Buffer distance is wrong.")

print("\n=== 4. Run exact_extract once and classify the NaN ===")
buf = net_r.copy()
buf["geometry"] = buf.geometry.buffer(1)          # same buffer the original code uses
res = exact_extract(raster_path, buf, ["mean", "count"], output="pandas")
nan_mask = res["mean"].isna().to_numpy()
zero_cov = (res["count"].to_numpy() == 0)
print(f"NaN means: {nan_mask.sum()} / {len(res)} ({nan_mask.mean():.1%})")
print(f"  -> covered NO raster cells (OUTSIDE extent):     {(nan_mask & zero_cov).sum()}")
print(f"  -> covered cells but ALL nodata:                 {(nan_mask & ~zero_cov).sum()}")
print(f"  -> of NaN rows, centroid outside bbox:           {(nan_mask & ~inside.to_numpy()).sum()}")
print("\nIf most NaN are 'OUTSIDE extent' -> network is bigger than the UTCI footprint.")
print("If most NaN are 'ALL nodata'     -> rasters mask large areas; buffer onto valid cells or accept the loss.")