import os
import math
import glob
import multiprocessing as mp
import time
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union
from scipy.ndimage import label
from rasterstats import zonal_stats
import pandas as pd
import geopandas as gpd
import networkx as nx

demfile_pattern = f"/storage/project/r-rbasu31-0/shared/Metro_Boston/Bus_Scenario/Everett_01/01_Input_S5_*/cDSM.tif"
demfiles = sorted(glob.glob(demfile_pattern))

# ── CONFIG ────────────────────────────────────────────────────────────────────────
PRJ_CRS = 6491
TILE_SIZE   = 5000      # pixels
BUFFER_PX   = 100       # pixels
CONNECTIVITY_STRUCT = np.array([[0,1,0],
                                [1,1,1],
                                [0,1,0]], dtype=int)
BUFFER_DIST = 2.0       # meters
MIN_AREA    = 2.0       # m²
TOL = 0.5               # tolerance of geometry simplification 
# ────────────────────────────────────────────────────────────────────────────────

def process_tile(args):
    """Read one window, vectorize, buffer/merge/dedeuffer, zonal‐stats, crop, write."""
    row_off, col_off, win_h, win_w = args
    out_fp = os.path.join(INT_DIR, f"chunk_r{row_off}_c{col_off}.geojson")
#     if os.path.exists(out_fp):
#         return  # skip existing

    with rasterio.open(RASTER_PATH) as src:
        # build buffered window, clipped to raster bounds
        buf = BUFFER_PX
        row0 = max(0, row_off - buf)
        col0 = max(0, col_off - buf)
        row1 = min(src.height, row_off + win_h + buf)
        col1 = min(src.width,  col_off + win_w + buf)

        window = Window(col0, row0, col1 - col0, row1 - row0)
        data = src.read(1, window=window)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = 0

        if data.sum() == 0:
            return  # nothing here

        # connected‐component labeling
        mask = data != 0
        labeled, nlabels = label(mask, structure=CONNECTIVITY_STRUCT)

        # vectorize all regions at once
        geoms = []
        transform = src.window_transform(window)
        for geom, val in shapes(labeled, mask=(labeled > 0), transform=transform):
            # only keep the “shape” geometry, ignore val
            shp = shape(geom)
            # buffer→merge step will be done below
            geoms.append(shp)

        if not geoms:
            return

        # merge all, buffer out then back in, filter by area
        merged = unary_union(geoms).buffer(BUFFER_DIST)
        cleaned = merged.buffer(-BUFFER_DIST)
        if cleaned.is_empty:
            return

        # explode multi‐geoms and filter small bits
        polys = []
        for poly in (cleaned.geoms if hasattr(cleaned, 'geoms') else [cleaned]):
            if poly.area >= MIN_AREA:
                polys.append(poly)

        if not polys:
            return

        # compute zonal stats (mean) on original raster
        zs = zonal_stats(
            polys,
            RASTER_PATH,
            stats=['mean'],
            all_touched=True,
            geojson_out=False
        )

        # build GeoDataFrame
        gdf = gpd.GeoDataFrame(
            [{'mean_h': z['mean']} for z in zs],
            geometry=polys,
            crs=PRJ_CRS
        )

        # crop back to the un‐buffered tile extent
        # tile bounds in map coords:
        top_left = src.transform * (col_off, row_off)
        bottom_right = src.transform * (col_off + win_w, row_off + win_h)
        tile_box = box(
            top_left[0], bottom_right[1],
            bottom_right[0], top_left[1]
        )
        gdf['geometry'] = gdf.geometry.intersection(tile_box)
        gdf = gdf[~gdf.geometry.is_empty]

        # write out
        if not os.path.exists(INT_DIR):
            os.makedirs(INT_DIR)
        # simplification
        gdf_simple = gdf.copy()
        gdf_simple["geometry"] = gdf_simple.geometry.simplify(tolerance=TOL, preserve_topology=True)
        gdf_simple.to_file(out_fp, driver="GeoJSON")

def main():
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"  # no size limit
    # prepare tile windows
    with rasterio.open(RASTER_PATH) as src:
        nrows = math.ceil(src.height / TILE_SIZE)
        ncols = math.ceil(src.width  / TILE_SIZE)
        raster_h = src.height  
        raster_w = src.width

    tasks = []
    for i in range(nrows):
        for j in range(ncols):
            row_off = i * TILE_SIZE
            col_off = j * TILE_SIZE
            win_h = min(TILE_SIZE, raster_h - row_off)
            win_w = min(TILE_SIZE, raster_w  - col_off)
            tasks.append((row_off, col_off, win_h, win_w))
            
    print("Preparing Parallel processing")
    # parallel processing
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(process_tile, tasks)

    # merge all chunks
    all_files = glob.glob(os.path.join(INT_DIR, "chunk_*.geojson"))
    gdfs = [gpd.read_file(fp) for fp in all_files]
    if gdfs:
        print(gdfs[0].crs)
        merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), geometry = 'geometry', crs=gdfs[0].crs)
        #    Each node is a feature index; edges link features that intersect.
        G = nx.Graph()
        G.add_nodes_from(merged.index)
        
        # use spatial index for speed
        sindex = merged.sindex
        for idx, geom in merged.geometry.items():
            # find candidates whose bbox touches this one
            possible = list(sindex.intersection(geom.bounds))
            for j in possible:
                if idx < j and geom.intersects(merged.geometry[j]):
                    G.add_edge(idx, j)    

        # ── 3) Extract connected components ──────────────────────────────────────────
        #    Each component is a set of indices to dissolve together.
        components = list(nx.connected_components(G))
        # map each original index → its component ID
        comp_map = {}
        for comp_id, comp in enumerate(components):
            for idx in comp:
                comp_map[idx] = comp_id
        
        merged["grp"] = merged.index.map(comp_map)

        # ── 4) Dissolve by component, averaging mean_h ─────────────────────────────
        dissolved = merged.dissolve(
            by="grp",
            aggfunc={ "mean_h": "mean" }  # aggregates mean_h across the group
        ).reset_index(drop=True)

        final = dissolved[["mean_h", "geometry"]]
        final['mean_h'] = final['mean_h'].round().astype(int)

        # ── 5) Simplification ───────────────────────────────────────────────────────
        final_simple = final.copy()
        final_simple["geometry"] = final_simple.geometry.simplify(tolerance=TOL, preserve_topology=True)
        output_dir = os.path.dirname(OUTPUT_PATH)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        final_simple.to_file(OUTPUT_PATH)
        print("Saved final dissolved vegetation")

if __name__ == "__main__":
    start = time.time()
    for demfile in demfiles:
        RASTER_PATH = demfile
        OUTPUT_DIR = os.path.dirname(demfile)
        
        INT_DIR      = os.path.join(OUTPUT_DIR, "chunks")   
        os.makedirs(INT_DIR, exist_ok=True)
        
        OUTPUT_PATH = OUTPUT_DIR + "/vegetation.shp"
        main()
    print(f"processing time: {time.time() - start} s")