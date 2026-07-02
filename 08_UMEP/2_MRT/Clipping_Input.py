from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds


INPUT_DIR = Path("/storage/project/r-rbasu31-0/shared/Metro_Boston/01_Input")
OUTPUT_DIR = Path("/storage/project/r-rbasu31-0/shared/Concord/01_Input")
CLIP_GEOJSON = Path("/storage/project/r-rbasu31-0/shared/Metro_Boston/concord_5k.geojson")   # change this

OUTPUT_CRS_WKT = """PROJCS["NAD83(2011) / Massachusetts Mainland",
    GEOGCS["NAD83(2011)",
        DATUM["NAD83_National_Spatial_Reference_System_2011",
            SPHEROID["GRS 1980",6378137,298.257222101,
                AUTHORITY["EPSG","7019"]],
            AUTHORITY["EPSG","1116"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","6318"]],
    PROJECTION["Lambert_Conformal_Conic_2SP"],
    PARAMETER["latitude_of_origin",41],
    PARAMETER["central_meridian",-71.5],
    PARAMETER["standard_parallel_1",42.6833333333333],
    PARAMETER["standard_parallel_2",41.7166666666667],
    PARAMETER["false_easting",200000],
    PARAMETER["false_northing",750000],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH],
    AUTHORITY["EPSG","6491"]]"""


def main():
    gdf = gpd.read_file(CLIP_GEOJSON)
    xmin, ymin, xmax, ymax = gdf.total_bounds

    tif_files = list(INPUT_DIR.rglob("*.tif"))
    print(f"Found {len(tif_files)} tif files")

    for tif_path in tif_files:
        rel_path = tif_path.relative_to(INPUT_DIR)
        out_path = OUTPUT_DIR / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with rasterio.open(tif_path) as src:
                window = from_bounds(xmin, ymin, xmax, ymax, src.transform)
                window = window.round_offsets().round_lengths()

                data = src.read(window=window)
                transform = src.window_transform(window)

                meta = src.meta.copy()
                meta.update({
                    "height": data.shape[1],
                    "width": data.shape[2],
                    "transform": transform,
                    "crs": OUTPUT_CRS_WKT
                })

                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(data)

            print(f"Clipped: {tif_path} -> {out_path}")

        except Exception as e:
            print(f"Failed: {tif_path}")
            print(f"  {e}")


if __name__ == "__main__":
    main()