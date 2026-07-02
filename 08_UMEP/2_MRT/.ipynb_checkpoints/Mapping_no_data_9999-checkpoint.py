import numpy as np
import rasterio
from rasterio.enums import ColorInterp

INPUT_PATH = "/storage/project/r-rbasu31-0/shared/Metro_Boston/Bus_Scenario/Chelsea/01_Input/cDSM_fixed.tif"
OUTPUT_PATH = "/storage/project/r-rbasu31-0/shared/Metro_Boston/Bus_Scenario/Chelsea/01_Input/cDSM_fixed.tif"

def remap_nodata(input_path: str, output_path: str) -> None:
    """
    Reads a GeoTIFF, replaces current NoData pixels with 0,
    then writes the result with NoData set to -9999.

    Args:
        input_path:  Path to the source GeoTIFF.
        output_path: Path for the output GeoTIFF.
    """
    print(f"Reading {input_path}")
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        nodata_val = src.nodata          # e.g. -9999, 255, nan, None
        print(f"No data is currently {nodata_val}")
        # Update the profile for the output file
        profile.update(
            dtype=rasterio.float32,     # float32 handles -9999 cleanly
            nodata=-9999,
            bigtiff = "YES"
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                data = src.read(band_idx).astype(np.float32)

                if nodata_val is not None:
                    if np.isnan(nodata_val):
                        mask = np.isnan(data)
                    else:
                        mask = np.isclose(data, float(nodata_val))
                else:
                    # Fall back to the band's internal mask (alpha / mask band)
                    mask = src.dataset_mask() == 0   # 0 = masked/invalid

                # Step 1 – map old NoData → 0
                data[mask] = 0

                # Step 2 – write with new NoData = -9999
                # (no pixels currently equal -9999 after the remap,
                #  so the band contains real zeros where NoData used to be)
                dst.write(data, band_idx)

    print(f"Done → {output_path}  (NoData is now -9999; old NoData pixels → 0)")

if __name__ == "__main__":
    remap_nodata(INPUT_PATH, OUTPUT_PATH)