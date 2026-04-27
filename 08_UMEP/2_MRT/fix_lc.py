from osgeo import gdal
import numpy as np

infile = "/storage/scratch1/2/glin71/SOLWEIG/input/concord_5k/LULC_6491_5class.tif"
outfile = "/storage/scratch1/2/glin71/SOLWEIG/input/concord_5k/land_cover_fixed.tif"

src = gdal.Open(infile)
if src is None:
    raise RuntimeError(f"Could not open: {infile}")

band = src.GetRasterBand(1)
arr = band.ReadAsArray()

# Convert to float first so we can safely catch NaN
arr = arr.astype(np.float32)

# Build clean output
out = np.ones(arr.shape, dtype=np.uint8)

# Keep only values already in 1..7
valid_mask = np.isfinite(arr) & (arr >= 1) & (arr <= 7) & (np.floor(arr) == arr)
out[valid_mask] = arr[valid_mask].astype(np.uint8)

driver = gdal.GetDriverByName("GTiff")
dst = driver.Create(
    outfile,
    src.RasterXSize,
    src.RasterYSize,
    1,
    gdal.GDT_Byte
)

dst.SetGeoTransform(src.GetGeoTransform())
dst.SetProjection(src.GetProjection())

out_band = dst.GetRasterBand(1)
out_band.WriteArray(out)

# For this smoke test, force nodata to 1 too
out_band.SetNoDataValue(1)
out_band.FlushCache()

dst = None
src = None

# Re-open and verify
check = gdal.Open(outfile)
check_band = check.GetRasterBand(1)
check_arr = check_band.ReadAsArray()
print("Output dtype:", check_arr.dtype)
print("Unique values:", np.unique(check_arr))
print("NoData:", check_band.GetNoDataValue())