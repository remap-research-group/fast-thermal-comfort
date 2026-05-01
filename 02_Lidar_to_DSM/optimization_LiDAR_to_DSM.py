import matplotlib.pyplot as plt
import numpy as np
import tempfile
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
import laspy
import glob
import os
import pdal
import subprocess
import traceback
from osgeo import gdal, osr 
import math 
import json 
import time
import subprocess
import traceback
import json 
import argparse
from functools import partial
from multiprocessing import Pool
from rasterio.windows import Window
from scipy.interpolate import griddata
from tqdm import tqdm
from typing import Tuple, List


# Include Live Indication!!

### --- Configuration (User MUST update these paths) --- ###
LIDAR_DIR = "/storage/project/r-rbasu31-0/hyu483/Metro_Atlanta/Raw/Lidar/test_126"
FILTERED_LIDAR_DIR = "/storage/project/r-rbasu31-0/hyu483/Metro_Atlanta/Raw/Lidar/veg_classified/"
FILTER_CLASS = [5] # ground, building, water, bridge deck
### ---------------------------------------------------- ###
### --- Configuration (User MUST update these paths) --- ###
LIDAR_TO_MERGE = "/storage/project/r-rbasu31-0/hyu483/Metro_Atlanta/Raw/Lidar/classified/" 
LIDAR_MERGED = "/storage/project/r-rbasu31-0/hyu483/Metro_Atlanta/Raw/Lidar/Lidar_merged.laz"
### ---------------------------------------------------- ###
### --- Configuration (User MUST update these paths) --- ###
INPUT_LIDAR = "/storage/project/r-rbasu31-0/hyu483/Metro_Atlanta/Raw/Lidar/Lidar_merged.laz"
BUFFERED_TILES_DIR = "/storage/project/r-rbasu31-0/hyu483/Test_UMEP/LiDAR/LiDAR_DSM/tiles_buffered" 
INTERMEDIATE_DEBUFFERED_DIR = "/storage/project/r-rbasu31-0/hyu483/Test_UMEP/LiDAR/LiDAR_DSM/tiles_debuffered"
OUTPUT_TIF_PATH = "/storage/project/r-rbasu31-0/hyu483/Test_UMEP/LiDAR/LiDAR_DSM/DSM_Final/DSM_merged.tif"

# Tiling and Rasterization parameters
TILE_LENGTH = 1000.0 
BUFFER = 20.0        
RESOLUTION = 1.0     
OUTPUT_TYPE = "mean"
DIM = "Z"      
NODATA = -9999.0     
SOURCE_CRS = 6349
DEST_CRS = 26916

### --- Configuration (User MUST update these paths) --- ###
INPUT_TO_INTERPOLATE = "/storage/project/r-rbasu31-0/hyu483/Test_UMEP/LiDAR/LiDAR_DSM/DSM_Final/DSM_merged.tif"
INTERPOLATED_OUTPUT_PATH = "/storage/project/r-rbasu31-0/hyu483/Test_UMEP/LiDAR/LiDAR_DSM/DSM_Final/DSM_intp.tif"

# Parameter
INVALID_NUM = 0




def filter_lidar_by_classification(
    input_dir: str,
    output_dir: str,
    classification_filter: list
) -> None:
    """
    Filters LAZ files based on specified classification codes and writes the filtered outputs.

    Parameters:
    - input_dir (str): Directory containing the input LAZ files.
    - output_dir (str): Directory where filtered LAZ files will be saved.
    - classification_filter (list): List of classification codes to retain (e.g., [2, 6, 9, 17]).

    Returns:
    - None
    """
    os.makedirs(output_dir, exist_ok=True)
    laz_files = glob.glob(os.path.join(input_dir, "*.laz"))

    for file_path in laz_files:
        try:
            las = laspy.read(file_path)
            mask = np.isin(las.classification, classification_filter)
            filtered_las = las[mask]

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}_filtered.laz")

            filtered_las.write(output_file)
            print(f"Filtered_done for {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def merge_laz_files(input_dir: str, output_file: str) -> None:
    """
    Merges all .laz files in the input directory into a single LAZ file using PDAL.

    Parameters:
    - input_dir (str): Directory containing the input LAZ files.
    - output_file (str): Path to the output merged LAZ file.

    Returns:
    - None
    """
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    laz_files = os.path.join(os.path.join(input_dir, "*.laz"))
    print(laz_files)
    if not laz_files:
        print(f"No LAZ files found in: {input_dir}")
        return
    
    # Create the pipeline:
    # 1. Read all .laz files matching the pattern.
    # 2. Merge them.
    # 3. Write the merged output to a new file.
    pipeline_merge = (
        pdal.Reader.las(filename=laz_files)
        | pdal.Filter.merge()
        | pdal.Filter.voxeldownsize(
            cell=0.1,        
            mode="center"     
        )
        | pdal.Writer.las(filename=output_file)
    )

    print("Executing pipeline...")

    try:
        pipeline_merge.execute()
        print("File read successfully.")
    except Exception as e:
        print("Error reading LAZ file:", e)

    print(f"Merged .laz files saved to: {output_file}")


def calculate_gdal_sub_geotransform(parent_gt, x_offset_pixels, y_offset_pixels):
    """
    Calculates the geotransform for a sub-region (subset) of a raster.
    This is crucial for correctly georeferencing the debuffered tiles.

    Args:
        parent_gt (tuple): The geotransform of the parent raster.
                           Format: (top_left_x, pixel_width, row_rotation_x, top_left_y, col_rotation_y, pixel_height).
        x_offset_pixels (int): X-offset (column offset) of the sub-region's top-left corner
                               relative to the parent's top-left corner, in pixels.
        y_offset_pixels (int): Y-offset (row offset) of the sub-region's top-left corner
                               relative to the parent's top-left corner, in pixels.

    Returns:
        tuple: The new geotransform for the sub-region.
    """
    # Calculate the new top-left X and Y coordinates based on the parent's geotransform
    # and the pixel offsets.
    new_top_left_x = parent_gt[0] + x_offset_pixels * parent_gt[1] + y_offset_pixels * parent_gt[2]
    new_top_left_y = parent_gt[3] + x_offset_pixels * parent_gt[4] + y_offset_pixels * parent_gt[5]

    # The pixel size and rotation components remain the same as the parent raster.
    return (new_top_left_x, parent_gt[1], parent_gt[2], new_top_left_y, parent_gt[4], parent_gt[5])


def tile_and_rasterize_lidar(input_laz_file, output_dir, tile_length, buffer, resolution, output_type, dimension, nodata, origin_x=None, origin_y=None):
    """
    Tiles a LiDAR .laz file with a buffer using PDAL's filters.splitter,
    and then rasterizes each tile to a GeoTIFF using writers.gdal.
    These output tiles will include the buffer around their core area.

    Args:
        input_laz_file (str): Path to the input .laz file.
        output_dir (str): Directory to save the output GeoTIFF tiles (these will be buffered).
        tile_length (float): Side length of the square tiles (e.g., 1000.0 for 1km x 1km tiles).
        buffer (float): Amount of overlap to include in each tile (in ground units).
        resolution (float): Resolution of the output raster (in ground units).
        output_type (str): Aggregation method for rasterization (e.g., "mean", "min", "max").
        dimension (str): The point dimension to use for rasterization (e.g., "Z").
        nodata (float): NoData value for raster cells with no points.
        origin_x (float, optional): X origin for the tiling grid. If None, PDAL determines it.
        origin_y (float, optional): Y origin for the tiling grid. If None, PDAL determines it.

    Returns:
        bool: True if the process was successful, False otherwise.
    """

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory for buffered tiles: {output_dir}")

    # Define the PDAL pipeline as a Python dictionary.
    # 1. Reads the LAZ file
    # 2. Splits it into buffered tiles
    # 3. then writes each tile as a GeoTIFF.
    pipeline_definition = [
        {
            "type": "readers.las",
            "filename": input_laz_file
        },
        {
            "type": "filters.splitter",
            "length": tile_length, # Defines the side length of the core tile
            "buffer": buffer      # Defines the buffer around the core tile
        },
        {
            "type": "writers.gdal",
            # The '#' in the filename is a placeholder for the tile index,
            # allowing PDAL to create multiple output files.
            "filename": os.path.join(output_dir, "atlanta_buffered_tile_#.tif"),
            "gdaldriver": "GTiff",
            "resolution": resolution,
            "output_type": output_type,
            "dimension": dimension,
            "nodata": nodata
        }
    ]

    # Add origin_x and origin_y to the splitter filter if provided
    if origin_x is not None:
        pipeline_definition[1]["origin_x"] = origin_x
    if origin_y is not None:
        pipeline_definition[1]["origin_y"] = origin_y

    # Create a PDAL Pipeline object from the definition
    pipeline_json = json.dumps(pipeline_definition)

    pipeline = pdal.Pipeline(pipeline_json)
    
    print(f"Executing PDAL pipeline for {input_laz_file} to create buffered tiles...")
    try:
        # Execute the pipeline and get the number of points processed
        count = pipeline.execute()
        print(f"Successfully processed {count} points and created buffered tiles.")
        print(f"Buffered raster tiles saved to: {output_dir}")
        return True
    except pdal.PDALError as e:
        print(f"PDAL Error during tiling and rasterization: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during tiling and rasterization: {e}")
        return False


def debuffer_and_save_gdal_tile(
    buffered_raster_path: str,
    debuffered_raster_path: str,
    actual_buffer_on_left_pixels: int,
    actual_buffer_on_top_pixels: int,
    core_tile_width_pixels: int,
    core_tile_height_pixels: int
):
    """
    Clips the core (non-buffered) data from a buffered raster using GDAL
    and saves it with correct global georeferencing.

    Args:
        buffered_raster_path (str): Path to the input buffered GeoTIFF tile.
        debuffered_raster_path (str): Path to save the output debuffered GeoTIFF tile.
        actual_buffer_on_left_pixels (int): Number of buffer pixels on the left side of the buffered raster.
        actual_buffer_on_top_pixels (int): Number of buffer pixels on the top side of the buffered raster.
        core_tile_width_pixels (int): Expected width of the core (debuffered) tile in pixels.
        core_tile_height_pixels (int): Expected height of the core (debuffered) tile in pixels.

    Returns:
        str or None: Path to the debuffered tile if successful, None otherwise.
    """
    # Open the buffered source raster in read-only mode
    src_buffered_ds = gdal.Open(buffered_raster_path, gdal.GA_ReadOnly)
    if src_buffered_ds is None:
        print(f"ERROR: Could not open buffered raster: {buffered_raster_path}")
        return None

    try:
        # Get the first raster band (assuming single-band DEM)
        src_buffered_band = src_buffered_ds.GetRasterBand(1)
        if src_buffered_band is None:
            print(f"ERROR: Could not get band from {buffered_raster_path}")
            src_buffered_ds = None # Close dataset
            return None

        # Retrieve geotransform, projection, NoData value, and data type from the buffered source
        buffered_gt = src_buffered_ds.GetGeoTransform()
        buffered_proj = src_buffered_ds.GetProjection()
        no_data_value = src_buffered_band.GetNoDataValue()
        gdal_data_type = src_buffered_band.DataType

        # Read the core data from the buffered raster.
        # xoff, yoff define the top-left pixel of the window to read.
        # win_xsize, win_ysize define the width and height of the window to read.
        core_data = src_buffered_band.ReadAsArray(
            xoff=actual_buffer_on_left_pixels,
            yoff=actual_buffer_on_top_pixels,
            win_xsize=core_tile_width_pixels,
            win_ysize=core_tile_height_pixels
        )

        if core_data is None:
            print(f"ERROR: Failed to read core data from {buffered_raster_path}")
            src_buffered_ds = None
            return None

        # Optional: Check if the read data shape matches expected core dimensions.
        # This helps in debugging if buffer/tile calculations are slightly off,
        # especially for edge tiles where buffers might be truncated.
        if core_data.shape[0] != core_tile_height_pixels or core_data.shape[1] != core_tile_width_pixels:
            print(f"WARNING: Read core data shape ({core_data.shape}) does not match expected ({core_tile_height_pixels}, {core_tile_width_pixels}) for {debuffered_raster_path}. This might indicate issues with buffer/tile size calculations or edge tiles.")
            core_tile_height_pixels = core_data.shape[0]
            core_tile_width_pixels = core_data.shape[1]

        # Calculate the correct geotransform for this debuffered (core) tile.
        # This transform ensures the debuffered tile is placed correctly in global coordinates.
        final_core_geotransform = calculate_gdal_sub_geotransform(
            buffered_gt,
            actual_buffer_on_left_pixels,
            actual_buffer_on_top_pixels
        )

        # Get the GDAL driver for GeoTIFF
        driver = gdal.GetDriverByName("GTiff")
        if driver is None:
            print("ERROR: GTiff driver not available.")
            src_buffered_ds = None
            return None

        # Create the output directory for debuffered tiles if it doesn't exist
        os.makedirs(os.path.dirname(debuffered_raster_path), exist_ok=True)

        # Create the new debuffered raster dataset
        dst_ds = driver.Create(
            debuffered_raster_path,
            xsize=core_tile_width_pixels,
            ysize=core_tile_height_pixels,
            bands=1, 
            eType=gdal_data_type, # Use the same data type as the source
            options=["COMPRESS=LZW"] # Add LZW compression to the output GeoTIFF
        )
        if dst_ds is None:
            print(f"ERROR: Could not create output raster: {debuffered_raster_path}")
            src_buffered_ds = None
            return None

        # Set the geotransform and projection for the new debuffered raster
        dst_ds.SetGeoTransform(final_core_geotransform)
        dst_ds.SetProjection(buffered_proj) # Preserve the Coordinate Reference System (CRS)

        # Write the extracted core data to the new raster band
        dst_band = dst_ds.GetRasterBand(1)
        dst_band.WriteArray(core_data)
        # Set the NoData value if it exists in the source
        if no_data_value is not None:
            dst_band.SetNoDataValue(no_data_value)

        # Flush the cache and close the destination dataset to ensure data is written to disk
        dst_band.FlushCache()
        dst_ds = None # Closing the dataset saves it

        print(f"Successfully debuffered: {buffered_raster_path} -> {debuffered_raster_path}")
        return debuffered_raster_path

    except Exception as e:
        print(f"ERROR during debuffering for {buffered_raster_path} to {debuffered_raster_path}: {e}\n{traceback.format_exc()}")
        return None
    finally:
        # Ensure the source buffered dataset is closed
        if src_buffered_ds:
            src_buffered_ds = None


def merge_gdal_tiles(input_tile_dir, output_vrt_name="merged_dem.vrt", output_tif_path=OUTPUT_TIF_PATH, tile_prefix="atlanta_dem_debuffered_tile_"):
    """
    Merges GeoTIFF tiles into a single GeoTIFF using GDAL's command-line utilities.
    It first creates a Virtual Raster (VRT) and then translates it to a final GeoTIFF.

    Args:
        input_tile_dir (str): Directory containing the GeoTIFF tiles to be merged.
        output_vrt_name (str): Name for the intermediate Virtual Raster (VRT) file.
        output_tif_name (str): Name for the final merged GeoTIFF file.
        tile_prefix (str): The prefix of the tile filenames to identify them (e.g., "atlanta_dem_debuffered_tile_").

    Returns:
        bool: True if the merge was successful, False otherwise.
    """
    print("\n--- Merging tiles using GDAL ---")

    # Construct the full paths for the VRT and final TIF files
    output_vrt_path = os.path.join(input_tile_dir, output_vrt_name)

    # Find all tile files in the input directory that match the specified prefix and extension
    tile_files = [os.path.join(input_tile_dir, f) for f in os.listdir(input_tile_dir) if f.startswith(tile_prefix) and f.endswith(".tif")]

    if not tile_files:
        print(f"No tiles found in {input_tile_dir} with prefix '{tile_prefix}'. Skipping merge.")
        return False

    # 1: Create a VRT (Virtual Raster) from the tiles
    # gdalbuildvrt command: gdalbuildvrt <output_vrt> <input_files...>
    gdalbuildvrt_command = [
        "gdalbuildvrt",
        output_vrt_path,
    ] + tile_files # Append the list of found tile files to the command

    print(f"Executing: {' '.join(gdalbuildvrt_command)}")
    try:
        # Run the gdalbuildvrt command. check=True raises an exception for non-zero exit codes.
        subprocess.run(gdalbuildvrt_command, check=True, capture_output=True, text=True)
        print(f"Successfully created VRT: {output_vrt_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating VRT: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: 'gdalbuildvrt' command not found. Make sure GDAL is installed and in your system's PATH.")
        return False

    # 2: Convert the VRT to a single GeoTIFF
    # gdal_translate command: gdal_translate <vrt_file> <output_tif_file>
    gdal_translate_command = [
        "gdal_translate",
        output_vrt_path,
        output_tif_path,
        "-co", "COMPRESS=LZW" # Add LZW compression to the final output GeoTIFF
    ]
    print(f"Executing: {' '.join(gdal_translate_command)}")
    try:
        # Run the gdal_translate command
        subprocess.run(gdal_translate_command, check=True, capture_output=True, text=True)
        print(f"Successfully created merged GeoTIFF: {output_tif_path}")
        # Remove the temporary VRT file after successful merging
        os.remove(output_vrt_path)
        print(f"Removed temporary VRT: {output_vrt_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting VRT to GeoTIFF: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: 'gdal_translate' command not found. Make sure GDAL is installed and in your system's PATH.")
        return False

def reproject_raster(
    input_raster: str,
    output_raster: str,
    target_epsg: int,
    overwrite: bool = True
    ) -> None:
    """
    Reprojects a raster from one EPSG CRS to another using gdalwarp.

    Parameters:
    - input_raster (str): Path to the input raster file.
    - output_raster (str): Path to the output raster file.
    - target_epsg (int): EPSG code for the target CRS (default: 6446).
    - overwrite (bool): Whether to overwrite the output file if it exists.

    Returns:
    - None
    """
    if not overwrite and os.path.exists(output_raster):
        raise FileExistsError(f"Output raster already exists and overwrite=False: {output_raster}")
    
    
    if not os.path.exists(input_raster):
        raise FileNotFoundError(f"Input raster does not exist: {input_raster}")
    
    with rasterio.open(input_raster) as src:
        src_crs = src.crs
        transform, width, height = calculate_default_transform(
            src_crs, target_epsg, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_epsg,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_raster, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst,i),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=target_epsg,
                    resampling=Resampling.nearest)

        print(f"Reprojection complete: {output_raster}")
    # if not overwrite and os.path.exists(output_raster):
    #     raise FileExistsError(f"Output raster already exists and overwrite=False: {output_raster}")
    
    

    # cmd = [
    #     "gdalwarp",
    #     "-r", resampling,
    #     "-s_srs", f"EPSG:{source_epsg}",
    #     "-t_srs", f"EPSG:{target_epsg}",
    #     "-of", output_format,
    # ]

    # if overwrite:
    #     cmd.append("-overwrite")

    # cmd.extend([input_raster, output_raster])

    # try:
    #     subprocess.run(cmd, check=True)
    #     print(f"Reprojection complete: {output_raster}")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during reprojection: {e}")


def process_window(
    ji_window: Tuple[int, Window],
    input_path: str,
    no_data_value: float,
    method: str,
    fill_value: float,
    buffer: int,
    invalid_lt: float
) -> Tuple[int, Window, np.ndarray]:
    """
    Interpolates invalid data within a single window of a raster.

    This function is designed to be called by a multiprocessing Pool. It reads
    a buffered window, identifies valid and invalid pixels, and uses vectorized
    `griddata` to perform interpolation.

    Args:
        ji_window: A tuple containing the window index and a rasterio Window object.
        input_path: Path to the source raster file.
        no_data_value: The value representing no data in the raster.
        method: Interpolation method to use ('linear', 'nearest', 'cubic').
        fill_value: The value to use for pixels that cannot be interpolated.
        buffer: The buffer size (in pixels) to add around the window to avoid edge effects.
        invalid_lt: Values less than this number are considered invalid.

    Returns:
        A tuple containing the window index, the original window, and the
        interpolated numpy array for that window.
    """
    ji, window = ji_window

    # Open the source raster inside each worker process for process safety
    with rasterio.open(input_path) as src:
        height, width = src.height, src.width

        # Create a buffered read-window to avoid edge effects during interpolation
        rs = max(0, window.row_off - buffer)
        re = min(height, window.row_off + window.height + buffer)
        cs = max(0, window.col_off - buffer)
        ce = min(width, window.col_off + window.width + buffer)
        read_w = Window(cs, rs, ce - cs, re - rs)

        data = src.read(1, window=read_w, boundless=True).astype(np.float32)

        # Define masks for valid and invalid data points
        # Invalid points are no_data, less than a threshold, or NaN.
        invalid_mask = (
            (data == no_data_value) |
            (data < invalid_lt) |
            np.isnan(data)
        )
        valid_mask = ~invalid_mask

        # If the buffered window contains no valid data, fill the whole window
        # with the fill_value and return.
        if not valid_mask.any():
            out = np.full((window.height, window.width), fill_value, dtype=np.float32)
            return ji, window, out

        # Get the coordinates and values of all valid points
        rows, cols = np.indices(data.shape)
        valid_pts = np.column_stack((rows[valid_mask], cols[valid_mask]))
        valid_vals = data[valid_mask]

        # Get the coordinates of all invalid points that need to be filled
        interp_pts = np.column_stack((rows[invalid_mask], cols[invalid_mask]))
        
        # Create a copy of the data to hold the interpolated values
        filled = data.copy()

        # *** CORE OPTIMIZATION ***
        # Instead of looping through each point, we pass all points to griddata
        # at once. This is massively faster as it uses SciPy's vectorized C-backend.
        if interp_pts.size > 0:
             interpolated_values = griddata(
                points=valid_pts,
                values=valid_vals,
                xi=interp_pts,
                method=method,
                fill_value=fill_value
            )
             filled[invalid_mask] = interpolated_values

        # Extract the central, non-buffered part of the processed window
        row_offset = window.row_off - rs
        col_offset = window.col_off - cs
        out = filled[row_offset:row_offset + window.height, col_offset:col_offset + window.width]

        return ji, window, out

def interpolate_tif_mp(
    input_tif: str,
    output_tif: str,
    invalid_lt: float,
    no_data_value: int = -9999,
    method: str = 'linear',
    fill_value: int = -9999,
    search_radius: int = 50,
    tile_size: int = 500,
    n_workers: int = 4,
):
    """
    Interpolates a TIF file using multiprocessing.

    Args:
        input_tif: Path to the input TIF file.
        output_tif: Path for the output interpolated TIF file.
        invalid_lt: Values less than this will be treated as invalid.
        no_data_value: The no-data value in the source raster.
        method: Interpolation method ('linear', 'nearest', 'cubic').
        fill_value: Value to fill in where interpolation is not possible.
        search_radius: Buffer size around each tile for seamless interpolation.
        tile_size: The size of tiles to process in parallel.
        n_workers: Number of worker processes to use.
    """
    with rasterio.open(input_tif) as src:
        height, width = src.height, src.width
        profile = src.profile.copy()
        profile.update(dtype=np.float32, nodata=fill_value)

        windows = []
        idx = 0
        for row_off in range(0, height, tile_size):
            for col_off in range(0, width, tile_size):
                win_h = min(tile_size, height - row_off)
                win_w = min(tile_size, width - col_off)
                win = Window(col_off, row_off, win_w, win_h)
                windows.append((idx, win))
                idx += 1
        print(f"Created {len(windows)} windows of up to {tile_size}x{tile_size} pixels.")

    # Use functools.partial to freeze parameters for the worker function
    worker_fn = partial(
        process_window,
        input_path=input_tif,
        no_data_value=no_data_value,
        method=method,
        fill_value=fill_value,
        buffer=search_radius,
        invalid_lt=invalid_lt,
    )

    # Open the destination file for writing
    with rasterio.open(output_tif, 'w', **profile) as dst:
        # Use a multiprocessing Pool
        with Pool(n_workers) as pool:
            for _, win, data in tqdm(
                pool.imap_unordered(worker_fn, windows),
                total=len(windows),
                desc="Interpolating windows"
            ):
                dst.write(data.astype(profile['dtype']), 1, window=win)

    print(f"\nDone. Interpolated file saved to: {output_tif}")


if __name__ == "__main__":
    filter_lidar_by_classification(LIDAR_DIR, FILTERED_LIDAR_DIR, FILTER_CLASS)
    merge_laz_files(LIDAR_TO_MERGE, LIDAR_MERGED)
    # Ensure output directories exist before starting the process
    os.makedirs(BUFFERED_TILES_DIR, exist_ok=True)
    os.makedirs(INTERMEDIATE_DEBUFFERED_DIR, exist_ok=True)

    # --- Step 1: Tile LAZ and Rasterize to Buffered GeoTIFFs ---
    print("\n--- Step 1: Tiling LAZ and Rasterizing to Buffered GeoTIFFs ---")
    tiling_successful = tile_and_rasterize_lidar(
        input_laz_file=INPUT_LIDAR,
        output_dir=BUFFERED_TILES_DIR, # Buffered tiles will be saved here
        tile_length=TILE_LENGTH,
        buffer=BUFFER,
        resolution=RESOLUTION,
        output_type=OUTPUT_TYPE,
        dimension=DIM,
        nodata=NODATA
    )

    if tiling_successful:
        # --- Step 2: Debuffer each GeoTIFF tile ---
        print("\n--- Step 2: Debuffering each GeoTIFF tile ---")
        # Define prefixes for the buffered and debuffered tiles for easy identification
        buffered_tile_prefix = "atlanta_buffered_tile_" # Matches the filename used in writers.gdal
        debuffered_tile_prefix = "atlanta_debuffered_tile_" # New prefix for the debuffered tiles

        # Calculate buffer and core tile dimensions in pixels based on ground units and resolution
        buffer_pixels = int(BUFFER / RESOLUTION)
        core_tile_width_pixels = int(TILE_LENGTH / RESOLUTION)
        core_tile_height_pixels = int(TILE_LENGTH / RESOLUTION)

        debuffering_successful_count = 0
        total_buffered_tiles = 0

        # Iterate through all files in the directory where buffered tiles were saved
        for filename in os.listdir(BUFFERED_TILES_DIR):
            # Check if the file is a buffered GeoTIFF tile
            if filename.startswith(buffered_tile_prefix) and filename.endswith(".tif"):
                total_buffered_tiles += 1
                buffered_tile_path = os.path.join(BUFFERED_TILES_DIR, filename)

                # Construct the new filename for the debuffered tile
                debuffered_filename = filename.replace(buffered_tile_prefix, debuffered_tile_prefix)
                debuffered_tile_path = os.path.join(INTERMEDIATE_DEBUFFERED_DIR, debuffered_filename)

                # Call the debuffering function for the current tile
                debuffered_result = debuffer_and_save_gdal_tile(
                    buffered_raster_path=buffered_tile_path,
                    debuffered_raster_path=debuffered_tile_path,
                    actual_buffer_on_left_pixels=buffer_pixels,
                    actual_buffer_on_top_pixels=buffer_pixels,
                    core_tile_width_pixels=core_tile_width_pixels,
                    core_tile_height_pixels=core_tile_height_pixels
                )
                if debuffered_result:
                    debuffering_successful_count += 1
                else:
                    print(f"Failed to debuffer tile: {buffered_tile_path}")

        # Check the overall success of the debuffering step
        if total_buffered_tiles > 0 and debuffering_successful_count == total_buffered_tiles:
            print(f"Successfully debuffered all {debuffering_successful_count} tiles.")
            debuffering_overall_successful = True
        elif total_buffered_tiles == 0:
            print("No buffered tiles found to debuffer. Skipping debuffering and merging.")
            debuffering_overall_successful = False
        else:
            print(f"Completed debuffering with {debuffering_successful_count}/{total_buffered_tiles} tiles successfully processed. Please check the logs above for specific errors.")
            debuffering_overall_successful = True # Indicate partial or full failure

        # --- Step 3: Merge the debuffered tiles ---

        if debuffering_overall_successful:
            print("\n--- Step 3: Merging debuffered tiles ---")
            merge_gdal_tiles(
                input_tile_dir=INTERMEDIATE_DEBUFFERED_DIR, # Merge tiles from the debuffered directory
                output_vrt_name="atlanta_merged_dem.vrt", # Custom VRT name for the final merge
                output_tif_path=OUTPUT_TIF_PATH, # Custom TIF name for the final merged output
                tile_prefix=debuffered_tile_prefix # Specify the prefix for the debuffered tiles to merge
            )
        
        # --- Step 4: Reproject the output file ---
        output_dir = os.path.dirname(OUTPUT_TIF_PATH)
        file_name = os.path.basename(OUTPUT_TIF_PATH)
        filename_wo_extension = os.path.splitext(file_name)[0]
        OUTPUT_TIF_REPROJECTED = os.path.join(output_dir, f"{filename_wo_extension}_reprojected.tif")

        if os.path.exists(OUTPUT_TIF_PATH):
            print(f"Reprojecting {file_name}...")
            reproject_raster(OUTPUT_TIF_PATH, OUTPUT_TIF_REPROJECTED, target_epsg = DEST_CRS)

        else:
            print("Skipping final tile merging due to errors or no tiles found during debuffering.")
    else:
        print("Skipping debuffering and merging steps due to errors during the initial tiling and rasterization.")
