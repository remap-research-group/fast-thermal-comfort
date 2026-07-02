import psutil, os
from multiprocessing import Process, Queue, cpu_count
import multiprocessing as mp
from queue import Empty
import shutil
from osgeo import gdal, ogr, osr
import tempfile
from IPython.display import clear_output
import os
import time
from glob import glob
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import sqlite3

data_dir = '/storage/project/r-rbasu31-0/hyu483/Test_UMEP/UROCK/input'
output_dir = '/storage/project/r-rbasu31-0/hyu483/Test_UMEP/UROCK_output_35_spawn'
	
## Need to process to convert geojson to shp file for tiling ##
buildings_path = os.path.join(data_dir, 'building_26986.shp')
veg_path = os.path.join(data_dir, 'veg_26986.shp')
template_rast_path = os.path.join(data_dir, "../DSM_26986_2.tif")
building_height_col = 'bd_h'
veg_height_col = 'mean_h'

# --- Configuration for Tiling ---
TILE_SIZE = 200 # Size of each tile in map units (e.g., meters). This is the unbuffered size.
BUFFER_SIZE = 200 # Buffer distance around each tile for URock calculation, in map units.
                  # This ensures features near tile edges are considered.

# Define a base directory for all intermediate temporary files
BASE_TEMP_DIR = os.path.join(output_dir, "temp_intermediate_files")
output_base_dir = os.path.join(output_dir, "urock_tiled_output")
os.makedirs(output_base_dir, exist_ok=True)

def get_layer_srs(layer):
    """
    Retrieves the Spatial Reference System (SRS) from an OGR layer.
    Args:
        layer (ogr.Layer): The OGR layer object.
    Returns:
        osr.SpatialReference: The SRS object, or None if not found.
    """
    srs = layer.GetSpatialRef()
    if srs:
        return srs
    return None

def debuffer_and_save_raster_gdal(input_raster_path, output_raster_path, original_extent_tuple):
    """
    Crops an input raster to remove the buffer, extracting only the area corresponding
    to the original (unbuffered) tile extent.
    Args:
        input_raster_path (str): Path to the buffered raster output from URock.
        output_raster_path (str): Path where the debuffered raster will be saved.
        original_extent_tuple (tuple): The (minX, minY, maxX, maxY) of the *unbuffered* tile area.
    Returns:
        str: Path to the debuffered raster if successful, None otherwise.
    """
    src_ds = gdal.Open(input_raster_path, gdal.GA_ReadOnly)
    if src_ds is None:
        print(f"Error: Could not open input raster '{input_raster_path}' for debuffering.")
        return None

    # The URock output raster covers the buffered area. We want to clip it
    # back to the unbuffered original extent.
    minX_orig, minY_orig, maxX_orig, maxY_orig = original_extent_tuple

    try:
        gdal.Translate(
            output_raster_path,
            src_ds,
            projWin=[minX_orig, maxY_orig, maxX_orig, minY_orig] # GDAL order: (ulx, uly, lrx, lry)
        )
        print(f"Info: Debuffered '{input_raster_path}' to '{output_raster_path}' (original extent: {original_extent_tuple}).")
        if os.path.exists(output_raster_path):
            return output_raster_path
    except Exception as e:
        print(f"Error debuffering raster {input_raster_path}: {e}")
    finally:
        src_ds = None # Close dataset
    return None

def gdal_slice_vector_by_window(input_shp_path, output_shp_path, extent_tuple, target_srs_wkt=None):
    """
    Slices a vector shapefile by a given bounding box extent, cutting geometries.
    Optionally reprojects features to a target SRS.
    Args:
        input_shp_path (str): Path to the input shapefile.
        output_shp_path (str): Path where the sliced output shapefile will be saved.
        extent_tuple (tuple): A tuple (minX, minY, maxX, maxY) defining the clipping window.
        target_srs_wkt (str, optional): Well-Known Text (WKT) of the target SRS.
                                        If provided, features will be reprojected to this SRS.
                                        If None, the input shapefile's SRS is preserved.
    Returns:
        str: Path to the output shapefile if successful, None otherwise.
    """
    driver = ogr.GetDriverByName("ESRI Shapefile")
    source_ds = ogr.Open(input_shp_path, 0) # 0 means read-only
    if source_ds is None:
        print(f"Error: Could not open input shapefile {input_shp_path}")
        return None
    source_layer = source_ds.GetLayer()

    if source_layer is None:
        print(f"Error: No layer found in {input_shp_path}")
        source_ds = None
        return None

    # Get source SRS and set up coordinate transformation if a target SRS is provided
    source_srs = get_layer_srs(source_layer)
    coord_trans = None
    target_srs = None # Initialize target_srs here
    if target_srs_wkt and source_srs:
        target_srs = osr.SpatialReference()
        target_srs.ImportFromWkt(target_srs_wkt)
        # Only create a transformation if the source and target SRS are different
        if not source_srs.IsSame(target_srs):
            coord_trans = osr.CoordinateTransformation(source_srs, target_srs)
            print(f"Info: Created coordinate transformation for '{input_shp_path}' to target SRS.")
    
    # Create clipping polygon from the extent_tuple
    minX, minY, maxX, maxY = extent_tuple
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minX, minY)
    ring.AddPoint(maxX, minY)
    ring.AddPoint(maxX, maxY)
    ring.AddPoint(minX, maxY)
    ring.AddPoint(minX, minY) # Close the ring
    clip_polygon = ogr.Geometry(ogr.wkbPolygon)
    clip_polygon.AddGeometry(ring)

    # Delete output shapefile if it already exists (important for overwriting)
    if os.path.exists(output_shp_path):
        driver.DeleteDataSource(output_shp_path)
    
    out_ds = driver.CreateDataSource(output_shp_path)
    if out_ds is None:
        print(f"Error: Could not create output shapefile {output_shp_path}")
        source_ds = None
        return None
    
    # Determine the SRS for the output layer (target SRS if reprojecting, otherwise source SRS)
    out_srs = target_srs if target_srs_wkt else source_srs
    out_layer = out_ds.CreateLayer(
        os.path.basename(output_shp_path).split('.')[0], # Layer name from filename
        srs=out_srs,
        geom_type=source_layer.GetGeomType() # Keep original geometry type for now
    )
    
    # Copy all field definitions from the source layer to the output layer
    source_layer_defn = source_layer.GetLayerDefn()
    for i in range(source_layer_defn.GetFieldCount()):
        field_defn = source_layer_defn.GetFieldDefn(i)
        out_layer.CreateField(field_defn)

    # Set a spatial filter on the source layer using the provided extent
    # This helps to pre-filter features for efficiency before intersection
    source_layer.SetSpatialFilterRect(minX, minY, maxX, maxY)

    # Iterate over features that intersect the filter extent and clip them
    feature_count = 0
    for feature in source_layer:
        geom = feature.GetGeometryRef()
        if geom:
            # Transform the original geometry to the output SRS *before* clipping if reprojecting
            if coord_trans:
                geom_for_clipping = geom.Clone()
                geom_for_clipping.Transform(coord_trans)
            else:
                geom_for_clipping = geom.Clone() # Clone to avoid modifying original

            # Perform the actual clipping
            clipped_geom = geom_for_clipping.Intersection(clip_polygon)
            
            # Only add the feature if clipping resulted in a valid geometry
            if clipped_geom and not clipped_geom.IsEmpty():
                # If the original geometry type was e.g., wkbPolygon, and clipping resulted
                # in a wkbMultiPolygon (e.g., if a polygon was split into two by the clip extent),
                # GDAL/OGR will handle it, but it's good practice to be aware.
                # If the output layer was created with a strict single geometry type,
                # you might need to handle multi-part geometries resulting from intersection.
                # For simplicity here, we assume the output driver can handle mixed types or
                # the result of intersection (e.g., a Polygon clipped by a box might become a MultiPolygon).
                # If strict type is required, you'd iterate parts of a MultiGeometry.

                new_feature = ogr.Feature(out_layer.GetLayerDefn())
                new_feature.SetGeometry(clipped_geom)
                
                # Copy field attributes
                for i in range(new_feature.GetFieldCount()):
                    new_feature.SetField(new_feature.GetFieldDefnRef(i).GetNameRef(), feature.GetField(i))
                
                out_layer.CreateFeature(new_feature)
                new_feature = None # Release feature resources
                feature_count += 1
    
    print(f"Info: Sliced and clipped '{input_shp_path}' to '{output_shp_path}'. Copied {feature_count} features.")
    source_ds = None # Close source dataset
    out_ds = None    # Close output dataset
    return output_shp_path
    
def gdal_slice_raster_by_window(input_raster_path, output_raster_path, extent_tuple):
    """
    Slices a raster by a given bounding box extent using gdal.Translate.
    Args:
        input_raster_path (str): Path to the input raster.
        output_raster_path (str): Path for the output sliced raster.
        extent_tuple (tuple): A tuple (minX, minY, maxX, maxY) defining the clipping window.
    Returns:
        str: Path to the output raster if successful, None otherwise.
    """
    minX, minY, maxX, maxY = extent_tuple
    
    try:
        gdal.Translate(
            output_raster_path,
            input_raster_path,
            projWin=[minX, maxY, maxX, minY] 
        )
        print(f"Info: Sliced '{input_raster_path}' to '{output_raster_path}' (Extent: {extent_tuple}).")
        if os.path.exists(output_raster_path):
            return output_raster_path
    except Exception as e:
        print(f"Error slicing raster {input_raster_path}: {e}")
    return None
    
def calculate_tile_extents(template_raster_path, tile_size, buffer_size):
    """
    Calculates the unbuffered and buffered extents for a grid of tiles
    that cover the entire template raster.
    Args:
        template_raster_path (str): Path to the template raster, used to determine the full extent.
        tile_size (float): The desired side length of each unbuffered tile (e.g., in meters).
        buffer_size (float): The buffer distance to apply around each unbuffered tile.
    Returns:
        list of tuples: Each tuple contains (tile_id, unbuffered_extent, buffered_extent).
                        An empty list is returned if the template raster cannot be opened.
    """
    src_ds = gdal.Open(template_raster_path, gdal.GA_ReadOnly)
    if src_ds is None:
        print(f"Error: Could not open template raster '{template_raster_path}' to calculate tile extents.")
        return []

    # Get the full extent of the template raster
    gt = src_ds.GetGeoTransform()
    minX_full = gt[0]
    maxY_full = gt[3]
    maxX_full = gt[0] + src_ds.RasterXSize * gt[1] # gt[1] is x-resolution
    minY_full = gt[3] + src_ds.RasterYSize * gt[5] # gt[5] is y-resolution (negative for north-up)

    full_extent = (minX_full, minY_full, maxX_full, maxY_full)
    print(f"Full template raster extent: {full_extent}")

    tile_extents_list = []
    tile_id = 0

    # Generate the starting X coordinates for each tile
    # We iterate from minX_full up to (but not exceeding) maxX_full
    current_x = minX_full
    while current_x < maxX_full:
        current_y = minY_full
        # Generate the starting Y coordinates for each tile
        while current_y < maxY_full:
            # Calculate the unbuffered tile extent for the current grid cell
            unbuffered_minX = current_x
            unbuffered_minY = current_y
            unbuffered_maxX = min(current_x + tile_size, maxX_full) # Clip to full extent
            unbuffered_maxY = min(current_y + tile_size, maxY_full) # Clip to full extent

            unbuffered_extent = (unbuffered_minX, unbuffered_minY, unbuffered_maxX, unbuffered_maxY)

            # Calculate the buffered extent
            buffered_minX = max(minX_full, unbuffered_minX - buffer_size) # Ensure buffer doesn't go outside full extent
            buffered_minY = max(minY_full, unbuffered_minY - buffer_size)
            buffered_maxX = min(maxX_full, unbuffered_maxX + buffer_size)
            buffered_maxY = min(maxY_full, unbuffered_maxY + buffer_size)
            buffered_extent = (buffered_minX, buffered_minY, buffered_maxX, buffered_maxY)

            tile_extents_list.append((tile_id, unbuffered_extent, buffered_extent))
            tile_id += 1
            current_y += tile_size
        current_x += tile_size
    
    src_ds = None # Close the GDAL dataset
    print(f"Calculated {len(tile_extents_list)} tiles for processing.")
    return tile_extents_list

def urock_test(buildings_input, veg_input, template_input, output_dir, out_file):
    from qgis import processing
    """
    Wrapper function to run the URock processing algorithm.
    This function is designed to be called by each multiprocessing worker.
    Args:
        buildings_input (str): Path to the input buildings shapefile (for the current tile).
        veg_input (str): Path to the input vegetation shapefile (for the current tile).
        template_input (str): Path to the template raster for URock output (for the current tile).
        output_dir (str): Directory where URock will save its output files for this tile.
        out_file (str): Base filename for URock's output files (e.g., 'wind_field_tile_X').
    """
    start_time = time.time()
    print(f"Process {os.getpid()}: Running URock for tile output '{out_file}'...", flush=True)
    
    try:
        # Ensure the output directory exists for URock
        os.makedirs(output_dir, exist_ok=True)
        # Call the QGIS processing.run function
        # Note: 'SAVE_NETCDF' was commented out in your original code with a note
        # saying it 'must be true to work'. Please adjust this parameter if necessary.
        processing.run("umep:Urban Wind Field: URock", {
            'BUILDINGS': buildings_input,
            'HEIGHT_FIELD_BUILD':building_height_col,
            'VEGETATION': veg_input,
            'VEGETATION_CROWN_TOP_HEIGHT':veg_height_col,
            'VEGETATION_CROWN_BASE_HEIGHT':'',
            'ATTENUATION_FIELD':'','INPUT_PROFILE_FILE':'',
            'INPUT_WIND_HEIGHT':10,
            'INPUT_WIND_SPEED':2,
            'INPUT_WIND_DIRECTION':45,
            'RASTER_OUTPUT': template_input, # Template raster for output extent/resolution
            'HORIZONTAL_RESOLUTION':3,
            'VERTICAL_RESOLUTION':3,
            'WIND_HEIGHT':'1.5',
            'UROCK_OUTPUT': output_dir,
            'OUTPUT_FILENAME': out_file,
            'SAVE_RASTER':True, # Save the raster output
            'SAVE_VECTOR':False, # Save the vector output
            'SAVE_NETCDF':False, # Set to True if this is truly required by UMEP
            'LOAD_OUTPUT':False
        })
        print(f"Process {os.getpid()}: Successfully ran URock for tile output '{out_file}'.", flush=True)
    except Exception as e:
        print(f"Process {os.getpid()}: ERROR running URock for tile output '{out_file}': {e}", flush=True)
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Process {os.getpid()}: Elapsed time for URock ('{out_file}'): {elapsed:.2f} seconds.", flush=True)

def process_single_tile(tile_info_tuple):
    # from qgis import processing
    """
    This is the core function executed by each worker process in the multiprocessing pool.
    It orchestrates the slicing of input data, running URock, and debuffering the output for a single tile.
    Args:
        tile_info_tuple (tuple): A tuple containing:
            - tile_id (int): Unique identifier for the tile.
            - unbuffered_extent (tuple): (minX, minY, maxX, maxY) of the original tile area.
            - buffered_extent (tuple): (minX, minY, maxX, maxY) of the tile area with buffer.
            - global_paths_dict (dict): A dictionary containing paths to the original (full) input datasets
                                        and the base output directory.
    """

    tile_id, unbuffered_extent, buffered_extent, global_paths = tile_info_tuple

    
    # Extract original paths from the global_paths_dict
    buildings_orig_path = global_paths['buildings']
    veg_orig_path = global_paths['veg']
    template_rast_orig_path = global_paths['template_rast']
    base_output_dir = global_paths['base_output_dir']
    
    # # Initialize QGIS processing for this specific worker process.
    # init_qgis_processing()
    
    temp_input_files_dir = os.path.join(BASE_TEMP_DIR, f"tile_{tile_id}_sliced_inputs")
    os.makedirs(temp_input_files_dir, exist_ok=True)
    temp_urock_raw_output_dir = os.path.join(BASE_TEMP_DIR, f"tile_{tile_id}_urock_raw_outputs")
    os.makedirs(temp_urock_raw_output_dir, exist_ok=True)
        
    # Define temporary paths for the sliced input files (these will be in the temp_dir)
    temp_buildings_shp = os.path.join(temp_input_files_dir, f"buildings_tile_{tile_id}.shp")
    temp_veg_shp = os.path.join(temp_input_files_dir, f"veg_tile_{tile_id}.shp")
    temp_template_rast_tif = os.path.join(temp_input_files_dir, f"template_tile_{tile_id}.tif")

    # Define a temporary output directory within the tile's temp_dir for URock's outputs
    urock_output_filename = f"wind_field_raw_tile_{tile_id}"
    
    # Define the final path for the debuffered raster output, which will go to the main output directory
    final_output_raster_path = os.path.join(base_output_dir, f"urock_wind_field_tile_{tile_id}WS.tif")
    urock_raw_raster_output_path = os.path.join(temp_urock_raw_output_dir, f"z1_5/{urock_output_filename}WS.tif")
    # If URock also generates vector outputs you want to keep, you'd define similar paths for them.
    if os.path.exists(final_output_raster_path):
        print(f"Process {os.getpid()} - Tile {tile_id}: Final output already exists at {final_output_raster_path}", flush=True)
        return f"Process {os.getpid()} - Tile {tile_id}: Final output already exists."
    if os.path.exists(urock_raw_raster_output_path):
        print(f"Process {os.getpid()} - Tile {tile_id}: Found existing URock output, debuffering...", flush=True)
        try:
            debuffer_and_save_raster_gdal(urock_raw_raster_output_path, final_output_raster_path, unbuffered_extent)
            return f"Process {os.getpid()} - Tile {tile_id}: Successfully processed existing URock output."
        except Exception as e:
            return f"Tile {tile_id}: Failed to process existing URock output - {e}"
    try:
        # Step 1: Get the SRS from the original template raster. This ensures consistency
        # if vector layers need to be reprojected during slicing.
        template_ds_full = gdal.Open(template_rast_orig_path, gdal.GA_ReadOnly)
        if template_ds_full is None:
            raise Exception(f"Failed to open original template raster {template_rast_orig_path} for SRS.")
        template_srs_wkt = template_ds_full.GetProjection()
        template_ds_full = None # Close dataset immediately after getting info

        # Step 2: Slice the original vector layers (buildings and vegetation) to the buffered extent.
        # The output shapefiles are created in the tile's temporary directory.
        buildings_sliced_path = gdal_slice_vector_by_window(buildings_orig_path, temp_buildings_shp, buffered_extent, template_srs_wkt)
        veg_sliced_path = gdal_slice_vector_by_window(veg_orig_path, temp_veg_shp, buffered_extent, template_srs_wkt)

        if not buildings_sliced_path or not veg_sliced_path:
            print(f"Process {os.getpid()} - Tile {tile_id}: Skipping URock, failed to slice input vector data.", flush=True)
            return # Exit if slicing fails

        # Check if the sliced building file actually has features
        # Running URock on an empty area is a common cause for hanging
        check_ds = gdal.OpenEx(buildings_sliced_path)
        lyr = check_ds.GetLayer()
        feat_count = lyr.GetFeatureCount()
        check_ds = None # Close

        # Check vegettion 
        check_ds_2 = gdal.OpenEx(veg_sliced_path)
        lyr_2 = check_ds_2.GetLayer()
        feat_count_veg = lyr_2.GetFeatureCount()
        check_ds_2 = None # Close veg

        if feat_count == 0:
            print(f"Process {os.getpid()} - Tile {tile_id}: Skipping URock,- No buildings in this tile.", flush=True)
            return

        # Step 3: Slice the original template raster to the buffered extent.
        template_sliced_path = gdal_slice_raster_by_window(template_rast_orig_path, temp_template_rast_tif, buffered_extent)
        if not template_sliced_path:
            print(f"Process {os.getpid()} - Tile {tile_id}: Skipping URock, failed to slice template raster.", flush=True)
            return # Exit if slicing fails

        # Step 4: Run the URock algorithm using the sliced (buffered) input data.
        # URock will save its raw outputs to temp_urock_output_dir.
        print(f"Tile {tile_id}: Running URock...", flush=True)
        try:
            if feat_count_veg != 0:
                urock_test(buildings_sliced_path, veg_sliced_path, template_sliced_path, temp_urock_raw_output_dir, urock_output_filename)
            else:
                urock_test(buildings_sliced_path, '', template_sliced_path, temp_urock_raw_output_dir, urock_output_filename)
        except Exception as urock_err:
            return f"Tile {tile_id}: URock Internal Error - {str(urock_err)}"

        # Step 5: After URock runs, locate its raw raster output.
        # URock usually names the raster output as {OUTPUT_FILENAME}.tif
               
        if os.path.exists(urock_raw_raster_output_path):
            # Step 6: Debuffer the URock raster output to extract only the unbuffered area.
            debuffer_and_save_raster_gdal(urock_raw_raster_output_path, final_output_raster_path, unbuffered_extent)
        else:
            print(f"Process {os.getpid()} - Tile {tile_id}: URock raster output not found at '{urock_raw_raster_output_path}'. Cannot debuffer.", flush=True)
            
        time.sleep(0.1)
        
        # if os.path.exists(temp_input_files_dir):
        #     try:
        #         shutil.rmtree(temp_input_files_dir)
        #         print(f"Process {os.getpid()} - Tile {tile_id}: Cleaned up temporary directory '{temp_input_files_dir}'.", flush=True)
        #     except OSError as e:
        #         print(f"Process {os.getpid()} - Tile {tile_id}: Error removing temporary directory '{temp_input_files_dir}': {e}", flush=True)
        return (f"Process {os.getpid()} - Tile {tile_id}: Finished processing.")
    except Exception as e:
        print(f"Process {os.getpid()} - Tile {tile_id}: An unhandled error occurred: {e}", flush=True)
        return f"Tile {tile_id}: Failed - {e}"
    finally:
        import gc
        gc.collect()


def worker_loop(task_q, result_q, base_temp_dir):
    # The 'spawn' method starts a clean slate, so all imports
    import os, sys, time
    from queue import Empty

    # 1. Setup Environment
    # IMPORTANT: set per-worker config BEFORE QGIS init
    worker_profile = os.path.join(base_temp_dir, f"qgis_profile_{os.getpid()}")
    os.makedirs(worker_profile, exist_ok=True)
    os.environ["QGIS_CUSTOM_CONFIG_PATH"] = worker_profile
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    profiles_path = '/storage/home/hcoda1/4/hyu483/qgis_332/'
    sys.path.append(f"{profiles_path}/profiles/default/python/plugins/")

    # 2. QGIS Initialization
    from qgis.core import QgsApplication
    QgsApplication.setPrefixPath("/storage/home/hcoda1/4/hyu483/conda_envs/qgis332", True)

    app = QgsApplication([], False)
    app.initQgis()

    try:
        # 3. Import and Init Processing
        import processing
        from processing.core.Processing import Processing
        Processing.initialize()

        from processing_umep.processing_umep_provider import ProcessingUMEPProvider
        umep_provider = ProcessingUMEPProvider()
        QgsApplication.processingRegistry().addProvider(umep_provider)

        while True:
            try:
                task = task_q.get(timeout=30)
                if task is None:
                    result_q.put(f"{os.getpid()}: exit signal")
                    break
                print(task)
                result = process_single_tile(task)
                result_q.put(result)

            except Empty:
                result_q.put(f"{os.getpid()}: timeout waiting for tasks")
                break
            except Exception as e:
                result_q.put(f"{os.getpid()}: Worker Error: {e}")
                break

    finally:
        # ONLY ONCE per worker
        app.exitQgis()
    

def main():
    """
    Main execution function.
    Orchestrates the entire multiprocessing workflow:
    1. Sets up the main output directory.
    2. Calculates all tile extents (unbuffered and buffered).
    3. Prepares arguments for each worker process.
    4. Uses a multiprocessing Pool to run `process_single_tile` in parallel.
    """
    # # Define the base directory where all final debuffered tile outputs will be saved

    # 
    mp.set_start_method('spawn', force=True)
    # Step 1: Calculate all unbuffered and buffered tile extents based on the template raster.
    print("\nStep 1: Calculating tile extents...")
    tile_definitions = calculate_tile_extents(template_rast_path, TILE_SIZE, BUFFER_SIZE)
    if not tile_definitions:
        print("No tiles were generated. Please check input paths and tile size configuration.",)
        return

    # Step 2: Prepare the arguments that will be passed to each `process_single_tile` worker.
    # We pass global paths to the original datasets so each worker can slice them.
    global_paths_for_workers = {
        'buildings': buildings_path,
        'veg': veg_path,
        'template_rast': template_rast_path,
        'base_output_dir': output_base_dir
    }
    
    # Create a list of tasks, where each task is a tuple of (tile_id, unbuffered_extent, buffered_extent, global_paths_dict)
    tasks = [
    (tile_id, unbuffered_extent, buffered_extent, global_paths_for_workers)
    for tile_id, unbuffered_extent, buffered_extent in tile_definitions]

    # Use fork
    # ctx = mp.get_context("fork")
    task_q = Queue()
    result_q = Queue()

    start_time_total = time.time()    

    # 1. Fill the task queue first
    for t in tasks:
        task_q.put(t)
    

    # sTART wORKERS
    num_workers = max(1, min(cpu_count() - 1, 30))
    workers = []
    for _ in range(num_workers):
        p = Process(target=worker_loop, args=(task_q, result_q, BASE_TEMP_DIR))
        p.start()
        workers.append(p)

    # 3. Add Poison Pills (one for each worker) AFTER starting them
    for _ in range(num_workers):
        task_q.put(None)
        
    # 4. Monitor results until ALL tasks are accounted for
    total_tasks = len(tasks)
    processed_count = 0

    print(f"[Master] Monitoring {total_tasks} tasks...")
    finished_tiles = set()
    exit_count=0
    
    # Add task counter and worker recycling logic
    # worker_task_counts = {p.pid: 0 for p in workers}
    
    # Keep loop running as long as tasks are pending OR workers are alive
    while processed_count < total_tasks or exit_count < num_workers:
        try:
            # Increase timeout slightly to allow for heavy URock processing
            msg = result_q.get(timeout=10) 
            print(f"[Master] {msg}", flush=True)
            
            # Only count actual finished tasks, not 'exit signal' messages
            if "Finished processing" in str(msg) or "Successfully processed" in str(msg):
                processed_count += 1
            elif "Failed" in str(msg):
                processed_count += 1 # Still count it so the loop can finish
        except Empty:
            # Check if workers are still alive
            if not any(p.is_alive() for p in workers):
                print("All workers died unexpectedly.")
                break
                
    # while any(p.is_alive() for p in workers):
    #     try:
    #         msg = result_q.get(timeout=0.5)
    #         print(msg, flush=True)
    #         for i, p in enumerate(workers[:]):
    #             if not p.is_alive():
    #                 continue
                
    #             if worker_task_counts[p.pid] >= tasks_per_worker:
    #                 p.terminate()
    #                 p.join()
    #                 new_p = Process(target=worker_loop, args=(task_q, result_q))
    #                 new_p.start()
    #                 workers[i] = new_p
    #                 worker_task_counts[new_p.pid] = 0
    #                 print(f"Recycled worker after {tasks_per_worker} tasks")
                    
    #     except Exception:  # queue.Empty
    #         pass
            
    # while not result_q.empty():
    #     print(result_q.get(), flush=True)

    for p in workers:
        p.join(timeout=60)
        if p.is_alive():
            print(f"[Master] Worker {p.pid} still alive; terminating.", flush=True)
            p.terminate()
            p.join()
    
    # Step 3: Determine the number of processes to use.
    # It's common practice to use (CPU count - 1) to leave one core free for system tasks.
    end_time_total = time.time()
    print(f"\nAll tiles processed successfully. Total elapsed time: {end_time_total - start_time_total:.2f} seconds.")

    merged_dir = os.path.join(output_base_dir, "merged")
    os.makedirs(merged_dir, exist_ok = True)
    merged_output_path = os.path.join(merged_dir, "merged_urock_wind_field_WS.tif")
    tile_files = glob(os.path.join(output_base_dir, "urock_wind_field_tile_*.tif"))
    if tile_files:
        print(f"Merging {len(tile_files)} tile outputs into {merged_output_path}...")
        gdal.Warp(merged_output_path, tile_files, options=gdal.WarpOptions(format="GTiff"))
        print("Merge complete.")
    else:
        print("No tile files found to merge.")

if __name__ == "__main__":
    # This ensures `main()` is called only when the script is executed directly,
    # not when imported as a module (important for multiprocessing on Windows).
    main()