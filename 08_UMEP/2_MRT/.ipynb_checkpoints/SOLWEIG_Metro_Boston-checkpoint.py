import os
import time
import sys
import glob
from osgeo import gdal, osr  # osr is needed for SpatialReference
import zipfile
import traceback
import numpy as np
import multiprocessing
import shutil

# === CONFIGURATION (Identical to your previous script) ===
tile_size = 500
buffer_pixels = 100

AREA = "Boston"
SCENARIO = "hot"
UTC_DIFF = -4
MIN_CPU_NUM = 50

profiles_path = "/storage/home/hcoda1/4/hyu483/qgis_332/"
qgis_prefix_path = "/storage/home/hcoda1/4/hyu483/conda_envs/qgis332"
shared_plugins_dir = os.path.join(profiles_path, "profiles/default/python/plugins")
shared_umep_plugin_dir = os.path.join(shared_plugins_dir, "processing_umep")

# Temporary directories
base_source_data_dir = f"/storage/project/r-rbasu31-0/shared/Metro_Boston/01_Input"
temp_base_dir = f"/storage/project/r-rbasu31-0/shared/Metro_Boston/02_Int_{SCENARIO}_final"
base_tile_buffered_input_dir = os.path.join(temp_base_dir, "Buffered_Inputs")
base_solweig_buffered_output_dir = os.path.join(temp_base_dir, "Buffered_SOLWEIG_Output")
worker_profiles_base_dir = os.path.join(temp_base_dir, "Worker_QGIS_Profiles")

# Output directories
final_output_base_dir = f"/storage/project/r-rbasu31-0/shared/Metro_Boston/03_Output_{SCENARIO}_SOLWEIG"
base_tile_debuffered_output_dir = os.path.join(final_output_base_dir, "Debuffered_Tiles")
merged_output_dir = os.path.join(final_output_base_dir, "Merged_Output")

# Default meteo path, overridable via --meteo flag
DEFAULT_METEO = f"/storage/project/r-rbasu31-0/shared/Met_Data/{AREA}_{SCENARIO}_new_UTC{UTC_DIFF}.txt"

PATHS_CONFIG = {
    "bDSM": os.path.join(base_source_data_dir, "DSM_fixed.tif"),
    "cDSM": os.path.join(base_source_data_dir, "cDSM_fixed.tif"),
    "dem": os.path.join(base_source_data_dir, "DEM_fixed.tif"),
    "wall_aspect": os.path.join(base_source_data_dir, "wall_aspect.tif"),
    "wall_height": os.path.join(base_source_data_dir, "wall_height.tif"),
    "lc": os.path.join(base_source_data_dir, "LULC_5class_fixed_reproj.tif"),
    "svf_input_dir": os.path.join(base_source_data_dir, "SVF"),
    "meteo": DEFAULT_METEO, 
}

# base_source_data_dir = "/storage/home/hcoda1/2/glin71/SOLWEIG/input/sample"  # ***CHANGE HERE***
# temp_base_dir = "/storage/scratch1/2/glin71/SOLWEIG-scratch"
# base_tile_buffered_input_dir = os.path.join(temp_base_dir, "Buffered_Inputs")
# base_solweig_buffered_output_dir = os.path.join(temp_base_dir, "Buffered_SOLWEIG_Output")
# worker_profiles_base_dir = os.path.join(temp_base_dir, "Worker_QGIS_Profiles")

# # Output directories
# final_output_base_dir = "/storage/scratch1/2/glin71/output/sample/SOLWEIG"  # ***CHANGE HERE***
# base_tile_debuffered_output_dir = os.path.join(final_output_base_dir, "Debuffered_Tiles")
# merged_output_dir = os.path.join(final_output_base_dir, "Merged_Output")

# PATHS_CONFIG = {
#     # Digital surface models
#     "bDSM": os.path.join(base_source_data_dir, "lidar_bg_mini.tif"),
#     "cDSM": os.path.join(base_source_data_dir, "lidar_veg_mini.tif"),
#     "dem": os.path.join(base_source_data_dir, "DEM.tif"),

#     # Wall parameters
#     "wall_aspect": os.path.join(base_source_data_dir, "wall_aspect.tif"),
#     "wall_height": os.path.join(base_source_data_dir, "wall_height.tif"),

#     # Land cover
#     "lc": os.path.join(base_source_data_dir, "land_cover_test.tif"),

#     # Sky view factors directory
#     "svf_input_dir": os.path.join(base_source_data_dir, "svfs"),

#     # Meteorological forcing file
#     "meteo": os.path.join(base_source_data_dir, "meteo_hot_typical.txt"),
# }

SOLWEIG_FIXED_PARAMS = {
    'TRANS_VEG': 3, 'LEAF_START': 97, 'LEAF_END': 300, 'CONIFER_TREES': False,
    'INPUT_TDSM': None, 'INPUT_THEIGHT': 25, 'USE_LC_BUILD': False,
    'SAVE_BUILD': False, 'INPUT_ANISO': '', 'ALBEDO_WALLS': 0.2, 'ALBEDO_GROUND': 0.15,
    'EMIS_WALLS': 0.9, 'EMIS_GROUND': 0.95, 'ABS_S': 0.7, 'ABS_L': 0.95,
    'POSTURE': 0, 'CYL': True, 'ONLYGLOBAL': True, 'UTC': UTC_DIFF,
    'POI_FILE': None, 'POI_FIELD': '', 'AGE': 35, 'ACTIVITY': 80, 'CLO': 0.9,
    'WEIGHT': 75, 'HEIGHT': 180, 'SEX': 0, 'SENSOR_HEIGHT': 10,
    'OUTPUT_TMRT': True, 'OUTPUT_KDOWN': False, 'OUTPUT_KUP': False, 'OUTPUT_LDOWN': False,
    'OUTPUT_LUP': False, 'OUTPUT_SH': False, 'OUTPUT_TREEPLANTER': False
}


gdal.UseExceptions()  # Enable GDAL exceptions
 
# --- Per-worker global state ---
_WORKER_PROCESSING = None
_WORKER_QGS_APP = None
_WORKER_TEMP_DIR = None              # The *symlink* path: <plugin_dir>/temp.
                                     # Always resolves through to the current per-tile dir.
_WORKER_PLUGIN_DIR = None
_WORKER_PROFILE_DIR = None
_WORKER_UMEP_PROVIDER = None
_WORKER_TILE_TEMPS_ROOT = None       # Real per-tile temp directories live under here.
_WORKER_CURRENT_TILE_TEMP = None     # The real directory the symlink currently points at.
 
 
def prepare_worker_umep_plugin(worker_profile_dir: str) -> tuple[str, str, str, str]:
    """
    Create a worker-private QGIS profile plugin directory and install
    processing_umep/temp as a *symlink* into a per-tile real directory.
 
    Each tile run will repoint the symlink at a fresh empty directory under
    `<worker_profile_dir>/tile_temps/<tile_id>/`. This sidesteps the NFS
    silly-rename / EBUSY problem entirely: we never have to delete files that
    UMEP/QGIS may still hold open between tiles. The previous tile's directory
    is just orphaned for NFS to reap once the holding handles close.
 
    worker_profile_dir must be the actual QGIS profile directory, i.e.
    <custom_config_root>/profiles/<profile_name>
 
    Returns (worker_plugins_dir, worker_umep_plugin_dir, plugin_temp_link, tile_temps_root).
    """
    worker_plugins_dir = os.path.join(worker_profile_dir, "python", "plugins")
    worker_umep_plugin_dir = os.path.join(worker_plugins_dir, "processing_umep")
    plugin_temp_link = os.path.join(worker_umep_plugin_dir, "temp")
    tile_temps_root = os.path.join(worker_profile_dir, "tile_temps")
 
    os.makedirs(worker_plugins_dir, exist_ok=True)
 
    if not os.path.isdir(shared_umep_plugin_dir):
        raise RuntimeError(f"Shared UMEP plugin directory not found: {shared_umep_plugin_dir}")
 
    if not os.path.isdir(worker_umep_plugin_dir):
        shutil.copytree(
            shared_umep_plugin_dir,
            worker_umep_plugin_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "temp")
        )
 
    os.makedirs(tile_temps_root, exist_ok=True)
 
    # The symlink must already point at a valid directory before QGIS init runs,
    # in case the UMEP plugin probes processing_umep/temp during load.
    initial_tile_temp = os.path.join(tile_temps_root, "_init")
    os.makedirs(initial_tile_temp, exist_ok=True)
 
    # Replace whatever is currently at processing_umep/temp with a symlink. This
    # may have been left as a real directory by a previous version of this script;
    # if rmtree cannot fully remove it (e.g. NFS .nfs* held-open files inside),
    # rename it aside instead — the symlink's slot needs to be free for os.symlink.
    try:
        if os.path.islink(plugin_temp_link):
            os.unlink(plugin_temp_link)
        elif os.path.isdir(plugin_temp_link):
            shutil.rmtree(plugin_temp_link, ignore_errors=True)
            if os.path.exists(plugin_temp_link):
                aside = f"{plugin_temp_link}.legacy.{os.getpid()}.{int(time.time())}"
                os.rename(plugin_temp_link, aside)
        elif os.path.lexists(plugin_temp_link):
            os.unlink(plugin_temp_link)
    except OSError as e:
        # Last-ditch: if we still cannot clear the slot, rename it aside.
        aside = f"{plugin_temp_link}.legacy.{os.getpid()}.{int(time.time())}"
        try:
            os.rename(plugin_temp_link, aside)
        except OSError:
            raise RuntimeError(
                f"Could not free processing_umep/temp slot at {plugin_temp_link}: {e}"
            ) from e
 
    os.symlink(initial_tile_temp, plugin_temp_link)
 
    return worker_plugins_dir, worker_umep_plugin_dir, plugin_temp_link, tile_temps_root
 
 
 
def initialize_worker(worker_profiles_root: str):
    """
    Initializer run once per worker process.
 
    Each worker gets its own copied processing_umep plugin directory, which means the
    fixed temp filenames used by SOLWEIG (svf.tif, svfN.tif, etc.) are isolated per worker.
    """
    global _WORKER_PROCESSING, _WORKER_QGS_APP, _WORKER_TEMP_DIR, _WORKER_PLUGIN_DIR, _WORKER_PROFILE_DIR, _WORKER_UMEP_PROVIDER, _WORKER_TILE_TEMPS_ROOT, _WORKER_CURRENT_TILE_TEMP
 
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
 
    worker_name = multiprocessing.current_process().name.replace(" ", "_")
    worker_pid = os.getpid()
 
    # QGIS expects custom profiles at:
    #   <QGIS_CUSTOM_CONFIG_PATH>/profiles/<QGIS_PROFILE>/python/plugins
    # Build that exact structure for each worker so the private plugin copy lives in
    # a normal-looking profile instead of an ad-hoc directory layout.
    worker_custom_config_root = os.path.join(worker_profiles_root, f"{worker_name}_{worker_pid}")
    worker_profile_name = "default"
    worker_profile_dir = os.path.join(worker_custom_config_root, "profiles", worker_profile_name)
    os.makedirs(worker_profile_dir, exist_ok=True)
 
    worker_plugins_dir, worker_umep_plugin_dir, plugin_temp_link, tile_temps_root = \
        prepare_worker_umep_plugin(worker_profile_dir)
 
    # Make sure this worker imports processing_umep from its private plugin copy.
    if worker_plugins_dir not in sys.path:
        sys.path.insert(0, worker_plugins_dir)
 
    # These are helpful for QGIS settings isolation. TMPDIR/TEMP/TMP point at the
    # symlink path; the OS resolves it freshly per file open, so each tile run sees
    # whatever real directory the symlink currently targets.
    os.environ["QGIS_CUSTOM_CONFIG_PATH"] = worker_custom_config_root
    os.environ["QGIS_PROFILE"] = worker_profile_name
    os.environ["TMPDIR"] = plugin_temp_link
    os.environ["TEMP"] = plugin_temp_link
    os.environ["TMP"] = plugin_temp_link
 
    # Avoid /run/user permission errors in batch jobs.
    xdg_runtime_dir = os.path.join(worker_custom_config_root, "xdg_runtime")
    os.makedirs(xdg_runtime_dir, exist_ok=True)
    try:
        os.chmod(xdg_runtime_dir, 0o700)
    except PermissionError:
        pass
    os.environ["XDG_RUNTIME_DIR"] = xdg_runtime_dir
 
    from qgis.core import QgsApplication
    import processing
    from processing.core.Processing import Processing
    from processing_umep.processing_umep_provider import ProcessingUMEPProvider
 
    QgsApplication.setPrefixPath(qgis_prefix_path, True)
    qgs_app = QgsApplication([], False)
    qgs_app.initQgis()
 
    Processing.initialize()
 
    registry = QgsApplication.processingRegistry()
    private_umep_provider = ProcessingUMEPProvider()
    added_ok = registry.addProvider(private_umep_provider)
    if not added_ok and registry.providerById("umep") is None:
        raise RuntimeError("Failed to add the private UMEP provider to the QGIS processing registry.")
 
    # Fail early if the SOLWEIG algorithm is not actually creatable in this worker.
    try:
        test_alg = registry.createAlgorithmById("umep:Outdoor Thermal Comfort: SOLWEIG")
    except Exception as e:
        raise RuntimeError(f"UMEP provider loaded but SOLWEIG algorithm could not be created in worker init: {e}") from e
    if test_alg is None:
        raise RuntimeError("UMEP provider loaded but SOLWEIG algorithm id was not available in worker init.")
    del test_alg
 
    _WORKER_PROCESSING = processing
    _WORKER_QGS_APP = qgs_app
    _WORKER_TEMP_DIR = plugin_temp_link
    _WORKER_PLUGIN_DIR = worker_umep_plugin_dir
    _WORKER_PROFILE_DIR = worker_profile_dir
    _WORKER_UMEP_PROVIDER = private_umep_provider
    _WORKER_TILE_TEMPS_ROOT = tile_temps_root
    _WORKER_CURRENT_TILE_TEMP = os.path.join(tile_temps_root, "_init")
 
    print(f"[worker pid={worker_pid}] Initialized QGIS with private UMEP plugin: {_WORKER_PLUGIN_DIR}")
    print(f"[worker pid={worker_pid}] QGIS custom config root: {worker_custom_config_root}")
    print(f"[worker pid={worker_pid}] UMEP temp symlink: {_WORKER_TEMP_DIR} -> {_WORKER_CURRENT_TILE_TEMP}")
    print(f"[worker pid={worker_pid}] Per-tile temp dirs root: {_WORKER_TILE_TEMPS_ROOT}")
 
 
 
def set_tile_umep_temp_dir(tile_id: str) -> str:
    """
    Give the SOLWEIG run for `tile_id` a brand-new, empty UMEP temp directory by
    atomically repointing the worker's processing_umep/temp symlink at it.
 
    This is the per-tile replacement for the old clear-in-place strategy. We
    never have to delete files that QGIS/UMEP may still hold open from a previous
    run; the previous tile's directory is just orphaned and best-effort removed.
    On NFS, any held-open files become .nfs* silly-renames inside the orphaned
    dir and get reaped automatically when their last fd closes.
 
    Returns the new real per-tile temp directory.
    """
    global _WORKER_CURRENT_TILE_TEMP
    if _WORKER_PLUGIN_DIR is None or _WORKER_TILE_TEMPS_ROOT is None:
        raise RuntimeError("Worker UMEP environment has not been initialized.")
 
    plugin_temp_link = os.path.join(_WORKER_PLUGIN_DIR, "temp")
 
    # Pick a unique target name; if a same-named dir somehow already exists,
    # disambiguate with pid + timestamp so we always start empty.
    new_tile_temp = os.path.join(_WORKER_TILE_TEMPS_ROOT, tile_id)
    if os.path.exists(new_tile_temp):
        new_tile_temp = os.path.join(
            _WORKER_TILE_TEMPS_ROOT, f"{tile_id}_{os.getpid()}_{int(time.time())}"
        )
    os.makedirs(new_tile_temp, exist_ok=True)
 
    # Atomic symlink swap: create a fresh symlink at a temp name and rename it
    # over the existing one. os.rename on a symlink is atomic on POSIX.
    swap_link = f"{plugin_temp_link}.swap.{os.getpid()}"
    try:
        if os.path.lexists(swap_link):
            os.unlink(swap_link)
    except OSError:
        pass
    os.symlink(new_tile_temp, swap_link)
    os.rename(swap_link, plugin_temp_link)
 
    # Best-effort cleanup of the previous tile's dir. We do NOT block on this:
    # any .nfs* held-open files will fail to remove and are simply left behind.
    prev = _WORKER_CURRENT_TILE_TEMP
    _WORKER_CURRENT_TILE_TEMP = new_tile_temp
    if prev and os.path.isdir(prev) and prev != new_tile_temp:
        shutil.rmtree(prev, ignore_errors=True)
 
    return new_tile_temp
 
 
def shutdown_worker_qgis():
    """Shut down the worker-local QGIS app cleanly."""
    global _WORKER_QGS_APP, _WORKER_PROCESSING, _WORKER_TEMP_DIR, _WORKER_PLUGIN_DIR, _WORKER_PROFILE_DIR, _WORKER_UMEP_PROVIDER, _WORKER_TILE_TEMPS_ROOT, _WORKER_CURRENT_TILE_TEMP
    if _WORKER_QGS_APP is not None:
        try:
            _WORKER_QGS_APP.exitQgis()
        except Exception:
            pass
    _WORKER_QGS_APP = None
    _WORKER_PROCESSING = None
    _WORKER_TEMP_DIR = None
    _WORKER_PLUGIN_DIR = None
    _WORKER_PROFILE_DIR = None
    _WORKER_UMEP_PROVIDER = None
    _WORKER_TILE_TEMPS_ROOT = None
    _WORKER_CURRENT_TILE_TEMP = None
 
 
def worker_loop(task_q, result_q, worker_profiles_root):
    """
    Mirror the long-lived worker architecture from the UROCK script: initialize QGIS/UMEP
    once, keep the provider object alive for the worker lifetime, then process many tiles
    from a queue.
    """
    worker_pid = os.getpid()
    try:
        initialize_worker(worker_profiles_root)
    except Exception as e:
        result_q.put(f"[worker pid={worker_pid}] INIT ERROR: {e}")
        shutdown_worker_qgis()
        return
 
    try:
        while True:
            task = task_q.get()
            if task is None:
                break
            try:
                result = process_tile_with_buffer(*task)
            except Exception as e:
                result = f"[worker pid={worker_pid}] ERROR: {e}\n{traceback.format_exc()}"
            result_q.put(result)
    finally:
        shutdown_worker_qgis()
 
 
# === HELPER FUNCTION ===
def calculate_gdal_sub_geotransform(parent_gt, x_offset_pixels, y_offset_pixels):
    """
    Calculates the GeoTransform for a sub-window of a raster based on pixel offsets.
    """
    # parent_gt = [ulx, x_res, x_skew, uly, y_skew, y_res]
    new_ulx = parent_gt[0] + x_offset_pixels * parent_gt[1] + y_offset_pixels * parent_gt[2]
    new_uly = parent_gt[3] + x_offset_pixels * parent_gt[4] + y_offset_pixels * parent_gt[5]
    return (new_ulx, parent_gt[1], parent_gt[2], new_uly, parent_gt[4], parent_gt[5])
 
 
# === FUNCTION TO SLICE RASTER BY WINDOW AND SAVE (GDAL) ===
def gdal_slice_raster_by_window(
    raster_path: str,
    out_path: str,
    slice_c_off: int,
    slice_r_off: int,
    slice_win_width: int,
    slice_win_height: int,
    output_format: str = "GTiff",
    # srcSRS: str = "EPSG:26986", ###
    # dst_srs: str = "EPSG:26986", ###
    resample_alg: str = "bilinear", ###
    creation_options: list = None
    ):
    if creation_options is None:
        creation_options = ["TILED=YES", "COMPRESS=LZW"]
 
    # Open source and grab its geotransform
    src_ds = gdal.Open(raster_path)
    if not src_ds:
        raise RuntimeError(f"Could not open {raster_path}")
 
    gt = src_ds.GetGeoTransform()
    origin_x, px_w, _, origin_y, _, px_h = gt
 
    # Compute the geographic bounds of the pixel window
    minx = origin_x + slice_c_off * px_w
    maxx = minx + slice_win_width * px_w
 
    # Note: px_h is typically negative, so maxy = origin_y + slice_r_off*px_h
    maxy = origin_y + slice_r_off * px_h
    miny = maxy + slice_win_height * px_h
 
    # Make sure your output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
 
    # Build the WarpOptions
    warp_opts = gdal.WarpOptions(
        format=output_format,
        # dstSRS=dst_srs,
        resampleAlg=resample_alg,
        creationOptions=creation_options,
        outputBounds=[minx, miny, maxx, maxy],
        width=slice_win_width,
        height=slice_win_height
    )
 
    # Do the warp (crop)
    ds = gdal.Warp(
        destNameOrDestDS=out_path,
        srcDSOrSrcDSTab=raster_path,
        options=warp_opts
    )
    if ds is None:
        raise RuntimeError("gdal.Warp failed")
 
    # Close datasets
    ds = None
    src_ds = None
 
    return out_path
 
 
# === FUNCTION TO DEBUFFER AND SAVE RASTER (GDAL) ===
def debuffer_and_save_raster_gdal(
    buffered_raster_path: str,
    debuffered_raster_path: str,
    original_full_raster_geotransform: tuple, # GeoTransform of the *original* large source raster
    original_core_tile_col_off_in_full: int, # Column offset of the core tile within the full raster
    original_core_tile_row_off_in_full: int, # Row offset of the core tile within the full raster
    core_tile_width: int,                   # Width of the non-buffered core data
    core_tile_height: int,                  # Height of the non-buffered core data
    actual_buffer_on_left_pixels: int,      # Buffer pixels on the left *within the buffered_raster_path*
    actual_buffer_on_top_pixels: int        # Buffer pixels on the top *within the buffered_raster_path*
    ):
    """
    Clips the core data from a buffered raster using GDAL and saves it
    with correct global georeferencing.
    """
    src_buffered_ds = gdal.Open(buffered_raster_path, gdal.GA_ReadOnly)
    if src_buffered_ds is None:
        print(f"ERROR: Could not open buffered raster: {buffered_raster_path}")
        return None
 
    try:
        src_buffered_band = src_buffered_ds.GetRasterBand(1)
        if src_buffered_band is None:
            print(f"ERROR: Could not get band from {buffered_raster_path}")
            src_buffered_ds = None
            return None
 
        # Read the core data from the buffered raster
        # xoff, yoff, xsize, ysize for ReadAsArray
        core_data = src_buffered_band.ReadAsArray(
            xoff=actual_buffer_on_left_pixels,
            yoff=actual_buffer_on_top_pixels,
            win_xsize=core_tile_width,
            win_ysize=core_tile_height
        ).astype(float)
 
        if core_data is None:
            print(f"ERROR: Failed to read core data from {buffered_raster_path}")
            src_buffered_ds = None
            return None
 
        if core_data.shape[0] != core_tile_height or core_data.shape[1] != core_tile_width:
            print(f"ERROR: Read core data shape ({core_data.shape}) does not match expected ({core_tile_height}, {core_tile_width}) for {debuffered_raster_path}")
            src_buffered_ds = None
            return None
 
        # Calculate the correct geotransform for this debuffered (core) tile.
        # This transform places the core tile in its correct global position.
        final_core_geotransform = calculate_gdal_sub_geotransform(
            original_full_raster_geotransform,
            original_core_tile_col_off_in_full,
            original_core_tile_row_off_in_full
        )
 
        # Create the output debuffered raster
        driver = gdal.GetDriverByName("GTiff")
        if driver is None:
            print("ERROR: GTiff driver not available.")
            src_buffered_ds = None
            return None
 
        os.makedirs(os.path.dirname(debuffered_raster_path), exist_ok=True)
 
        # Get data type from source buffered band
        gdal_data_type = src_buffered_band.DataType
 
        dst_ds = driver.Create(
            debuffered_raster_path,
            xsize=core_tile_width,
            ysize=core_tile_height,
            bands=1, # Assuming single band TMRT
            eType=gdal_data_type,
            options=["COMPRESS=LZW"] # Add other options if needed
        )
        if dst_ds is None:
            print(f"ERROR: Could not create output raster: {debuffered_raster_path}")
            src_buffered_ds = None
            return None
 
        dst_ds.SetGeoTransform(final_core_geotransform)
        dst_ds.SetProjection(src_buffered_ds.GetProjection()) # Preserve CRS
 
        dst_band = dst_ds.GetRasterBand(1)
        dst_band.WriteArray(core_data)
        no_data_value = src_buffered_band.GetNoDataValue()
        if no_data_value is not None:
            dst_band.SetNoDataValue(no_data_value)
 
        dst_band.FlushCache()
        dst_ds = None # Close and save
 
        return debuffered_raster_path
 
    except Exception as e:
        print(f"ERROR during debuffering for {buffered_raster_path} to {debuffered_raster_path}: {e}\n{traceback.format_exc()}")
        return None
    finally:
        if src_buffered_ds:
            src_buffered_ds = None
 
 
# === WORKER FUNCTION FOR PROCESSING A SINGLE TILE ===
def process_tile_with_buffer(
    core_r_offset, core_c_offset,
    core_tile_width, core_tile_height,
    buffer_px,
    full_raster_total_width, full_raster_total_height,
    original_full_raster_geotransform_tuple, original_full_raster_crs_wkt, # GDAL specific
    paths_cfg_dict,
    base_buffered_input_dir_worker,
    base_solweig_out_dir_worker,
    base_debuffered_out_dir_worker,
    tile_id_str
    ):
    if _WORKER_PROCESSING is None:
        raise RuntimeError("Worker processing environment was not initialized.")
 
    print(f"[{tile_id_str}] Starting processing with buffer...")
    tile_processing_start_time = time.time()
    debuffered_output_files_for_this_tile = []
 
    try:
        # --- 1. Calculate Buffered Window for Slicing Inputs ---
        slice_r_off = max(0, core_r_offset - buffer_px)
        slice_c_off = max(0, core_c_offset - buffer_px)
        slice_r_end = min(full_raster_total_height, core_r_offset + core_tile_height + buffer_px)
        slice_c_end = min(full_raster_total_width, core_c_offset + core_tile_width + buffer_px)
        slice_win_height = (slice_r_end - slice_r_off)
        slice_win_width = (slice_c_end - slice_c_off)
 
        if slice_win_width <= 0 or slice_win_height <= 0:
            msg = f"[{tile_id_str}] Skipped: Calculated buffered slice window has zero/negative dimension. W:{slice_win_width} H:{slice_win_height}"
            print(msg)
            return msg # Return error/skip message
 
        actual_buffer_left = core_c_offset - slice_c_off
        actual_buffer_top = core_r_offset - slice_r_off
 
        # --- 2. Prepare Directories for this Tile ---
        current_tile_buffered_data_dir = os.path.join(base_buffered_input_dir_worker, tile_id_str)
        current_tile_solweig_output_dir = os.path.join(base_solweig_out_dir_worker, tile_id_str)
        current_tile_buffered_svf_dir = os.path.join(current_tile_buffered_data_dir, "svf_buffered_tiles")
        os.makedirs(current_tile_solweig_output_dir, exist_ok=True)
        current_tile_debuffered_output_dir = os.path.join(base_debuffered_out_dir_worker, tile_id_str)
        os.makedirs(current_tile_debuffered_output_dir, exist_ok=True)
 
        # *** EDITED SECTION 1: CHECK FOR EXISTING OUTPUT ***
        required_tmrt_filenames = [
            "Tmrt_2023_208_0900D.tif", "Tmrt_2023_208_1300D.tif", "Tmrt_2023_208_1700D.tif",
            "Tmrt_2023_236_0900D.tif", "Tmrt_2023_236_1300D.tif", "Tmrt_2023_236_1700D.tif",
        ]
        all_files_exist = all(os.path.exists(os.path.join(current_tile_solweig_output_dir, f)) for f in required_tmrt_filenames)
        if all_files_exist:
            print(f"[{tile_id_str}] Found all {len(required_tmrt_filenames)} required SOLWEIG output files. Skipping processing run.")
        else:
            os.makedirs(current_tile_buffered_data_dir, exist_ok=True)
            os.makedirs(current_tile_buffered_svf_dir, exist_ok=True)
            # --- 3. Slice Main Input Rasters (Buffered) ---
            tile_specific_buffered_inputs = {}
            for key in ["bDSM", "cDSM", "dem", "wall_aspect", "wall_height", "lc"]:
                in_path = paths_cfg_dict[key]
                out_filename = f"{key}_{tile_id_str}_buffered.tif"
                out_path = os.path.join(current_tile_buffered_data_dir, out_filename)
                tile_specific_buffered_inputs[key] = gdal_slice_raster_by_window(
                    raster_path=in_path, out_path=out_path,
                    slice_c_off=slice_c_off, slice_r_off=slice_r_off,
                    slice_win_width=slice_win_width, slice_win_height=slice_win_height
                )
                if tile_specific_buffered_inputs[key] is None: # Check if slicing failed
                     raise RuntimeError(f"Failed to slice {key} for {tile_id_str}")
 
            # --- 4. Clip and Zip SVF Files (Buffered) ---
            source_svf_files = sorted(glob.glob(os.path.join(paths_cfg_dict['svf_input_dir'], "*.tif")))
            clipped_svf_paths_for_zip = []
            if source_svf_files:
                for svf_file_path in source_svf_files:
                    filename = os.path.basename(svf_file_path)
                    out_svf_filename = f"{os.path.splitext(filename)[0]}_{tile_id_str}_buffered{os.path.splitext(filename)[1]}"
                    out_svf_tile_path = os.path.join(current_tile_buffered_svf_dir, out_svf_filename)
                    clipped_path = gdal_slice_raster_by_window(
                        raster_path=svf_file_path, out_path=out_svf_tile_path,
                        slice_c_off=slice_c_off, slice_r_off=slice_r_off,
                        slice_win_width=slice_win_width, slice_win_height=slice_win_height
                    )
                    if clipped_path:
                        clipped_svf_paths_for_zip.append(clipped_path)
                    else:
                        print(f"[{tile_id_str}] WARNING: Failed to slice SVF file: {svf_file_path}")
 
            tile_svf_zip_path = None
            if clipped_svf_paths_for_zip:
                tile_svf_zip_path = os.path.join(current_tile_buffered_data_dir, f"svfs_{tile_id_str}_buffered.zip")
                if os.path.exists(tile_svf_zip_path):
                    os.remove(tile_svf_zip_path)
                with zipfile.ZipFile(tile_svf_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for svf_tile_path in clipped_svf_paths_for_zip:
                        original_basename = os.path.basename(svf_tile_path).split(f'_{tile_id_str}_buffered')[0] + os.path.splitext(svf_tile_path)[1]
                        zipf.write(svf_tile_path, arcname=original_basename)
 
            elif source_svf_files:
                print(f"[{tile_id_str}] WARNING: SVF files found but no SVFs were successfully clipped/zipped for buffered tile.")
 
            # --- 5. Run SOLWEIG on Buffered Inputs ---
            solweig_params = {
                'INPUT_DSM': tile_specific_buffered_inputs.get('bDSM'),
                'INPUT_SVF': tile_svf_zip_path, # This can be None if SVFs failed
                'INPUT_HEIGHT': tile_specific_buffered_inputs.get('wall_height'),
                'INPUT_ASPECT': tile_specific_buffered_inputs.get('wall_aspect'),
                'INPUT_CDSM': tile_specific_buffered_inputs.get('cDSM'),
                'INPUT_LC': tile_specific_buffered_inputs.get('lc'),
                'INPUT_DEM': tile_specific_buffered_inputs.get('dem'),
                'INPUTMET': paths_cfg_dict['meteo'],
                'OUTPUT_DIR': current_tile_solweig_output_dir,
                **SOLWEIG_FIXED_PARAMS,
            }
            critical_inputs_present = all([
                solweig_params['INPUT_DSM'], solweig_params['INPUT_LC'], solweig_params['INPUT_DEM'],
                # Make INPUT_SVF optional if your workflow allows it, otherwise keep it mandatory
                solweig_params['INPUT_SVF'] if source_svf_files else True, # Only critical if source SVFs exist
                solweig_params['INPUT_CDSM'], solweig_params['INPUT_HEIGHT'], solweig_params['INPUT_ASPECT']
            ])
 
            if not critical_inputs_present:
                missing_inputs = [k for k, v in solweig_params.items() if v is None and k.startswith('INPUT_')]
                msg = f"[{tile_id_str}] Skipped SOLWEIG: missing critical input files: {missing_inputs}."
                print(msg)
                # Still try to cleanup before returning
                shutil.rmtree(current_tile_buffered_data_dir, ignore_errors=True)
                return msg
 
            print(f"[{tile_id_str}] Running SOLWEIG. Output dir: {solweig_params['OUTPUT_DIR']}")
            # Repoint processing_umep/temp at a fresh empty per-tile directory.
            # No deletion of the previous tile's temp is required here — the swap
            # function takes care of best-effort cleanup, and any NFS-held-open
            # files from the prior run stay safely inside the orphaned directory.
            tile_temp_dir = set_tile_umep_temp_dir(tile_id_str)
            print(f"[{tile_id_str}] UMEP temp now points at: {tile_temp_dir}")
            _WORKER_PROCESSING.run("umep:Outdoor Thermal Comfort: SOLWEIG", solweig_params)
            proc_time = (time.time() - tile_processing_start_time) / 60
            print(f"[{tile_id_str}] SOLWEIG processing complete in {proc_time:.2f} mins.")
 
        # *** EDITED SECTION 2: CLEANUP ***
        print(f"[{tile_id_str}] Cleaning up temporary buffered input directory: {current_tile_buffered_data_dir}")
        try:
            shutil.rmtree(current_tile_buffered_data_dir, ignore_errors=True)
        except Exception as e:
            print(f"[{tile_id_str}] WARNING: Could not clean up temporary directory {current_tile_buffered_data_dir}: {e}")
 
        proc_time = (time.time() - tile_processing_start_time) / 60
        print(f"[{tile_id_str}] Tile task finished in {proc_time:.2f} mins.")
 
        # --- 6. Debuffer the TMRT Output ---
        tmrt_glob_pattern = os.path.join(current_tile_solweig_output_dir, "Tmrt_*.tif")
        all_buffered_tmrt_files = glob.glob(tmrt_glob_pattern)
 
        if not all_buffered_tmrt_files:
            print(f"[{tile_id_str}] WARNING: No TMRT output files found in {current_tile_solweig_output_dir} after SOLWEIG run.")
            # This might not be an error if SOLWEIG was not expected to produce TMRT in some cases
            # but usually it's an indication of a problem with the SOLWEIG run itself.
 
        for buffered_tmrt_path in all_buffered_tmrt_files:
            buffered_tmrt_filename_base = os.path.basename(buffered_tmrt_path)
            debuffered_tmrt_filename = f"{os.path.splitext(buffered_tmrt_filename_base)[0]}_{tile_id_str}_debuffered.tif"
            debuffered_tmrt_path = os.path.join(current_tile_debuffered_output_dir, debuffered_tmrt_filename)
 
            # Parameters for debuffer_and_save_raster_gdal:
            debuffered_file = debuffer_and_save_raster_gdal(
                buffered_raster_path=buffered_tmrt_path,
                debuffered_raster_path=debuffered_tmrt_path,
                original_full_raster_geotransform=original_full_raster_geotransform_tuple,
                original_core_tile_col_off_in_full=core_c_offset,
                original_core_tile_row_off_in_full=core_r_offset,
                core_tile_width=core_tile_width,
                core_tile_height=core_tile_height,
                actual_buffer_on_left_pixels=actual_buffer_left,
                actual_buffer_on_top_pixels=actual_buffer_top
            )
            if debuffered_file:
                debuffered_output_files_for_this_tile.append(debuffered_file)
                print(f"[{tile_id_str}] Successfully debuffered: {debuffered_file}")
            else:
                print(f"[{tile_id_str}] ERROR failed to debuffer: {buffered_tmrt_path}")
 
        if not debuffered_output_files_for_this_tile and all_buffered_tmrt_files:
            msg = f"[{tile_id_str}] ERROR: Found TMRT files but failed to debuffer any."
            print(msg)
            return msg
 
        proc_time = (time.time() - tile_processing_start_time) / 60
        msg = f"[{tile_id_str}] Tile processing (buffered) complete in {proc_time:.2f} mins. {len(debuffered_output_files_for_this_tile)} TMRT files debuffered."
        print(msg)
        return debuffered_output_files_for_this_tile
 
    except Exception as e:
        err_msg = f"[{tile_id_str}] ERROR processing tile: {e}\n{traceback.format_exc()}"
        print(err_msg)
        return err_msg
 
 
def merge_debuffered_outputs(results_from_pool, overall_start_time):
    print("\n=== Tile processing complete. ===")
 
    organized_debuffered_files = {}
    failed_tile_processing_count = 0
    for result_item in results_from_pool:
        if isinstance(result_item, str) and ("ERROR" in result_item or "Skipped" in result_item or "WARNING" in result_item):
            failed_tile_processing_count += 1
        elif isinstance(result_item, list):
            for debuffered_path in result_item:
                if os.path.exists(debuffered_path):
                    filename = os.path.basename(debuffered_path)
                    parts = filename.split('_tile_')[0]
                    tmrt_type_key = parts
                    if tmrt_type_key not in organized_debuffered_files:
                        organized_debuffered_files[tmrt_type_key] = []
                    organized_debuffered_files[tmrt_type_key].append(debuffered_path)
                else:
                    print(f"Warning: Worker reported debuffered file, but not found: {debuffered_path}")
        else:
            failed_tile_processing_count += 1
            print(f"  Unknown result type from worker for a tile: {type(result_item)} - {str(result_item)[:200]}")
 
    print(f"\nTile processing summary: {len(results_from_pool) - failed_tile_processing_count} tiles had successful outcomes (returned list of paths).")
    print(f"Tiles that returned error/skip messages or unknown result types: {failed_tile_processing_count}")
 
    if not organized_debuffered_files:
        print("No debuffered files were successfully organized for merging.")
        overall_proc_time_early_exit = (time.time() - overall_start_time) / 60
        print(f"\nTotal script execution time: {overall_proc_time_early_exit:.2f} minutes.")
        return
 
    # --- Merge Each Set of Debuffered TMRT Tiles using GDAL ---
    for tmrt_type, file_list_to_merge in organized_debuffered_files.items():
        if not file_list_to_merge:
            print(f"No files to merge for TMRT type: {tmrt_type}")
            continue
 
        print(f"\nAttempting to merge {len(file_list_to_merge)} debuffered tiles for TMRT type: {tmrt_type}...")
        merged_tmrt_output_path = os.path.join(merged_output_dir, f"{tmrt_type}_merged_final.tif")
        vrt_path = os.path.join(merged_output_dir, f"{tmrt_type}_temp.vrt")
 
        try:
            valid_files_for_vrt = [f for f in file_list_to_merge if os.path.exists(f)]
            if not valid_files_for_vrt:
                print(f"ERROR: No valid source files found for merging {tmrt_type} after checking existence.")
                continue
 
            vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=False)
            vrt_ds = gdal.BuildVRT(vrt_path, valid_files_for_vrt, options=vrt_options)
            if vrt_ds is None:
                print(f"ERROR: Failed to build VRT for {tmrt_type}.")
                continue
            vrt_ds = None
 
            nodata_val = None
            if valid_files_for_vrt:
                first_tile_ds = gdal.Open(valid_files_for_vrt[0])
                if first_tile_ds:
                    nodata_val = first_tile_ds.GetRasterBand(1).GetNoDataValue()
                    first_tile_ds = None
 
            translate_options_list = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"]
            if nodata_val is not None:
                final_ds = gdal.Translate(
                    merged_tmrt_output_path,
                    vrt_path,
                    format="GTiff",
                    creationOptions=translate_options_list,
                    noData=nodata_val if nodata_val is not None else 'none'
                )
            else:
                final_ds = gdal.Translate(
                    merged_tmrt_output_path,
                    vrt_path,
                    format="GTiff",
                    creationOptions=translate_options_list
                )
 
            if final_ds is None:
                print(f"ERROR: Failed to translate VRT to TIFF for {tmrt_type}.")
                if os.path.exists(vrt_path):
                    os.remove(vrt_path)
                continue
            final_ds = None
 
            print(f"Successfully merged {tmrt_type} tiles to: {merged_tmrt_output_path}")
            if os.path.exists(vrt_path):
                os.remove(vrt_path)
 
        except Exception as e:
            print(f"ERROR during GDAL merging of {tmrt_type} tiles: {e}\n{traceback.format_exc()}")
            if os.path.exists(vrt_path) and os.path.isfile(vrt_path):
                try:
                    os.remove(vrt_path)
                except OSError as ose:
                    print(f"Could not remove temporary VRT {vrt_path}: {ose}")
 
    overall_proc_time = (time.time() - overall_start_time) / 60
    print(f"\nTotal script execution time: {overall_proc_time:.2f} minutes.")
 
 
# === MAIN SCRIPT EXECUTION ===
def main():
    overall_start_time = time.time()
    os.makedirs(base_tile_buffered_input_dir, exist_ok=True)
    os.makedirs(base_solweig_buffered_output_dir, exist_ok=True)
    os.makedirs(base_tile_debuffered_output_dir, exist_ok=True)
    os.makedirs(merged_output_dir, exist_ok=True)
    os.makedirs(worker_profiles_base_dir, exist_ok=True)
 
    try:
        meteo_data = np.genfromtxt(PATHS_CONFIG['meteo'], delimiter=None)
        if meteo_data.ndim == 0 or meteo_data.size == 0:
            raise ValueError("Meteorological data is empty.")
        print(f"Meteorological data shape: {meteo_data.shape}")
    except Exception as e:
        print(f"CRITICAL ERROR reading meteorological file: {PATHS_CONFIG['meteo']}. Error: {e}")
        return
 
    # --- Get dimensions and georeferencing from one of the main full-sized rasters using GDAL ---
    ref_ds = gdal.Open(PATHS_CONFIG['dem'], gdal.GA_ReadOnly)
    if ref_ds is None:
        print(f"CRITICAL ERROR: Could not open reference raster: {PATHS_CONFIG['dem']}")
        return
    full_raster_width = ref_ds.RasterXSize
    full_raster_height = ref_ds.RasterYSize
    original_raster_geotransform = ref_ds.GetGeoTransform()  # Tuple
    original_raster_crs_wkt = ref_ds.GetProjection()         # WKT string
    ref_ds = None
    print(f"Full source raster dimensions: {full_raster_width}x{full_raster_height}, Tile Size: {tile_size}x{tile_size}, Buffer: {buffer_pixels}px")
    print(f"Source GeoTransform: {original_raster_geotransform}")
    print(f"Source CRS (WKT): {original_raster_crs_wkt[:100]}...")
 
    tasks_for_pool = []
    for r_idx, core_r_off in enumerate(range(0, full_raster_height, tile_size)):
        for c_idx, core_c_off in enumerate(range(0, full_raster_width, tile_size)):
            current_core_tile_width = min(tile_size, full_raster_width - core_c_off)
            current_core_tile_height = min(tile_size, full_raster_height - core_r_off)
 
            if current_core_tile_width <= 0 or current_core_tile_height <= 0:
                print(f"Skipping task generation for zero-dimension core tile at r_offset={core_r_off}, c_offset={core_c_off}")
                continue
 
            tile_identifier_string = f"tile_{r_idx}_{c_idx}"
            task_args = (
                core_r_off, core_c_off,
                current_core_tile_width, current_core_tile_height,
                buffer_pixels,
                full_raster_width, full_raster_height,
                original_raster_geotransform, original_raster_crs_wkt,
                PATHS_CONFIG,
                base_tile_buffered_input_dir,
                base_solweig_buffered_output_dir,
                base_tile_debuffered_output_dir,
                tile_identifier_string
            )
            tasks_for_pool.append(task_args)
 
    if not tasks_for_pool:
        print("No tasks generated.")
        return
 
    print(f"\nGenerated {len(tasks_for_pool)} tasks for multiprocessing.")
 
    requested_processes = min(MIN_CPU_NUM, max(1, multiprocessing.cpu_count() - 1), len(tasks_for_pool))
    print(f"Using {requested_processes} long-lived worker processes with per-worker isolated QGIS/UMEP temp directories.")
 
    ctx = multiprocessing.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
 
    for task in tasks_for_pool:
        task_q.put(task)
    for _ in range(requested_processes):
        task_q.put(None)
 
    workers = [
        ctx.Process(
            target=worker_loop,
            args=(task_q, result_q, worker_profiles_base_dir)
        )
        for _ in range(requested_processes)
    ]
 
    for worker in workers:
        worker.start()
 
    results_from_pool = []
    expected_results = len(tasks_for_pool)
    while len(results_from_pool) < expected_results:
        try:
            result_item = result_q.get(timeout=5)
            results_from_pool.append(result_item)
        except Exception:
            if not any(worker.is_alive() for worker in workers):
                break
 
    for worker in workers:
        worker.join()
 
    if len(results_from_pool) < expected_results:
        missing_count = expected_results - len(results_from_pool)
        print(f"WARNING: Only collected {len(results_from_pool)} worker results out of {expected_results}. Missing {missing_count} result(s).")
        for worker in workers:
            if worker.exitcode not in (0, None):
                print(f"Worker pid={worker.pid} exited with code {worker.exitcode}")
 
    merge_debuffered_outputs(results_from_pool, overall_start_time)
 
 
if __name__ == '__main__':
    main()