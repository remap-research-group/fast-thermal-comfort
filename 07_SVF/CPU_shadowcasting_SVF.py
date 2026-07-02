import os, os.path
from osgeo import gdal
from osgeo.gdalconst import *
from matplotlib import pyplot as plt
import numpy as np
import rasterio
import time
import math
import sys
import glob

from numba import njit, prange
import numba



# ----------------------------Configuration-------------------------------------------------------
# demfile = 'my_buildingsshp_rasterized_1m.tif'
demfile_pattern = f"/storage/project/r-rbasu31-0/shared/Tucson/Oracle_Granada_15/Reprojected/cDSM.tif"
demfiles = sorted(glob.glob(demfile_pattern))

# OUTPUT_DIR = "/storage/project/r-rbasu31-0/shared/Metro_Boston/Bus_Scenario/Chelsea/01_Input_S1_1/SVF"
# demfile = 'boston_sample.tif'
# demfile = "/media/remap/NO_HEAT_RB/Metro_Boston/Raw/Building_Footprints_OvertureMaps/old/building_footprint_boston_height_raster.tif"

# CHANGE THIS VARIABLE AS NEEDED
# if its a canopy raster (and not a building raster), add word "veg" in output results¶
canopy = True # True when you are putting canopy DSM.tif
rangeDist = 200 
radius_start = 0
radius_step_size = 2
scale = 1
TILE_SIZE = 3200
BORDER = 400
# ----------------------------Configuration-------------------------------------------------------

print(f"Found {len(demfiles)} DSM files.", flush=True)

index = int(0)
iazimuth = np.hstack(np.zeros((1, 145)))
azistart = np.array([0, 4, 2, 5, 8, 0, 10, 0])
skyvaultaziint = np.array([12, 12, 15, 15, 20, 30, 60, 360]) #azimuth
skyvaultaltint = np.array([6, 18, 30, 42, 54, 66, 78, 90])   #altitude

for j in range(0, 8):
    for k in range(0, int(360 / skyvaultaziint[j])):
        iazimuth[index] = k * skyvaultaziint[j] + azistart[j]
        if iazimuth[index] > 360.:
            iazimuth[index] = iazimuth[index] - 360.
        index = index + 1

iazimuth = np.ascontiguousarray(iazimuth)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

@njit
def annulus_weight(altitude, aziinterval):
    n = 90.0
    steprad = (360/aziinterval) * np.pi/180.0
    annulus = 91.0 - altitude
    w = 1.0/(2.0*np.pi) * math.sin(np.pi/(2.0*n)) * math.sin((np.pi * (2.0 * annulus - 1.0)) / (2.0 * n))
    weight = steprad * w
    return weight

@njit(parallel=True)
def parallel_SVF_calculate(arrayDEM, azimuth_arr, radius_start, rangeDist, radius_step_size):

    rows, cols = arrayDEM.shape
    rows = int(rows)
    cols = int(cols)
    
    # output numpy arrays
    SVF_total = np.zeros(arrayDEM.shape, dtype=np.float32)
    SVF_east = np.zeros(arrayDEM.shape, dtype=np.float32)
    SVF_north = np.zeros(arrayDEM.shape, dtype=np.float32)
    SVF_west = np.zeros(arrayDEM.shape, dtype=np.float32)
    SVF_south = np.zeros(arrayDEM.shape, dtype=np.float32)
    
    iangle = [6, 18, 30, 42, 54, 66, 78, 90]
    aziinterval = [30, 30, 24, 24, 18, 12, 6, 1]
    annulino = [0, 12, 24, 36, 48, 60, 72, 84, 90]


    for idx in prange(0, rows*cols):
        # svf value holders for each pixel
        svf = 0
        svf_N = 0
        svf_E = 0
        svf_S = 0
        svf_W = 0

        # this represents the total svf (and weighting) if all patches in the cardinal direction angle sector were visible (so flag = 1)

        svf_N_max = 0
        svf_E_max = 0
        svf_S_max = 0
        svf_W_max = 0

        # calculate correct pixel location in arrayDEM based on idx
        px = (idx % cols)
        py = (idx // cols)

        # svf_weights = np.zeros(145, dtype=np.float64)
        # weight_idx = 0
        azimuth_idx=0
        for i in range(0,8):
            for j in range(0, aziinterval[i]):
                altitude = iangle[i];
                azimuth = azimuth_arr[azimuth_idx];
                altitude_degree = np.pi*altitude/180.0
                
                # convert the sun azimuth (clockwise zero at North) to theta (anticlockwise, zero at east)
                if (azimuth < 90 and azimuth > 0):
                    theta = np.pi*(90 - azimuth)/180.0
                else: #azimuth > 180 & azimuth<360
                    theta = np.pi*(450 - azimuth)/180.0 
                
                # the search radius for the SVF computing
                sh = 0
                f = arrayDEM[py,px]
                temp = 0

                radiusRange = range(radius_start, rangeDist, radius_step_size)

                for radius in radiusRange:
                    rayX = int(px + radius*math.cos(theta))
                    rayY = int(py - radius*math.sin(theta))
                    
                    if (rayX >= cols or rayX < 0 or rayY >= rows or rayY < 0):
                        break            
                    
                    # building height information
                    temp = arrayDEM[rayY, rayX] - radius*math.tan(altitude_degree); #need to do /scale if not using a 1m resolution, but doesn't matter since we are
                    
                    if f < temp:
                        f = temp
                if (f == arrayDEM[py,px]):
                    sh = 1 # visible
                else: 
                    sh = 0 # not visible
                
                for k in range(annulino[i]+1, annulino[i+1]+1):
                    weight = annulus_weight(k, aziinterval[i])

                    if (azimuth >= 270 or azimuth < 90):  # total North visibility
                        svf_N_max += weight
                    if (azimuth >= 0 and azimuth < 180):   # total East visibility
                        svf_E_max += weight
                    if (azimuth >= 90 and azimuth < 270):  # total South visibility
                        svf_S_max += weight
                    if (azimuth >= 180 and azimuth < 360): # total West visibility
                        svf_W_max += weight

                    weight *= sh # multiplies weights by flags, so now we only include those patches that are visible
                    svf = svf + weight
                    
                    # svf_weights[weight_idx] = svf
                    # weight_idx += 1
                    
                    if (azimuth >= 0 and azimuth < 180):
                        svf_E += weight
                    if (azimuth >= 90 and azimuth < 270):
                        svf_S += weight
                    if (azimuth >= 180 and azimuth < 360):
                        svf_W += weight
                    if (azimuth >= 270 or azimuth < 90):
                        svf_N += weight
                azimuth_idx+=1
                        
        # if weight_idx > 0:
        #     svf = np.sum(svf_weights[:weight_idx], dtype=np.float64)
        # else:
        #     svf = 0.0  # Default value

        # normalizing svf value for each direction by the total weightage of patches in that cardinal direction sector

        svf_N = svf_N / svf_N_max if svf_N_max > 0 else 0
        svf_E = svf_E / svf_E_max if svf_E_max > 0 else 0
        svf_S = svf_S / svf_S_max if svf_S_max > 0 else 0
        svf_W = svf_W / svf_W_max if svf_W_max > 0 else 0
    
        SVF_total[py,px] = svf
        SVF_north[py,px] = svf_N
        SVF_east[py,px] = svf_E
        SVF_south[py,px] = svf_S
        SVF_west[py,px] = svf_W

    return SVF_total, SVF_north, SVF_east, SVF_south, SVF_west             

# start_time = time.time()

# SVF_total, SVF_north, SVF_east, SVF_south, SVF_west = parallel_SVF_calculate(arrayDEM, iazimuth, radius_start, rangeDist, radius_step_size)

# end_time = time.time()
# elapsed_time = end_time - start_time
# print("Execution Time: " + str(elapsed_time), flush=True)
# import pycuda.driver as cuda
# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray

def process_large_svf(dsm, scale, rangeDist, radius_start, radius_step_size):
    """Process SVF on a large raster by tiling it into overlapping TILE_SIZExTILE_SIZE chunks."""

    height, width = dsm.shape

    svf_result = np.zeros((height, width), dtype=np.float32)
    # print("created svf_result", flush=True)
    svf_n_result = np.zeros((height, width), dtype=np.float32)
    # print("created north", flush=True)
    svf_e_result = np.zeros((height, width), dtype=np.float32)
    # print("created east", flush=True)
    svf_s_result = np.zeros((height, width), dtype=np.float32)
    # print("created south", flush=True)
    svf_w_result = np.zeros((height, width), dtype=np.float32)
    # print("created west", flush=True)

    num_tiles = ((height // (TILE_SIZE - 2 * BORDER)) + 1) * ((width // (TILE_SIZE - 2 * BORDER)) + 1)
    tile_count = 0


    for i in range(0, height, TILE_SIZE - 2 * BORDER):
        for j in range(0, width, TILE_SIZE - 2 * BORDER):
            i_start = max(i - BORDER, 0)
            i_end = min(i + TILE_SIZE + BORDER, height)
            j_start = max(j - BORDER, 0)
            j_end = min(j + TILE_SIZE + BORDER, width)

            dsm_tile = dsm[i_start:i_end, j_start:j_end]

            tile_svf, tile_svf_n, tile_svf_e, tile_svf_s, tile_svf_w = parallel_SVF_calculate(dsm_tile, iazimuth, radius_start, rangeDist, radius_step_size)

            valid_i_start = BORDER if i > 0 else 0
            valid_j_start = BORDER if j > 0 else 0
            valid_i_end = TILE_SIZE if i + TILE_SIZE < height else i_end - i_start
            valid_j_end = TILE_SIZE if j + TILE_SIZE < width else j_end - j_start

            svf_valid = tile_svf[valid_i_start:valid_i_end, valid_j_start:valid_j_end]
            n_valid = tile_svf_n[valid_i_start:valid_i_end, valid_j_start:valid_j_end]
            e_valid = tile_svf_e[valid_i_start:valid_i_end, valid_j_start:valid_j_end]
            s_valid = tile_svf_s[valid_i_start:valid_i_end, valid_j_start:valid_j_end]
            w_valid = tile_svf_w[valid_i_start:valid_i_end, valid_j_start:valid_j_end]

            result_i_start = i
            result_j_start = j
            result_i_end = result_i_start + svf_valid.shape[0]
            result_j_end = result_j_start + svf_valid.shape[1]

            svf_result[result_i_start:result_i_end, result_j_start:result_j_end] = svf_valid
            svf_n_result[result_i_start:result_i_end, result_j_start:result_j_end] = n_valid
            svf_e_result[result_i_start:result_i_end, result_j_start:result_j_end] = e_valid
            svf_s_result[result_i_start:result_i_end, result_j_start:result_j_end] = s_valid
            svf_w_result[result_i_start:result_i_end, result_j_start:result_j_end] = w_valid

            tile_count += 1
            percent_done = (tile_count / num_tiles) * 100
            print(f"Progress: {tile_count}/{num_tiles} tiles processed, {percent_done:.2f}% completed.", flush=True)

    return svf_result, svf_n_result, svf_e_result, svf_s_result, svf_w_result

for demfile in demfiles:
    print(f"\nProcessing: {demfile}", flush=True)

    # Derive output dir from the input path (puts SVF/ next to cDSM.tif)
    scenario_dir = os.path.dirname(demfile)           # e.g. .../01_Input_S1_2
    OUTPUT_DIR = os.path.join(scenario_dir, "SVF")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load raster
    src_ds = gdal.Open(demfile)
    gt = src_ds.GetGeoTransform()
    rb = src_ds.GetRasterBand(1)
    arrayDEM = rb.ReadAsArray().astype(np.float32)

    print(arrayDEM.shape, flush=True)

    if np.isnan(arrayDEM).any():
        print(f"Skipping {demfile}: NaN values detected.", flush=True)
        continue

    arrayDEM = np.ascontiguousarray(arrayDEM)

    # Run SVF
    t_start = time.time()
    svf_total, svf_north, svf_east, svf_south, svf_west = process_large_svf(
        arrayDEM, scale, rangeDist, radius_start, radius_step_size
    )
    print(f"Done in {time.time() - t_start:.1f}s", flush=True)

    # Copy metadata and write outputs
    with rasterio.open(demfile) as src:
        meta = src.meta.copy()
    meta.update({"dtype": svf_total.dtype, "count": 1})

    suffix = "veg" if canopy else ""
    outputs = {
        os.path.join(OUTPUT_DIR, f"svf{suffix}.tif"):  svf_total,
        os.path.join(OUTPUT_DIR, f"svfN{suffix}.tif"): svf_north,
        os.path.join(OUTPUT_DIR, f"svfE{suffix}.tif"): svf_east,
        os.path.join(OUTPUT_DIR, f"svfS{suffix}.tif"): svf_south,
        os.path.join(OUTPUT_DIR, f"svfW{suffix}.tif"): svf_west,
    }

    for filename, data in outputs.items():
        with rasterio.open(filename, "w", **meta) as dst:
            dst.write(data, 1)
        print(f"  Written: {filename}", flush=True)
    
    suffix = "aveg" if canopy else ""
    outputs = {
        os.path.join(OUTPUT_DIR, f"svf{suffix}.tif"):  svf_total,
        os.path.join(OUTPUT_DIR, f"svfN{suffix}.tif"): svf_north,
        os.path.join(OUTPUT_DIR, f"svfE{suffix}.tif"): svf_east,
        os.path.join(OUTPUT_DIR, f"svfS{suffix}.tif"): svf_south,
        os.path.join(OUTPUT_DIR, f"svfW{suffix}.tif"): svf_west,
    }

    for filename, data in outputs.items():
        with rasterio.open(filename, "w", **meta) as dst:
            dst.write(data, 1)
        print(f"  Written: {filename}", flush=True)