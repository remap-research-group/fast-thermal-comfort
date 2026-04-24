#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 13:59:31 2021

@author: Jérémy Bernard, University of Gothenburg
"""
import pandas as pd
import geopandas as gpd
import shutil
import numpy as np
# from scipy.interpolate import griddata
# from rasterio.transform import from_origin
# import rasterio
from scipy.interpolate import griddata
from osgeo.gdal import GDT_Float32

from .DataUtil import radToDeg, windDirectionFromXY, createIndex, prefix
from .Obstacles import windRotation
from osgeo.osr import SpatialReference
from osgeo.gdal import Grid, GridOptions, FillNodata, Open, GA_Update, GetDriverByName, Warp, WarpOptions  # CHANGE: added Warp and WarpOptions
from .GlobalVariables import HORIZ_WIND_DIRECTION, HORIZ_WIND_SPEED, WIND_SPEED,\
    ID_POINT, TEMPO_DIRECTORY, TEMPO_HORIZ_WIND_FILE, VERT_WIND_SPEED, GEOM_FIELD,\
    OUTPUT_DIRECTORY, MESH_SIZE, OUTPUT_FILENAME, DELETE_OUTPUT_IF_EXISTS,\
    OUTPUT_RASTER_EXTENSION, OUTPUT_VECTOR_EXTENSION, OUTPUT_NETCDF_EXTENSION,\
    WIND_GROUP, WINDSPEED_PROFILE, RLON, RLAT, LON, LAT, LEVELS, WINDSPEED_X,\
    WINDSPEED_Y, WINDSPEED_Z, VERT_WIND, Z, OUTPUT_FILENAME, PREFIX_NAME,\
    BASE_HEIGHT_FIELD, HEIGHT_FIELD
from datetime import datetime
import netCDF4 as nc4
import os
import processing

def saveBasicOutputs(cursor, z_out, dz, u, v, w, gridName,
                     verticalWindProfile, outputFilePath, meshSize,
                     stacked_blocks, outputFilename = OUTPUT_FILENAME,
                     outputRaster = None, saveRaster = True,
                     saveVector = True, saveNetcdf = True,
                     prefix_name = PREFIX_NAME, tmp_dir = TEMPO_DIRECTORY):

    # Get the srid of the input geometry
    cursor.execute(""" SELECT ST_SRID({0}) AS srid FROM {1} LIMIT 1
                   """.format( GEOM_FIELD,
                               gridName))
    srid = cursor.fetchall()[0][0]
    
    # -------------------------------------------------------------------
    # SAVE NETCDF -------------------------------------------------------
    # ------------------------------------------------------------------- 
    final_netcdf_path = None
    if saveNetcdf:    
        # Get the coordinate in lat/lon of each point 
        # WARNING : for now keep the data in local coordinates)
        cursor.execute(""" 
           SELECT ST_X({0}) AS LON, ST_Y({0}) AS LAT FROM 
           (SELECT ST_TRANSFORM(ST_SETSRID({0},{2}), 4326) AS {0} FROM {1})
           """.format( GEOM_FIELD,
                       gridName,
                       srid))
        coord = np.array(cursor.fetchall())
        # Convert to a 2D (X, Y) array
        nx = u.shape[0]
        ny = u.shape[1]
        longitude = np.array([[coord[i * nx + j, 0] for i in range(ny)] for j in range(nx)])
        latitude = np.array([[coord[i * nx + j, 1] for i in range(ny)] for j in range(nx)])
        
    
        # Save the data into a NetCDF file
        # If delete = False, add a suffix to the file
        netcdf_base_dir_name = os.path.join(outputFilePath, 
                                            prefix(outputFilename, prefix_name))
        if os.path.isfile(netcdf_base_dir_name + OUTPUT_NETCDF_EXTENSION):
            if DELETE_OUTPUT_IF_EXISTS:
                os.remove(netcdf_base_dir_name + OUTPUT_NETCDF_EXTENSION)
            else:
                netcdf_base_dir_name = renameFileIfExists(filedir = netcdf_base_dir_name,
                                                          extension = OUTPUT_NETCDF_EXTENSION)  
        final_netcdf_path = saveToNetCDF(longitude = longitude,
                                         latitude = latitude,
                                         x = range(nx),
                                         y = range(ny),
                                         u = u,
                                         v = v,
                                         w = w,
                                         verticalWindProfile = verticalWindProfile,
                                         path = netcdf_base_dir_name,
                                         urock_srid = srid,
                                         horizontal_res = meshSize,
                                         vertical_res = dz)

    horizOutputUrock = {z_i : "HORIZ_OUTPUT_UROCK_{0}".format(str(z_i).replace(".","_")) for z_i in z_out}
    for z_i in z_out:
        # Keep only wind field for a single horizontal plan (and convert carthesian
        # wind speed into polar at least for horizontal)
        tempoTable = "TEMPO_HORIZ"
        if z_i % dz % (dz / 2) == 0:
            n_lev = int(z_i / dz) + 1
            ufin = u[:,:,n_lev]
            vfin = v[:,:,n_lev]
            wfin = w[:,:,n_lev]
        else:
            n_lev = int((z_i + dz /2) / dz)
            n_lev1 = n_lev + 1
            weight1 = (z_i - (n_lev - 0.5) * dz) / dz
            weight = 1 - weight1
            ufin = (weight * u[:,:,n_lev] + weight1 * u[:,:,n_lev1])
            vfin = (weight * v[:,:,n_lev] + weight1 * v[:,:,n_lev1])
            wfin = (weight * w[:,:,n_lev] + weight1 * w[:,:,n_lev1])
        df = pd.DataFrame({HORIZ_WIND_SPEED: ((ufin ** 2 + vfin ** 2) ** 0.5).flatten("F"),
                           WIND_SPEED: ((ufin ** 2 + vfin ** 2 + wfin ** 2) ** 0.5).flatten("F"), 
                           HORIZ_WIND_DIRECTION: radToDeg(windDirectionFromXY(ufin, vfin)).flatten("F"), 
                           VERT_WIND_SPEED: wfin.flatten("F")}).rename_axis(ID_POINT)
        
        # Load horizontal wind speed, wind direction and
        # vertical wind speed in a file containing a geometry field
        csv_tmp_file = os.path.join(tmp_dir, TEMPO_HORIZ_WIND_FILE)
        df.to_csv(csv_tmp_file)
        cursor.execute(
            """
            DROP TABLE IF EXISTS {9};
            CREATE TABLE {9}({3} INTEGER, {5} DOUBLE, {6} DOUBLE, {7} DOUBLE, {11} DOUBLE)
                AS SELECT {3}, {5}, {6}, {7}, {11} FROM CSVREAD('{10}');
            {0}{1}
            DROP TABLE IF EXISTS {2};
            CREATE TABLE {2}
                AS SELECT   a.{3}, {4}, b.{5}, 
                            b.{6}, b.{7}, b.{11}
                FROM {8} AS a
                LEFT JOIN {9} AS b
                ON a.{3} = b.{3}
            """.format(createIndex(tableName=gridName, 
                                            fieldName=ID_POINT,
                                            isSpatial=False),
                        createIndex(tableName=tempoTable, 
                                             fieldName=ID_POINT,
                                             isSpatial=False),
                        horizOutputUrock[z_i]       , ID_POINT,
                        GEOM_FIELD                  , HORIZ_WIND_SPEED,
                        HORIZ_WIND_DIRECTION        , VERT_WIND_SPEED,
                        gridName                    , tempoTable,
                        csv_tmp_file,
                        WIND_SPEED))
        
        # -------------------------------------------------------------------
        # SAVE VECTOR -------------------------------------------------------
        # ------------------------------------------------------------------- 
        if saveVector or saveRaster:
            outputDir_zi = os.path.join(outputFilePath, 
                                        "z" + str(z_i).replace(".","_"))
            if not os.path.exists(outputDir_zi):
                os.mkdir(outputDir_zi)
            outputVectorFile = saveTable(cursor = cursor,
                                         tableName = horizOutputUrock[z_i],
                                         filedir = os.path.join(outputDir_zi,
                                                                prefix(outputFilename, prefix_name)+\
                                                                OUTPUT_VECTOR_EXTENSION),
                                         delete = DELETE_OUTPUT_IF_EXISTS)
                
            # -------------------------------------------------------------------
            # SAVE RASTER -------------------------------------------------------
            # -------------------------------------------------------------------     
            if saveRaster:
                # Save the all direction, horizontal and vertical wind speeds into a a different raster
                for var in [WIND_SPEED, HORIZ_WIND_SPEED, VERT_WIND_SPEED]:
                    saveRasterFile(cursor = cursor, 
                                   outputVectorFile = outputVectorFile,
                                   outputFilePathAndNameBase = os.path.join(outputDir_zi,
                                                                            prefix(outputFilename, prefix_name)),
                                   horizOutputUrock = horizOutputUrock,
                                   outputRaster = outputRaster, 
                                   z_i = z_i, 
                                   meshSize = meshSize,
                                   var2save = var,
                                   stacked_blocks = stacked_blocks,
                                   srid = srid,
                                   tmp_dir = tmp_dir)   

    return horizOutputUrock, final_netcdf_path
    
def saveToNetCDF(longitude,
                 latitude,
                 x,
                 y,
                 u,
                 v,
                 w,
                 verticalWindProfile,
                 path,
                 urock_srid,
                 horizontal_res,
                 vertical_res):
    """
    Create a netCDF file and save wind speed, direction and initial 
    vertical wind profile in it (based on https://pyhogs.github.io/intro_netcdf4.html )
    
    Parameters
    _ _ _ _ _ _ _ _ _ _ 
        longitude: np.array (2D - X, Y)
            Longitude of each of the (X, Y) points
        latitude: np.array (2D - X, Y)
            Longitude of each of the (X, Y) points
        x: np.array (1D)
            X grid coordinates in local referential
        y: np.array (1D)
            Y grid coordinates in local referential
        u: np.array (3D)
            Wind speed along East axis
        v: 2D (X, Y) array
            Wind speed along North axis
        w: 2D (X, Y) array
            Wind speed along vertical axis
        verticalWindProfile: pd.DataFrame
            Initial wind speed profile for each each z from ground (2 columns)
        path: String
            Path and filename to save NetCDF file
        urock_srid: int
            EPSG code initially used for the URock calculations
    
    Returns
    -------
        String being the path, filename and extension of the netCdf file where
        are stored the results
    """    
    # Opens a netCDF file in writing mode ('w')
    f = nc4.Dataset(path + OUTPUT_NETCDF_EXTENSION,'w', format='NETCDF4')
    
    # 3D WIND SPEED DATA
    # Creates a group within this file for the 3D wind speed
    wind3dGrp = f.createGroup(WIND_GROUP)
    
    # Creates dimensions within this group
    wind3dGrp.createDimension('rlon', len(x))
    wind3dGrp.createDimension('rlat', len(y))
    wind3dGrp.createDimension('z', verticalWindProfile.index.size)
    
    # Build the variables
    rlon = wind3dGrp.createVariable(RLON, 'i4', 'rlon')
    rlat = wind3dGrp.createVariable(RLAT, 'i4', 'rlat')
    z = wind3dGrp.createVariable(Z, 'f4', 'z')
    lon = wind3dGrp.createVariable(LON, 'f8', ('rlon', 'rlat'))
    lat = wind3dGrp.createVariable(LAT, 'f8', ('rlon', 'rlat'))
    windSpeed_x = wind3dGrp.createVariable(WINDSPEED_X, 'f4', ('rlon', 'rlat', 'z'))
    windSpeed_y = wind3dGrp.createVariable(WINDSPEED_Y, 'f4', ('rlon', 'rlat', 'z'))  
    windSpeed_z = wind3dGrp.createVariable(WINDSPEED_Z, 'f4', ('rlon', 'rlat', 'z'))
    
    # Fill the variables
    rlon[:] = x
    rlat[:] = y
    z[:] = verticalWindProfile[Z].values
    lon[:,:] = longitude
    lat[:,:] = latitude
    windSpeed_x[:,:,:] = u
    windSpeed_y[:,:,:] = v
    windSpeed_z[:,:,:] = w
    
    # VERTICAL WIND PROFILE DATA
    # Creates a group within this file for the vertical wind profile
    vertWindProfGrp = f.createGroup(VERT_WIND)
    
    # Creates dimensions within this group
    vertWindProfGrp.createDimension('z', verticalWindProfile.index.size)
    
    # Build the variables 
    z_profile = vertWindProfGrp.createVariable(Z, 'f4', 'z')
    WindSpeed = vertWindProfGrp.createVariable(WINDSPEED_PROFILE, 'f4', 'z')
    
    # Fill the variables
    z_profile[:] = verticalWindProfile[Z].values
    WindSpeed[:] = verticalWindProfile[HORIZ_WIND_SPEED].values
    
    
    # ADD METADATA
    #Add local attributes to variable instances
    lon.units = 'degrees east'
    lat.units = 'degrees north'
    windSpeed_x.units = 'meter per second'
    windSpeed_y.units = 'meter per second'
    windSpeed_z.units = 'meter per second'
    z.units = 'meters'
    WindSpeed.units = 'meter per second'
    z_profile.units = 'meters'

    #Add global attributes
    f.description = "URock dataset containing one group of 3D wind field value and one group of input vertical wind speed profile"
    f.history = "Created " + datetime.today().strftime("%y-%m-%d")
    
    # Add the srid (epsg code) used for the URock processing calculation
    f.urock_srid = urock_srid
    
    # Add horizontal and vertical resolution into the metadata
    f.horizontal_res = horizontal_res
    f.vertical_res = vertical_res
    
    f.close()
    
    return path + OUTPUT_NETCDF_EXTENSION
    
def saveTable(cursor, tableName, filedir, delete = False, 
              rotationCenterCoordinates = None, rotateAngle = None):
    """ Save a table in .geojson or .shp (the table can be rotated before saving if needed).
    
    Parameters
	_ _ _ _ _ _ _ _ _ _ 
        cursor: conn.cursor
            A cursor object, used to perform spatial SQL queries
		tableName : String
			Name of the table to save
        filedir: String
            Directory (including filename and extension) of the file where to 
            store the table
        delete: Boolean, default False
            Whether or not the file is delete if exist
        rotationCenterCoordinates: tuple of float, default None
            x and y values of the point used as center of rotation
        rotateAngle: float, default None
            Counter clock-wise rotation angle (in degree)

    
    Returns
	_ _ _ _ _ _ _ _ _ _ 	
		output_filedir: String
            Directory (including filename and extension) of the saved file
            (could be different from input 'filedir' since the file may 
             have been renamed if exists)"""
    # Rotate the table if needed
    if rotationCenterCoordinates is not None and rotateAngle is not None:
        tableName = windRotation(cursor = cursor,
                                 dicOfInputTables = {tableName: tableName},
                                 rotateAngle = rotateAngle,
                                 rotationCenterCoordinates = rotationCenterCoordinates)[0][tableName]
    
    # Get extension
    extension = "." + filedir.split(".")[-1]
    filedirWithoutExt = ".".join(filedir.split(".")[0:-1])
    
    # Define the H2GIS function depending on extension
    if extension.upper() == ".GEOJSON":
        h2_function = "GEOJSONWRITE"
    elif extension.upper() == ".SHP":
        h2_function = "SHPWRITE"
    elif extension.upper() == ".FGB":
        h2_function = "FGBWRITE"
    else:
        print("The extension should be .geojson, .shp or .fgb")
    # Delete files if exists and delete = True
    if delete and os.path.isfile(filedir):
        output_filedir = filedir
        os.remove(filedir)
        if extension.upper() == ".SHP":
            os.remove(filedirWithoutExt+".dbf")
            os.remove(filedirWithoutExt+".shx")
            if os.path.isfile(filedirWithoutExt+".prj"):
                os.remove(filedirWithoutExt+".prj")
    # If delete = False, add a suffix to the file
    elif os.path.isfile(filedir):
        output_filedir = renameFileIfExists(filedir = filedirWithoutExt,
                                            extension = extension) + extension
    else:
        output_filedir = filedir
    # Write files
    cursor.execute("""CALL {0}('{1}','{2}')""".format(h2_function,
                                                      output_filedir,
                                                      tableName))
    return output_filedir

def renameFileIfExists(filedir, extension):
    """ Rename a file with a numbering prefix if exists.
    
    Parameters
	_ _ _ _ _ _ _ _ _ _ 
        filedir: String
            Directory (including filename but without extension) of the file
    
    Returns
	_ _ _ _ _ _ _ _ _ _ 	
		newFileDir: String
            Directory with renamed file"""
    i = 1
    newFileDir = filedir
    while(os.path.isfile(newFileDir + extension)):
        newFileDir = filedir + "({0})".format(i)
        i += 1
    return newFileDir


def saveRasterFile(cursor, outputVectorFile, outputFilePathAndNameBase, 
                   horizOutputUrock, outputRaster, z_i, meshSize, var2save,
                   stacked_blocks, srid, tmp_dir):
    """ Save results in a raster file.
    
    Parameters
	_ _ _ _ _ _ _ _ _ _ 
        outputFilePathAndNameBase: String
            Directory (including filename but without extension) of the file
    
    Returns
	_ _ _ _ _ _ _ _ _ _ 	
		None"""
    # Define output path name
    outputFilePathAndNameBaseRaster = outputFilePathAndNameBase + var2save
    # If delete = False, add a suffix to the filename
    if (os.path.isfile(outputFilePathAndNameBaseRaster + OUTPUT_RASTER_EXTENSION)) \
        and (not DELETE_OUTPUT_IF_EXISTS):
        outputFilePathAndNameBaseRaster = renameFileIfExists(filedir = outputFilePathAndNameBaseRaster,
                                                             extension = OUTPUT_RASTER_EXTENSION)
    
    # Whether or not a raster output is given as input, the rasterization process is slightly different
    if outputRaster:
        outputRasterExtent = outputRaster.extent()
        resX = (outputRasterExtent.xMaximum() - outputRasterExtent.xMinimum()) / outputRaster.width()
        resY = (outputRasterExtent.yMaximum() - outputRasterExtent.yMinimum()) / outputRaster.height()
        xmin = outputRasterExtent.xMinimum()
        ymax = outputRasterExtent.yMaximum()
        xmax = outputRasterExtent.xMaximum()
        ymin = outputRasterExtent.yMinimum()

        # CHANGE: explicitly store the target template dimensions so the final raster
        # is forced to exactly match the supplied template raster
        target_width = int(outputRaster.width())
        target_height = int(outputRaster.height())
        target_bounds = (xmin, ymin, xmax, ymax)

        tmp_file = os.path.join(tmp_dir, f"interp_before_fillna_{var2save}.tif")
        # If a single output raster cell contains more than 4 points, average instead of interpolate
        if resX * resY > 4 * meshSize**2:
            Grid(destName = tmp_file,
                 srcDS = outputVectorFile,
                 options = GridOptions(format = "GTiff",
                                       zfield = var2save, 
                                       width = target_width,     # CHANGE: force exact template width
                                       height = target_height,   # CHANGE: force exact template height
                                       outputBounds = [xmin,     # CHANGE: corrected GDAL bounds order
                                                       ymin,
                                                       xmax,
                                                       ymax],
                                       algorithm = "average:radius1={0}:radius2={0}".format(1.1*meshSize)))
        else:
            # Interpolate with building constraints
            interp_vec_to_rast(outputVectorFile = outputVectorFile, 
                               stacked_blocks = stacked_blocks,
                               outputFilePathAndNameBaseRaster = ".".join(tmp_file.split(".")[0:-1]), 
                               extent = f'{xmin},{xmax},{ymin},{ymax} [EPSG:{srid}]',
                               resX = resX, 
                               resY = resY,
                               z_i = z_i,
                               colname = var2save,
                               tmp_dir = tmp_dir,
                               target_width = target_width,      # CHANGE: pass exact target width
                               target_height = target_height,    # CHANGE: pass exact target height
                               target_bounds = target_bounds,    # CHANGE: pass exact target bounds
                               target_srid = srid)               # CHANGE: pass target CRS
            
        # Interpolate values to fill no data
        ds = Open(tmp_file) 
        driver = GetDriverByName("GTiff")
        output_ds = driver.CreateCopy(outputFilePathAndNameBaseRaster + OUTPUT_RASTER_EXTENSION,
                                      ds)
        band = output_ds.GetRasterBand(1)
        _ = FillNodata(targetBand = band, maskBand = None, 
                       maxSearchDist = 9999, smoothingIterations = 0)
        
        # Set the srid
        vec_srid = gpd.read_file(outputVectorFile, rows = slice(0,)).crs.to_epsg()
        srs = SpatialReference()
        srs.ImportFromEPSG(vec_srid)
        output_ds.SetProjection(srs.ExportToWkt())
        
        # Release the datasets.
        ds = ds.FlushCache()
        output_ds = output_ds.FlushCache()
        
        # ET = Open(outputFilePathAndNameBaseRaster + OUTPUT_RASTER_EXTENSION, GA_Update) 
        # ETband = ET.GetRasterBand(1)
 
        # FillNodata(targetBand = ETband, maskBand = None, 
        #            maxSearchDist = 999, smoothingIterations = 0)
        # ET = None
    else:
        cursor.execute(
            """
            SELECT  ST_XMIN({0}) AS XMIN, ST_XMAX({0}) AS XMAX,
                    ST_YMIN({0}) AS YMIN, ST_YMAX({0}) AS YMAX
            FROM    (SELECT ST_ACCUM({0}) AS {0} FROM {1})
            """.format(GEOM_FIELD            , horizOutputUrock[z_i]))
        vectorBounds = cursor.fetchall()[0]

        # CHANGE: use round(...) before int(...) so tiny floating point error does not
        # create an extra row/column
        width = int(round((vectorBounds[1] - vectorBounds[0]) / meshSize)) + 1
        height = int(round((vectorBounds[3] - vectorBounds[2]) / meshSize)) + 1

        xmin = vectorBounds[0] - float(meshSize) / 2
        ymax = vectorBounds[3] + float(meshSize) / 2
        xmax = vectorBounds[1] + float(meshSize) / 2  # CHANGE: use xmax bound directly
        ymin = vectorBounds[2] - float(meshSize) / 2  # CHANGE: fixed off-by-one-cell ymin
        
        # Interpolate with building constraints
        interp_vec_to_rast(outputVectorFile = outputVectorFile, 
                           stacked_blocks = stacked_blocks,
                           outputFilePathAndNameBaseRaster = outputFilePathAndNameBaseRaster, 
                           extent = f'{xmin},{xmax},{ymin},{ymax} [EPSG:{srid}]',
                           resX = meshSize, 
                           resY = meshSize,
                           z_i = z_i,
                           colname = var2save,
                           tmp_dir = tmp_dir,
                           target_width = width,               # CHANGE: pass exact computed width
                           target_height = height,             # CHANGE: pass exact computed height
                           target_bounds = (xmin, ymin, xmax, ymax),  # CHANGE
                           target_srid = srid)                # CHANGE

def interp_vec_to_rast(outputVectorFile, stacked_blocks, outputFilePathAndNameBaseRaster, 
                       extent, resX, resY, z_i, colname, tmp_dir,
                       target_width = None, target_height = None,
                       target_bounds = None, target_srid = None):
    """ Interpolate and save wind data saved in a vector to a raster.
       
    Parameters
   	_ _ _ _ _ _ _ _ _ _ 
           outputVectorFile: String
               Directory (including filename and extension) of the file containing the 
               wind field to interpolate from vector to raster
           stacked_blocks: String
               Directory (including filename only - no extension) of the file containing stacked blocks 
               (building footprint + where it starts and ends vertically)
           outputFilePathAndNameBaseRaster: String
               Directory (including filename and extension) of the file that will 
               be used to save the results
           extent: String
               QGIS extent in the shape '{xmin},{xmax},{ymin},{ymax} [EPSG:epsgcode]'
           resX: float
               Resolution of the output raster along the X axis
           resY: float
               Resolution of the output raster along the Y axis
           z_i: float
               Height of the output wind field
           colname: string
               Name of the attribute column in teh vector table to convert to raster
           tmp_dir: String
               Path of the directory used for saving temporary results (should be unique)
           
       
   Returns
   	_ _ _ _ _ _ _ _ _ _ 	
           output_file_path: String
               Path and name of the file containing the resulting raster"""
    # gdf = gpd.read_file(outputVectorFile)
    
    # coordinates = np.array([point.coords[0] for point in gdf.geometry])  # [x, y]
    # values = gdf[colname].values  # The field 'V' to interpolate
    
    # # Define the raster grid (3m resolution)
    # resolution = min([resX, resY])
    # xmin, ymin, xmax, ymax = gdf.total_bounds
    # xmin -= resolution / 2
    # ymin -= resolution / 2
    # xmax -= resolution / 2
    # ymax += resolution / 2
    
    # # Create a grid of coordinates for interpolation
    # x_grid = np.arange(xmin, xmax, resolution)
    # y_grid = np.arange(ymin, ymax, resolution)
    # x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    
    # # Perform bilinear interpolation on the grid
    # grid_values = griddata(coordinates, values, (x_grid, y_grid), method='linear')
    # grid_values = grid_values[::-1, :]
    
    # # Define raster metadata
    # transform = from_origin(xmin, ymax, resolution, resolution)  # origin is top-left
    
    # # Save the interpolated grid to a raster file
    # interp_out = os.path.join(TEMPO_DIRECTORY,"interp_out.tif")
    # with rasterio.open(interp_out, 'w', driver='GTiff', 
    #                     height=grid_values.shape[0], width=grid_values.shape[1],
    #                     count=1, dtype=grid_values.dtype, crs=gdf.crs, transform=transform) as dst:
    #     dst.write(grid_values, 1)
    
    # Change the order of the points to make the TIN interpolation faster and working for all conditions
    order_changed = processing.run("native:orderbyexpression", 
                                   {'INPUT':outputVectorFile,
                                    'EXPRESSION':'randf(0,1)',
                                    'ASCENDING':False,
                                    'NULLS_FIRST':False,
                                    'OUTPUT':os.path.join(tmp_dir,
                                                          f"order_changed_{colname}")})["OUTPUT"]
    
    # Get the column number corresponding to the column name
    colnb = gpd.read_file(order_changed, rows = slice(0,)).columns.get_loc(colname) + 1
    
    # Interpolate the results without constraints
    interp_out = processing.run("qgis:tininterpolation",
                               {'INTERPOLATION_DATA':f'{order_changed}::~::0::~::{colnb}::~::0',
                                'METHOD':0,
                                'EXTENT':extent,
                                'PIXEL_SIZE':min(resX, resY),
                                'OUTPUT':os.path.join(tmp_dir,
                                                      f"interp_out_{colname}.tif")})["OUTPUT"]
    
    # If there are buildings in the study area, need to set wind speed = 0
    #if os.stat(stacked_blocks).st_size > 0:
    if False:
        # Rasterize the stacked blocks keeping the value of each stacked block base 
        block_base = processing.run("gdal:rasterize",
                                    {'INPUT':stacked_blocks,
                                     'FIELD':BASE_HEIGHT_FIELD,'BURN':0,'USE_Z':False,
                                     'UNITS':1,'WIDTH':resX,'HEIGHT':resY,
                                     'EXTENT':extent,
                                     'NODATA':None,'OPTIONS':'','DATA_TYPE':5,
                                     'INIT':-9999,'INVERT':False,'EXTRA':'',
                                     'OUTPUT':os.path.join(tmp_dir,
                                                           f"block_base_{colname}.tif")})["OUTPUT"]
        
        # Rasterize the stacked blocks keeping the value of each stacked block top 
        block_top = processing.run("gdal:rasterize",
                                   {'INPUT':stacked_blocks,
                                    'FIELD':HEIGHT_FIELD,'BURN':0,'USE_Z':False,
                                    'UNITS':1,'WIDTH':resX,'HEIGHT':resY,
                                    'EXTENT':extent,
                                    'NODATA':None,'OPTIONS':'','DATA_TYPE':5,
                                    'INIT':-9999,'INVERT':False,'EXTRA':'',
                                    'OUTPUT':os.path.join(tmp_dir,
                                                          f"block_top_{colname}.tif")})["OUTPUT"]
        
        # Keep the values only when there is no building at this position
        output_file_path = processing.run("gdal:rastercalculator",
                                          {'INPUT_A':block_base,'BAND_A':1,
                                           'INPUT_B':block_top,'BAND_B':1,
                                           'INPUT_C':interp_out,'BAND_C':1,
                                           'INPUT_D':None,'BAND_D':None,
                                           'INPUT_E':None,'BAND_E':None,
                                           'INPUT_F':None,'BAND_F':None,
                                           'FORMULA':f'((A == -9999) + (A < {z_i})) * ((B == -9999) + (B < {z_i})) * C',
                                           'NO_DATA':None,'EXTENT_OPT':0,'PROJWIN':None,
                                           'RTYPE':5,'OPTIONS':'','EXTRA':'',
                                           'OUTPUT':outputFilePathAndNameBaseRaster + OUTPUT_RASTER_EXTENSION})["OUTPUT"]   
    # Else directly save the result of the interpolation
    else:
        output_file_path = outputFilePathAndNameBaseRaster + OUTPUT_RASTER_EXTENSION

        # CHANGE: qgis:tininterpolation does not reliably honor the intended raster
        # dimensions from extent/pixel size alone. Force the interpolated raster onto
        # the exact target grid so the final output always matches the template.
        if target_width is not None and target_height is not None and target_bounds is not None:
            xmin, ymin, xmax, ymax = target_bounds

            aligned = Warp(output_file_path,
                           interp_out,
                           options = WarpOptions(format = "GTiff",
                                                outputBounds = (xmin, ymin, xmax, ymax),
                                                width = int(target_width),
                                                height = int(target_height),
                                                dstSRS = f"EPSG:{int(target_srid)}" if target_srid is not None else None,
                                                resampleAlg = "bilinear",
                                                multithread = True))
            aligned = None
        else:
            shutil.copy2(src = interp_out, dst = output_file_path)
    
    return output_file_path
        
def saveRockleZones(cursor, outputDataAbs, dicOfBuildZoneGridPoint, dicOfVegZoneGridPoint,
                    gridPoint, rotationCenterCoordinates, windDirection):
    """ Save the 2D Röckle zones (building and vegetation) as points.
    
    Parameters
	_ _ _ _ _ _ _ _ _ _ 
        cursor: conn.cursor
            A cursor object, used to perform spatial SQL queries
		outputDataAbs : Dictionary
			Object containing the absolute path where should be saved the Röckle points
        dicOfBuildZoneGridPoint: Dictionary
            Dictionary containing all building table names to be saved
        dicOfVegZoneGridPoint: Dictionary
            Dictionary containing all vegetation table names to be saved
        gridPoint: String
            Grid table name
        rotationCenterCoordinates: tuple of float, default None
            x and y values of the point used as center of rotation
        windDirection: float, default None
            Wind direction used for calculation (° clock-wise from North)
        
    
    Returns
	_ _ _ _ _ _ _ _ _ _ 	
		None"""
    # Creates a folder if not exist
    if not os.path.exists(outputDataAbs["point_2DRockleZone"]):
        os.mkdir(outputDataAbs["point_2DRockleZone"])
    # Save Building Röckle zones
    for t in dicOfBuildZoneGridPoint:
        cursor.execute("""
           DROP TABLE IF EXISTS point_Buildzone_{0};
           {5};
           {6};
           CREATE TABLE point_Buildzone_{0}
               AS SELECT   a.{2}, b.*
               FROM {3} AS a RIGHT JOIN {4} AS b
                   ON a.{1} = b.{1}
               WHERE b.{1} IS NOT NULL
           """.format( t                            , ID_POINT, 
                       GEOM_FIELD                   , gridPoint, 
                       dicOfBuildZoneGridPoint[t]   , createIndex(tableName=gridPoint, 
                                                                  fieldName=ID_POINT,
                                                                  isSpatial=False),
                       createIndex(tableName=dicOfBuildZoneGridPoint[t], 
                                   fieldName=ID_POINT,
                                   isSpatial=False)))
        saveTable(cursor = cursor,
                  tableName = "point_Buildzone_"+t,
                  filedir = os.path.join(outputDataAbs["point_2DRockleZone"], t+".geojson"),
                  delete = True,
                  rotationCenterCoordinates = rotationCenterCoordinates,
                  rotateAngle = - windDirection)
    
    # Save vegetation Röckle zones
    for t in dicOfVegZoneGridPoint:
        saveTable(cursor = cursor,
                  tableName = dicOfVegZoneGridPoint[t],
                  filedir = os.path.join(outputDataAbs["point_2DRockleZone"], t+".geojson"),
                  delete = True,
                  rotationCenterCoordinates = rotationCenterCoordinates,
                  rotateAngle = - windDirection)

def saveRasterOutputsDirect(cursor, gridName, z_out, dz, u, v, w,
                            outputFilePath, meshSize,
                            outputFilename=OUTPUT_FILENAME,
                            outputRaster=None,
                            prefix_name=PREFIX_NAME,
                            tmp_dir=TEMPO_DIRECTORY,
                            srid=None,
                            nodata_value=-9999.0):
    """
    Save north-up raster outputs directly from NumPy arrays, while preserving
    the important alignment logic of the original save pipeline.

    Key difference from the previous direct version:
    - values are aligned to geometry using gridName/ID_POINT exactly like
      saveBasicOutputs does, instead of assuming x_rot/y_rot flattening is
      spatially identical to the saved grid geometry.

    Parameters
    ----------
    cursor : H2 cursor
    gridName : str
        Name of the rotated output grid table (the same one passed to
        saveBasicOutputs in section 12, i.e. rotated_grid).
    z_out : iterable
    dz : float
    u, v, w : np.ndarray
        Rotated-back wind fields to save.
    outputFilePath : str
    meshSize : float
    outputFilename : str
    outputRaster : QgsRasterLayer or None
    prefix_name : str
    tmp_dir : str
    srid : int or None
    nodata_value : float

    Returns
    -------
    dict
        {z_i: {var_name: output_path, ...}, ...}
    """
    import os
    import json
    import numpy as np
    import pandas as pd
    from osgeo import gdal
    from osgeo.osr import SpatialReference

    if u.shape != v.shape or u.shape != w.shape:
        raise ValueError("u, v, and w must have the same shape")

    nx, ny, nz = u.shape
    n_points_expected = nx * ny

    # ------------------------------------------------------------
    # 1) Get authoritative output geometry from rotated_grid/gridName
    #    ordered by ID_POINT, exactly matching saveBasicOutputs logic.
    # ------------------------------------------------------------
    if srid is None:
        cursor.execute(f"""
            SELECT ST_SRID({GEOM_FIELD}) AS srid
            FROM {gridName}
            LIMIT 1
        """)
        srid = cursor.fetchall()[0][0]

    # create the index first in its own statement
    idx_sql = createIndex(tableName=gridName, fieldName=ID_POINT, isSpatial=False)
    if idx_sql and idx_sql.strip():
        cursor.execute(idx_sql)

    # then run a pure SELECT
    cursor.execute(f"""
        SELECT {ID_POINT},
            ST_X({GEOM_FIELD}) AS X,
            ST_Y({GEOM_FIELD}) AS Y
        FROM {gridName}
        ORDER BY {ID_POINT}
    """)
    coord_rows = cursor.fetchall()

    if len(coord_rows) != n_points_expected:
        raise ValueError(
            f"gridName has {len(coord_rows)} points but wind arrays expect "
            f"{n_points_expected} (= {nx}*{ny})"
        )

    coords = np.asarray(coord_rows, dtype=np.float64)
    id_points = coords[:, 0].astype(np.int64)
    x_points = coords[:, 1]
    y_points = coords[:, 2]

    # Sanity check: saveBasicOutputs assumes the flattened arrays align
    # with ID_POINT row order.
    if not np.array_equal(id_points, np.arange(n_points_expected)):
        raise ValueError(
            "ID_POINT ordering in gridName is not 0..N-1. "
            "This direct saver assumes the same ID_POINT alignment used by saveBasicOutputs."
        )

    # ------------------------------------------------------------
    # 2) Helper to match the original z-slice logic exactly
    # ------------------------------------------------------------
    def _extract_level(z_i):
        if z_i % dz % (dz / 2) == 0:
            n_lev = int(z_i / dz) + 1
            if n_lev >= nz:
                raise IndexError(f"Requested z_out={z_i} maps to n_lev={n_lev}, but nz={nz}")
            ufin = u[:, :, n_lev]
            vfin = v[:, :, n_lev]
            wfin = w[:, :, n_lev]
        else:
            n_lev = int((z_i + dz / 2) / dz)
            n_lev1 = n_lev + 1
            if n_lev1 >= nz:
                raise IndexError(
                    f"Requested z_out={z_i} maps to levels {n_lev}, {n_lev1}, but nz={nz}"
                )
            weight1 = (z_i - (n_lev - 0.5) * dz) / dz
            weight = 1 - weight1
            ufin = weight * u[:, :, n_lev] + weight1 * u[:, :, n_lev1]
            vfin = weight * v[:, :, n_lev] + weight1 * v[:, :, n_lev1]
            wfin = weight * w[:, :, n_lev] + weight1 * w[:, :, n_lev1]
        return ufin, vfin, wfin

    # ------------------------------------------------------------
    # 3) Helper: prepare output path like original code
    # ------------------------------------------------------------
    def _prepare_output_path(base_without_ext):
        out_path = base_without_ext + OUTPUT_RASTER_EXTENSION
        if os.path.isfile(out_path):
            if DELETE_OUTPUT_IF_EXISTS:
                os.remove(out_path)
            else:
                out_path = (
                    renameFileIfExists(base_without_ext, OUTPUT_RASTER_EXTENSION)
                    + OUTPUT_RASTER_EXTENSION
                )
        return out_path

    # ------------------------------------------------------------
    # 4) Helper: build a GDAL VRT over a CSV of XYZ values
    #    This avoids H2/vector export, but still lets us interpolate
    #    point data to a north-up raster like the original saver.
    # ------------------------------------------------------------
    def _write_xyz_csv_and_vrt(df_xyz, csv_path, vrt_path):
        df_xyz.to_csv(csv_path, index=False)

        # OGR CSV datasource layer name = file basename without extension
        src_layer_name = os.path.splitext(os.path.basename(csv_path))[0]

        vrt_xml = f"""<OGRVRTDataSource>
        <OGRVRTLayer name="points">
            <SrcDataSource>{csv_path}</SrcDataSource>
            <SrcLayer>{src_layer_name}</SrcLayer>
            <GeometryType>wkbPoint</GeometryType>
            <LayerSRS>EPSG:{int(srid)}</LayerSRS>
            <GeometryField encoding="PointFromColumns" x="X" y="Y"/>
        </OGRVRTLayer>
    </OGRVRTDataSource>
    """
        with open(vrt_path, "w", encoding="utf-8") as f:
            f.write(vrt_xml)

    # ------------------------------------------------------------
    # 5) Helper: determine raster grid exactly like original saver
    # ------------------------------------------------------------
    def _get_grid_params():
        if outputRaster:
            ext = outputRaster.extent()
            xmin = float(ext.xMinimum())
            xmax = float(ext.xMaximum())
            ymin = float(ext.yMinimum())
            ymax = float(ext.yMaximum())
            width = int(outputRaster.width())
            height = int(outputRaster.height())
            resX = (xmax - xmin) / width
            resY = (ymax - ymin) / height
        else:
            cursor.execute(f"""
                SELECT  ST_XMIN({GEOM_FIELD}) AS XMIN,
                        ST_XMAX({GEOM_FIELD}) AS XMAX,
                        ST_YMIN({GEOM_FIELD}) AS YMIN,
                        ST_YMAX({GEOM_FIELD}) AS YMAX
                FROM (SELECT ST_ACCUM({GEOM_FIELD}) AS {GEOM_FIELD}
                    FROM {gridName})
            """)
            xmin_pt, xmax_pt, ymin_pt, ymax_pt = map(float, cursor.fetchall()[0])

            width = int((xmax_pt - xmin_pt) / meshSize) + 1
            height = int((ymax_pt - ymin_pt) / meshSize) + 1

            xmin = xmin_pt - float(meshSize) / 2
            ymax = ymax_pt + float(meshSize) / 2
            xmax = xmin_pt + meshSize * (width - 0.5)
            ymin = ymax_pt - meshSize * (height + 0.5)

            resX = float(meshSize)
            resY = float(meshSize)

        return xmin, xmax, ymin, ymax, width, height, resX, resY

    # ------------------------------------------------------------
    # 6) Helper: interpolate to north-up raster
    #    We mimic the original behavior:
    #    - coarse raster: average nearby points
    #    - otherwise: linear interpolation + FillNodata
    # ------------------------------------------------------------
    def _grid_points_to_raster(vrt_path, out_path, var_name):
        xmin, xmax, ymin, ymax, width, height, resX, resY = _get_grid_params()

        tmp_interp = os.path.join(tmp_dir, f"interp_before_fillna_{var_name}.tif")

        if os.path.exists(tmp_interp):
            os.remove(tmp_interp)

        if outputRaster and (resX * resY > 4 * meshSize ** 2):
            # same branching idea as original saveRasterFile
            grid_options = gdal.GridOptions(
                format="GTiff",
                zfield=var_name,
                width=width,
                height=height,
                outputBounds=[xmin, ymax, xmax, ymin],
                algorithm=f"average:radius1={1.1 * meshSize}:radius2={1.1 * meshSize}",
            )
        else:
            # closest analogue to the original TIN-based interpolation path
            grid_options = gdal.GridOptions(
                format="GTiff",
                zfield=var_name,
                width=width,
                height=height,
                outputBounds=[xmin, ymax, xmax, ymin],
                algorithm=f"linear:radius=-1:nodata={nodata_value}",
            )

        ds_interp = gdal.Grid(destName=tmp_interp, srcDS=vrt_path, options=grid_options)
        if ds_interp is None:
            raise RuntimeError(f"GDAL Grid failed for {var_name}")
        ds_interp = None

        ds = gdal.Open(tmp_interp)
        if ds is None:
            raise RuntimeError(f"Could not open temporary interpolated raster {tmp_interp}")

        driver = gdal.GetDriverByName("GTiff")
        output_ds = driver.CreateCopy(
            out_path,
            ds,
            options=["COMPRESS=LZW", "TILED=YES"]
        )

        band = output_ds.GetRasterBand(1)
        band.SetNoDataValue(float(nodata_value))
        _ = gdal.FillNodata(
            targetBand=band,
            maskBand=None,
            maxSearchDist=9999,
            smoothingIterations=0
        )

        srs = SpatialReference()
        srs.ImportFromEPSG(int(srid))
        output_ds.SetProjection(srs.ExportToWkt())

        output_ds.FlushCache()
        ds = None
        output_ds = None

    # ------------------------------------------------------------
    # 7) Main save loop
    # ------------------------------------------------------------
    raster_outputs = {}

    for z_i in z_out:
        ufin, vfin, wfin = _extract_level(z_i)

        # EXACT same flattening convention as saveBasicOutputs
        flat_horiz_speed = np.sqrt(ufin ** 2 + vfin ** 2).flatten("F")
        flat_total_speed = np.sqrt(ufin ** 2 + vfin ** 2 + wfin ** 2).flatten("F")
        flat_vert_speed = wfin.flatten("F")

        df_xyz = pd.DataFrame({
            ID_POINT: id_points,
            "X": x_points,
            "Y": y_points,
            HORIZ_WIND_SPEED: flat_horiz_speed,
            WIND_SPEED: flat_total_speed,
            VERT_WIND_SPEED: flat_vert_speed,
        })

        outputDir_zi = os.path.join(outputFilePath, "z" + str(z_i).replace(".", "_"))
        if not os.path.exists(outputDir_zi):
            os.mkdir(outputDir_zi)

        base_name = os.path.join(outputDir_zi, prefix(outputFilename, prefix_name))
        raster_outputs[z_i] = {}

        for var_name in [WIND_SPEED, HORIZ_WIND_SPEED, VERT_WIND_SPEED]:
            csv_path = os.path.join(
                tmp_dir,
                f"urock_direct_points_{str(z_i).replace('.', '_')}_{var_name}.csv"
            )
            vrt_path = os.path.join(
                tmp_dir,
                f"urock_direct_points_{str(z_i).replace('.', '_')}_{var_name}.vrt"
            )
            _write_xyz_csv_and_vrt(
                df_xyz[["X", "Y", var_name]],
                csv_path,
                vrt_path
            )

            out_path = _prepare_output_path(base_name + var_name)
            _grid_points_to_raster(vrt_path, out_path, var_name)
            raster_outputs[z_i][var_name] = out_path

    return raster_outputs