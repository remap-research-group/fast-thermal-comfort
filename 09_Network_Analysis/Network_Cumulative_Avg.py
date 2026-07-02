import geopandas as gpd
import ast
import numpy as np
import pandas as pd
import warnings
from shapely.ops import unary_union
from shapely.geometry import MultiLineString
import os

# Load data (assuming your paths are correct in your environment)
heat_df = gpd.read_file("/media/remap/NO_HEAT_RB/Metro_Boston/Final/Heat_Risk_Analysis/02_Journal/segment_heat/2023_Q4_network_segments_heat.geojson")
trip_base = gpd.read_file("/media/remap/NO_HEAT_RB/Metro_Boston/Final/Heat_Risk_Analysis/02_Journal/demo/demo_trip.csv")
os.makedirs('/media/remap/NO_HEAT_RB/Metro_Boston/Final/Heat_Risk_Analysis/02_Journal/05_calculated_demo',exist_ok=True)

# Ensure hours are integers
trip_df = trip_base.copy()
trip_df['traversal_hour'] = trip_df['traversal_hour'].astype(float).astype('int64')

# NOTE: link_data_map is expected to be a global variable, initialized before the loops.
link_data_map = {} 

def read_heat_df(value_col):
    link_data_map_local = heat_df.set_index('stableEdgeId')[['geometry', value_col]].apply(
        lambda row: (row['geometry'].length, row[value_col]), axis=1
    ).to_dict()
    return link_data_map_local

def calculate_trip_utci(row, value_name):
    """
    Calculates sum_VALUE and avg_VALUE for a single trip row.
    The global link_data_map must be set before calling this.
    """
    traversal_lists_str = row['link_id_traversals']
    trip_mode = row['mode'] 

    try:
        traversal_lists = ast.literal_eval(traversal_lists_str)
    except (ValueError, SyntaxError):
        # Fallback for unparseable list strings
        print(f"Warning: Could not parse traversal_lists_str for trip mode {trip_mode}: {traversal_lists_str}")
        return {f'sum_{value_name}': 0, f'avg_{value_name}': 0, 'missing_link': []}

    tot_value = 0
    length_link = 0
    missing_links_info = []

    for link in traversal_lists:
        # Rely on the global link_data_map being set correctly
        if link in link_data_map:
            length, value = link_data_map[link]
            if length > 0:
                tot_value += length * value
                length_link += length
            else:
                missing_links_info.append(link)
        else:
            missing_links_info.append(link)
    # Edited
    # If any link is missing, return NaN for the values
    # if missing_links_info:
    #     return {f'sum_{value_name}': np.nan, f'avg_{value_name}': np.nan, 'missing_link': missing_links_info}

    avg_value = tot_value / length_link if length_link > 0 else 0

    return {f'sum_{value_name}': tot_value, f'avg_{value_name}': avg_value, 'length': length_link, 'missing_link': missing_links_info}
    
def create_trip_geometry(row, geom_map):
    """
    Looks up the geometries for a trip's link_id_traversals
    and merges them into a single geometry.
    """
    traversal_lists_str = row['link_id_traversals']
    try:
        traversal_list = ast.literal_eval(traversal_lists_str)
    except (ValueError, SyntaxError):
        # Return None if the list string is unparseable
        return None 
    
    geometries = []
    for link_id in traversal_list:
        # Get the geometry from our lookup map
        geom = geom_map.get(link_id)
        if geom:
            geometries.append(geom)
    
    if not geometries:
        # Return None if no geometries were found (e.g., all missing links)
        return None 
    
    # Use unary_union to merge all linestrings into one geometry.
    # This is robust and handles overlapping/touching segments.
    try:
        # Suppress RuntimeWarnings that can occur with unary_union
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return unary_union(geometries)
    except Exception as e:
        print(f"Warning: Could not create unary_union for activity_id {row.get('activity_id')}: {e}")
        # Fallback to MultiLineString if unary_union fails
        return MultiLineString(geometries)

# =================================================
# --- UTCI Calculation (from original script) ---
# =================================================

utci_list = ['UTCI_mp', 'UTCI_ap', 'UTCI_ep']
all_trip_results = pd.DataFrame()
all_missing_links = []

for utci in utci_list:
    # Set the global link_data_map for the current UTCI variable
    link_data_map = read_heat_df(utci)
    
    if utci == 'UTCI_mp':
        peak_hours = [8,9,10]
    elif utci == 'UTCI_ap':
        peak_hours = [12,13,14]
    elif utci == 'UTCI_ep':
        peak_hours = [16,17,18]

    trip_for_utci_calculation = trip_df[trip_df['traversal_hour'].isin(peak_hours)].copy()
    trip_for_utci_calculation['peak_hour'] = utci[5:]
    
    if not trip_for_utci_calculation.empty:
        # Pass the UTCI variable name to calculate_trip_utci
        results = trip_for_utci_calculation.apply(
            calculate_trip_utci, axis=1, result_type='expand', args=(utci,)
        )
        
        # Rename to 'sum_UTCI' and 'avg_UTCI' for consistency with original output
        results = results.rename(columns={f'sum_{utci}': 'sum_UTCI', f'avg_{utci}': 'avg_UTCI'})
        
        trip_for_utci_calculation['sum_UTCI'] = results['sum_UTCI']
        trip_for_utci_calculation['avg_UTCI'] = results['avg_UTCI']
        trip_for_utci_calculation['length'] = results['length'] # Edited
        
        if all_trip_results.empty:
            all_trip_results = trip_for_utci_calculation
        else:
            all_trip_results = pd.concat([all_trip_results, trip_for_utci_calculation])

        for sublist in results['missing_link']:
            if isinstance(sublist, list):
                all_missing_links.extend(sublist)

# =================================================
# --- NEW LST, NDVI, EVI Calculation ---
# =================================================

value_list = ['LST_Median', 'NDVI', 'EVI']

for value_col in value_list:
    # Set the global link_data_map for the current variable
    link_data_map = read_heat_df(value_col)
    
    # Apply calculation to the *entire* trip_df (no time filter)
    results = trip_df.apply(
        calculate_trip_utci, axis=1, result_type='expand', args=(value_col,)
    )

    # Assign the results to new, clearly named columns in trip_df
    sum_col = f'sum_{value_col}'
    avg_col = f'avg_{value_col}'
    
    trip_df[sum_col] = results[sum_col]
    trip_df[avg_col] = results[avg_col]
    
    # Collect missing links
    for sublist in results['missing_link']:
        if isinstance(sublist, list):
            all_missing_links.extend(sublist)

# =================================================
# --- Final Saving ---
# =================================================
utci_to_merge = all_trip_results[['activity_id', 'sum_UTCI', 'avg_UTCI', 'length', 'peak_hour']]

trip_df_final = pd.merge(
    trip_df, 
    utci_to_merge, 
    on='activity_id', 
    how='left'
)

if all_missing_links:
    unique_missing_links = list(set(all_missing_links))
    print(f"Total unique missing links across all calculations: {len(unique_missing_links)}")
    missing_dic = {'missing_link': unique_missing_links}
    missing_df = pd.DataFrame(missing_dic)


missing_df.to_csv('/media/remap/NO_HEAT_RB/Metro_Boston/Final/Heat_Risk_Analysis/02_Journal/2023_missing_links_all_0112.csv', index=False)
    # The file 2023_missing_links_all.csv has been created with all unique missing link IDs.

# Save the final enriched trip_df (full dataset with LST, NDVI, EVI values)
# You can use the trip_id to merge this back with the all_trip_results dataframe for a complete dataset.
final_output_path = '/media/remap/NO_HEAT_RB/Metro_Boston/Final/Heat_Risk_Analysis/02_Journal/2023_heat_risk_all_variables_0112.csv'
trip_df_final.to_csv(final_output_path, index=False)

# 1. Create a lookup dictionary from stableEdgeId to its geometry
#    We re-use heat_df which is already loaded at the top of the script.
print("Creating geometry lookup map for GeoJSON export...")
id_to_geom_map = heat_df.set_index('stableEdgeId')['geometry'].to_dict()

print("Creating trip geometries... (This may take a while)")
trip_df_final['geometry'] = trip_df_final.apply(
    create_trip_geometry, axis=1, args=(id_to_geom_map,)
)

# 3. Convert the DataFrame to a GeoDataFrame
print("Converting to GeoDataFrame...")
# Make sure to set the CRS from your original heat_df
trip_gdf = gpd.GeoDataFrame(
    trip_df_final, 
    geometry='geometry', 
    crs=heat_df.crs 
)

# 4. Filter out any rows that ended up with no geometry
trip_gdf_to_save = trip_gdf[~trip_gdf.geometry.isna() & ~trip_gdf.geometry.is_empty].copy()

# 5. Save the GeoJSON
#    WARNING: This file could be very large!
geojson_output_path = '/media/remap/NO_HEAT_RB/Metro_Boston/Final/Heat_Risk_Analysis/02_Journal/2023_heat_risk_trip_paths_0112.geojson'
print(f"Saving GeoJSON to {geojson_output_path}...")

trip_gdf_to_save.to_file(geojson_output_path, driver='GeoJSON')