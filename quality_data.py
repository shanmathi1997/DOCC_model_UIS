#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Tool to convert geopandas vector data into rasterized xarray data.
# !python -m pip install geocube


# In[ ]:


import json
from osmxtract import overpass, location
import geopandas as gpd
import math 

import os
import rasterio
from rasterio import features
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
from shapely.geometry import Point
import folium


# In[ ]:


import geemap
import ee

# # If you are running this notebook for the first time, you need to activate the command below for the authentication flow:
# ee.Authenticate()


# In[ ]:


ee.Authenticate()


# In[ ]:


try:
    # Initialize the library.
    ee.Initialize()
    print('Google Earth Engine has initialized successfully!')
except ee.EEException as e:
    print('Google Earth Engine has failed to initialize!')
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise


# In[ ]:


"""Reading the tweets as dataframe"""


# In[ ]:


df1 = pd.read_csv('edited_data.csv')
print(df1) 


# In[ ]:


#counting the repetitive rows
df1['count'] = df1.groupby(['x', 'y'])['x'].transform('count') 


# In[ ]:


df1


# In[ ]:


df1.median()


# In[ ]:


df1_unique = df1.drop_duplicates()


# In[ ]:


len(df1)


# In[ ]:


df1_unique


# In[ ]:


len(df1_unique)


# In[ ]:


"""
#plotting the tweets on base map
#sample data
map = folium.Map(location=[df1_unique['y'][0], df1_unique['x'][0]], zoom_start=4)

# Add markers for each city in the dataframe
for index, row in df1_unique.iterrows():
    folium.Marker(location=[row['y'], row['x']],icon=folium.Icon(color='red', icon='ok-circle')).add_to(map)"""


# In[ ]:


#map


# In[ ]:


# Convert the x,y points to a geopandas GeoDataFrame
geometry = [Point(xy) for xy in zip(df1_unique['x'], df1_unique['y'])]
gdf = gpd.GeoDataFrame(df1_unique, geometry=geometry, crs='EPSG:28992')


# In[ ]:


gdf = gdf.set_crs(epsg=28992)


# In[ ]:


#0.009 degrees * 111 km/degree = 0.999 km
# Create a buffer of 0.009 degrees around the points: 1000 meters
buffered_gdf = gdf.geometry.buffer(0.005)


# In[ ]:


# Plot the buffered points using geopandas
ax = buffered_gdf.plot(alpha=0.5, edgecolor='k')
gdf.plot(ax=ax, color='red', markersize=2)


# In[ ]:


gdf_buf = gdf.iloc[:3]


# In[ ]:


gdf_buf


# In[ ]:


bu_gdf = gdf_buf.geometry.buffer(0.005)


# In[ ]:


# Plot the buffered points using geopandas
ax = bu_gdf.plot(alpha=0.5, edgecolor='k')
gdf_buf.plot(ax=ax, color='red', markersize=2)
# Set the axis labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')


# In[ ]:


#creaing a column with the buffer 
gdf_buffers = gdf.copy()
gdf_buffers['geometry'] = gdf.buffer(0.005)


# In[ ]:


# Perform a spatial join to count the number of points within each buffer
join = gpd.sjoin(gdf_buffers, gdf, op='contains', how='left')
counts = join.groupby(join.index).size()

# Add the buffer counts as a new column in the original GeoDataFrame
gdf['buffer_counts'] = counts


# In[ ]:


gdf['buffer'] = gdf.geometry.buffer(0.005)


# In[ ]:


gdf


# In[ ]:


#visual representation of the first buffer
# Filter the GeoDataFrame to include only the first entry
first_entry = gdf.iloc[[0]]

# Get the polygon buffer geometry for the first entry
buffer_polygon = first_entry['buffer'].values[0]

# Filter the GeoDataFrame to include only the points within the buffer polygon
points_within_buffer = gdf[gdf.geometry.within(buffer_polygon)]
# Create a GeoDataFrame for the buffer polygon
buffer_gdf0 = gpd.GeoDataFrame(geometry=[buffer_polygon])

# Get the x and y extents of the buffer polygon
x_min, y_min, x_max, y_max = buffer_polygon.bounds

# Plot the buffer polygon and points within it
fig, ax = plt.subplots()
buffer_gdf0.plot(ax=ax, color='blue', alpha=0.5)
points_within_buffer.plot(ax=ax, color='red', markersize=5)

# Set the plot limits to match the extent of the buffer polygon
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Set the axis labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Show the plot
plt.show()


# In[ ]:


gdf['count'].describe()


# In[ ]:


gdf['count'].value_counts()


# In[ ]:


gdf['buffer_counts'].value_counts()


# In[ ]:


gdf['buffer_counts'].describe()


# In[ ]:


threshold = gdf['buffer_counts'].quantile(0.25)
threshold1 = gdf['count'].quantile(0.25)


# In[ ]:


print(threshold)
print(threshold1)


# In[ ]:


con1 = gdf['count'] > threshold1
con2 = gdf['buffer_counts'] > threshold
gdf_filtered = gdf[con1 & con2]


# In[ ]:


gdf_filtered


# In[ ]:


gdf_filtered = gdf_filtered.reset_index()


# In[ ]:


gdf_filtered 


# In[ ]:


# create a new column in the GeoDataFrame to store the polygon index
gdf_filtered['polygon_index'] = None

# assign a unique index to each point based on the buffer polygon it falls within
for i, poly in enumerate(gdf_filtered['buffer']):
    mask = gdf_filtered['geometry'].within(poly)
    gdf_filtered.loc[mask, 'polygon_index'] = i


# In[ ]:


gdf_filtered['polygon_index'].value_counts()


# In[ ]:


gdf_filtered['polygon_index'].nunique()


# In[ ]:


gdf_filtered


# In[ ]:


chcek0 = gdf_filtered.loc[gdf_filtered['polygon_index'] == 42538]


# In[ ]:


chcek0


# In[ ]:


map = folium.Map(location=[chcek0['y'][3], chcek0['x'][3]], zoom_start=4)

# Add markers for each city in the dataframe
for index, row in chcek0.iterrows():
       folium.Marker(location=[row['y'], row['x']],icon=folium.Icon(color='red', icon='ok-circle')).add_to(map)


# In[ ]:


map


# In[ ]:


# select only one point from each polygon_index based on the maximum count value
max_counts = gdf_filtered.groupby('polygon_index')['count'].idxmax()
selected_points = gdf_filtered.loc[max_counts]


# In[ ]:


selected_points


# In[ ]:


selected_points['polygon_index'].value_counts()


# In[ ]:


selected_points['polygon_index'].nunique()


# In[ ]:


filepath = "shapefiles_tweet"
selected_points.to_csv(f"{filepath}/selection_con1.csv")


# In[ ]:


chcek = selected_points.loc[selected_points['polygon_index'] == 42538]


# In[ ]:


chcek


# In[ ]:


map = folium.Map(location=[chcek['y'][7216], chcek['x'][7216]], zoom_start=4)

# # Add markers for each city in the dataframe
for index, row in chcek.iterrows():
       folium.Marker(location=[row['y'], row['x']],icon=folium.Icon(color='red', icon='ok-circle')).add_to(map)

map


# In[ ]:


# map = folium.Map(location=[selected_points['y'][265], selected_points['x'][265]], zoom_start=4)

# # Add markers for each city in the dataframe
# for index, row in selected_points.iterrows():
#     folium.Marker(location=[row['y'], row['x']],icon=folium.Icon(color='red', icon='ok-circle')).add_to(map)


# In[ ]:


# map


# In[ ]:


selection1 = selected_points.copy()


# In[ ]:


selection1 = selection1.reset_index()


# In[ ]:


selection1 = selection1.drop(columns = 'level_0')
selection1 = selection1.drop(columns = 'index')


# In[ ]:


selection1 = selection1.set_crs(epsg=28992)


# In[ ]:


selection1


# In[ ]:


# create a buffer around the points
selection_buffers = selection1.copy()
buffer_size = 0.09 # buffer size in degrees # 0.09 = 10,000 metres
selection1['buffer_2'] = selection1.buffer(buffer_size)
buffered_points = selection1.buffer(buffer_size)


# In[ ]:


# plot the buffer around the points
ax = selection1.plot(cmap='Set2', alpha=0.5)
buffered_points.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1)
# Set the axis labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')


# In[ ]:


count = []
for i, buff in enumerate(buffered_points):
    # create a mask to select points within the buffer
    mask = selection1.within(buff)
    # count the number of points within the buffer
    num_points = mask.sum()
    count.append(num_points)


# In[ ]:


selection1['count_buffer2'] = pd.Series(count)


# In[ ]:


"""
# Perform a spatial join to count the number of points within each buffer
join = gpd.sjoin(selection1,selection_buffers, op='contains', how='left')
counts = join.groupby(join.index).size()

# Add the buffer counts as a new column in the original GeoDataFrame
selection1['buffer_counts_1'] = counts"""


# In[ ]:


selection1 


# In[ ]:


selection1['count_buffer2'].describe()


# In[ ]:


selection1['count_buffer2'].value_counts()


# In[ ]:


selection1


# In[ ]:


# assign a unique index to each point based on the buffer polygon it falls within
for i, poly in enumerate(selection1['buffer_2']):
    mask = selection1['geometry'].within(poly)
    selection1.loc[mask, 'polygon_index_1'] = i


# In[ ]:


selection1


# In[ ]:


# calculate the threshold for buffer count
threshold_buffer_count = selection1['count_buffer2'].quantile(0.75)
print(threshold_buffer_count)
# select points where the buffer count is greater than the threshold
chosen_points = selection1.loc[selection1['count_buffer2'] >= threshold_buffer_count]


# In[ ]:


chosen_points


# In[ ]:


# select only one point from each polygon_index based on the maximum count value
max_counts = chosen_points.groupby('polygon_index_1')['count'].idxmax()
max_points = chosen_points.loc[max_counts]


# In[ ]:


chosen_points =  chosen_points.drop(max_points.index)


# In[ ]:


chosen_points


# In[ ]:


selection1 =  selection1.drop(chosen_points.index)


# In[ ]:


selection1


# In[ ]:


selection1 = selection1.reset_index()


# In[ ]:


filepath = "shapefiles_tweet"
selection1.to_csv(f"{filepath}/selection1.csv")


# In[ ]:


map = folium.Map(location=[selection1['y'][0], selection1['x'][0]], zoom_start=4)

# # Add markers for each city in the dataframe
for index, row in selection1.iterrows():
     folium.Marker(location=[row['y'], row['x']],icon=folium.Icon(color='red', icon='ok-circle')).add_to(map)


# In[ ]:


map


# In[ ]:


selected = selection1.iloc[:3]


# In[ ]:


selected


# ## Satellite Imagery

# In[ ]:


def get_bounding_box(latitude_in_degrees, longitude_in_degrees, half_side_in_km):
    assert half_side_in_km > 0
    assert latitude_in_degrees >= -90.0 and latitude_in_degrees  <= 90.0
    assert longitude_in_degrees >= -180.0 and longitude_in_degrees <= 180.0

    #lat = math.radians(latitude_in_degrees)
    #lon = math.radians(longitude_in_degrees)
    lat = latitude_in_degrees*(math.pi / 180)
    lon = longitude_in_degrees*(math.pi / 180)

    radius  = 6371
    # Radius of the parallel at given latitude
    #parallel_radius = radius*math.cos(lat)

    lat_min = lat - half_side_in_km/radius
    lat_max = lat + half_side_in_km/radius
    lon_min = lon - half_side_in_km/radius
    lon_max = lon + half_side_in_km/radius
    #lon_min = lon - half_side_in_km/parallel_radius
    #lon_max = lon + half_side_in_km/parallel_radius
    rad2deg = math.degrees

    box = (rad2deg(lat_min),rad2deg(lon_min),rad2deg(lat_max),rad2deg(lon_max))
    return (box)
    # return lat_min,lon_min,lat_max,lon_max


# In[ ]:


bounds =[]
for index, row in selection1.iterrows():
    # extract x and y values
    target_lon = row['x']
    target_lat = row['y']
    half_side_in_km = 0.3145

    #bounds = location.from_buffer(target_lat, target_lon, buffer_size=1000*half_side_in_km)
    bounds1 = get_bounding_box(target_lat,target_lon,half_side_in_km)
    bounds.append(bounds1)
    


# In[ ]:


bnds = bounds[1]
slope = (bnds[3]-bnds[1])/(bnds[2]-bnds[0])


# In[ ]:


slope


# In[ ]:


'''for i, value in enumerate(bounds):
    min_long
    query = overpass.ql_query(value, tag='building')
    response = overpass.request(query)
    feature_collection = overpass.as_geojson(response, 'polygon')
    gdf1 = gpd.GeoDataFrame.from_features(feature_collection)
    gdf_within_bound = gdf1.cx[min_lon:max_lon, min_lat:max_lat]
    num_vectors_within_bound = len(gdf_within_bound)'''


# In[ ]:


buildings_count = []
roads_count = []
for i, value in enumerate(bounds):
    query = overpass.ql_query(value, tag='building')
    response = overpass.request(query)
    feature_collection = overpass.as_geojson(response, 'polygon')
    gdf1 = gpd.GeoDataFrame.from_features(feature_collection)
    gdf_count = gdf1.cx[value[1]:value[3], value[0]:value[2]]
    #print(i)
    # Count the number of buildings
    num_buildings = len(gdf_count)
    #print(f"Number of buildings within the bounding box{i} : {num_buildings}")
    buildings_count.append(num_buildings)
    #gdf1 = gdf1.set_crs('EPSG:28992')
    #buildings.append(gdf1)
    #print(gdf1.columns)
    #filename_building = gdf1.to_file(f'output_building/output_building_{i}.shp') 
    #gdf1.plot()

    query2 = overpass.ql_query(value, tag='highway')
    response2 = overpass.request(query2)
    feature_collection2 = overpass.as_geojson(response2, 'linestring')
    gdf2 = gpd.GeoDataFrame.from_features(feature_collection2)
    gdf_count2 = gdf2.cx[value[1]:value[3], value[0]:value[2]]
    # Count the number of buildings
    num_roads = len(gdf_count2)
    #print(f"Number of roads within the bounding box{i} : {num_roads}")
    roads_count.append(num_roads)
    #gdf2 = gdf2.set_crs('EPSG:28992')
    #filename_road = gdf2.to_file(f'output_road/output_road_{i}.shp')
    #roads.append(gdf2)
    #gdf2.plot()


# In[ ]:


# Create a new column from the list with the same index as gdf
building_counts = pd.Series(buildings_count, index=selection1.index)

# Concatenate the new column with gdf
selction_new = pd.concat([selection1, building_counts.rename('building_counts')], axis=1)


# In[ ]:


selction_new


# In[ ]:


# Create a new column from the list with the same index as gdf
road_counts = pd.Series(roads_count, index=selction_new.index)
# Concatenate the new column with gdf
selction_new = pd.concat([selction_new, road_counts.rename('road_counts')], axis=1)


# In[ ]:


selction_new


# In[ ]:


selction_new['building_counts'].describe()


# In[ ]:


selction_new['road_counts'].describe()


# In[ ]:


threshold_building = selction_new['building_counts'].quantile(0.25)
threshold_road = selction_new['road_counts'].quantile(0.25)


# In[ ]:


con3 = selction_new['building_counts'] > threshold_building
con4 = selction_new['road_counts'] > threshold_road
selction_new_filtered = selction_new[con3 & con4]


# In[ ]:


selction_new_filtered


# In[ ]:


selction_new_filtered = selction_new_filtered.reset_index()


# In[ ]:


filepath = "shapefiles_tweet"
selction_new_filtered.to_csv(f"{filepath}/final.csv")


# In[ ]:


selction_new_rough = selction_new_filtered[:2]


# In[ ]:


selction_new_rough


# In[ ]:


bounds =[]
for index, row in  selction_new_filtered.iterrows():
    # extract x and y values
    target_lon = row['x']
    target_lat = row['y']
    half_side_in_km = 0.3145

    #bounds = location.from_buffer(target_lat, target_lon, buffer_size=1000*half_side_in_km)
    bounds1 = get_bounding_box(target_lat,target_lon,half_side_in_km)
    bounds.append(bounds1)


# In[ ]:


#trial
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import rasterio
import rasterio.mask
import geopandas as gpd
from rasterio.enums import Resampling


# In[ ]:



def rasterize(raster, vector):
    vector = vector
    geom = [shapes for shapes in vector.geometry]
    raster = rasterio.open(raster, 'r')

    # Rasterize vector using the shape and coordinate system of the raster
    img_gdf1 = features.rasterize(geom,
                                    out_shape = (64,64),
                                    fill = 0,
                                    out = None,
                                    transform = raster.transform,
                                    all_touched = True,
                                    #default_value = 1,
                                    dtype = None)
    # Define the metadata for the output tif file
    meta = raster.meta.copy()
    meta['width'] = 64
    meta['height'] = 64
    #meta.update({'count':raster.count})
    meta.update({'crs':raster.crs})
    meta.update({'dtype': rasterio.uint8})
    with rasterio.open(f'output_raster/Building/output_raster_building_{i+1}.tif', 'w', **meta) as out:
        out.write(img_gdf1, 1)
    plt.imshow(img_gdf1)


# In[ ]:


def rasterize1(raster, vector):
    vector = vector
    geom = [shapes for shapes in vector.geometry]
    raster = rasterio.open(raster, 'r')

    # Rasterize vector using the shape and coordinate system of the raster
    img_gdf1 = features.rasterize(geom,
                                    out_shape = (64,64),
                                    fill = 0,
                                    out = None,
                                    transform = raster.transform,
                                    all_touched = False,
                                    #default_value = 1,
                                    dtype = None)
    # Define the metadata for the output tif file
    meta = raster.meta.copy()
    meta['width'] = 64
    meta['height'] = 64
    meta.update({'crs':raster.crs})
    meta.update({'dtype': rasterio.uint8})
    with rasterio.open(f'output_raster/Road/output_raster_road_{i}.tif', 'w', **meta) as out:
        out.write(img_gdf1, 1)
    plt.imshow(img_gdf1)


# In[ ]:


from rasterio.transform import from_origin
# bands to be selected

band_names = ['B2', 'B3', 'B4', 'B8']
for i, value in enumerate(bounds):
    region = ee.Geometry.Rectangle(
    [[value[1], value[0]],
     [value[3], value[2]]]
    )

    myCollection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')         .filterBounds(region)         .filterDate('2022-06-01', '2022-09-30')         .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 10)         .select(['B2', 'B3', 'B4', 'B8'])
    listOfImages = myCollection.aggregate_array('system:index').getInfo()
    print('Number of images in the collection: ', len(listOfImages))
    img = myCollection.median().clip(region)
    #img = myCollection.median().clip(region).reproject(crs = 'EPSG:28992', scale=10)
    #img_resampled = img.resample('bicubic').reproject(crs='EPSG:28992', scale=10)
    filename = f'output_satellite/output_raster_{i}.tif'

    geemap.ee_export_image(
        img, filename=filename, scale=10, region=region, crs='EPSG:4326', file_per_band=False)

    query = overpass.ql_query(value, tag='building')
    response = overpass.request(query)
    feature_collection = overpass.as_geojson(response, 'polygon')
    gdf1 = gpd.GeoDataFrame.from_features(feature_collection)
    gdf_count = gdf1.cx[value[1]:value[3], value[0]:value[2]]
    # Count the number of buildings
    num_buildings = len(gdf_count)
    print(f"Number of buildings within the bounding box{i} : {num_buildings}")
    #gdf1 = gdf1.set_crs('EPSG:28992')
    #buildings.append(gdf1)
    #print(gdf1.columns)
    print(i)
    #filename_building = gdf1.to_file(f'output_building/output_building_{i}.shp') 
    gdf1.plot()

    query2 = overpass.ql_query(value, tag='highway')
    response2 = overpass.request(query2)
    feature_collection2 = overpass.as_geojson(response2, 'linestring')
    gdf2 = gpd.GeoDataFrame.from_features(feature_collection2)
    gdf_count2 = gdf2.cx[value[1]:value[3], value[0]:value[2]]
    # Count the number of buildings
    num_roads = len(gdf_count2)
    print(f"Number of roads within the bounding box{i} : {num_roads}")
    #gdf2 = gdf2.set_crs('EPSG:28992')
    #filename_road = gdf2.to_file(f'output_road/output_road_{i}.shp')
    #roads.append(gdf2)
    gdf2.plot()
    #vect = f'output_building/output_building_{i}.shp'
    #building_raster = rasterize(filename, gdf1)
    #road_raster = rasterize1(filename, gdf2)


# In[ ]:


import rasterio
from rasterio.plot import show


# In[ ]:


fp = r'output_satellite/output_raster_1.tif'
with rasterio.open(fp) as src:
    # Read the first band of the raster
    data = src.read(3)
    num_band =src.count

                   
# Display the raster
show(data)
print(data.shape)
print(num_band)


# In[ ]:


import rasterio
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import box


# In[ ]:


fp = r'EuroSAT_RGB/Forest/Forest_1.jpg'
with rasterio.open(fp) as src:
    # Read the first band of the raster
    data = src.read(1)
    num_band =src.count

                   
# Display the raster
show(data)
print(data.shape)
print(num_band)


# In[ ]:


from osgeo import gdal

# Open the GeoTIFF file
dataset = gdal.Open("EuroSAT_RGB/Forest/Forest_1.jpg")

# Get the spatial reference information
projection = dataset.GetProjection()
transform = dataset.GetGeoTransform()



# In[ ]:


# Get the image dimensions
cols = dataset.RasterXSize
rows = dataset.RasterYSize

# Calculate the pixel size in the x and y directions
pixel_width = transform[1]
pixel_height = transform[5]


# In[ ]:



# Calculate the latitude and longitude of the four corners of the image
min_x = transform[0]
max_y = transform[3]
max_x = min_x + pixel_width * cols
min_y = max_y + pixel_height * rows


# In[ ]:



# Print the latitude and longitude information
print("Latitude: {:.4f} to {:.4f}".format(min_y, max_y))
print("Longitude: {:.4f} to {:.4f}".format(min_x, max_x))


# In[ ]:




