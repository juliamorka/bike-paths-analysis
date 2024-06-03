import geopandas as gpd
import h3
import h3pandas
import osmnx as ox
from src.constants import HEX_RESOLUTION, COLS_TO_DROP, DEFAULT_CRS
from src.gpd_utils import *


def get_gdf_from_query(query: str) -> gpd.GeoDataFrame:
    return ox.geocode_to_gdf(query)


def get_h3_hexagons_gdf(query: str, resolution: int = HEX_RESOLUTION):
    place_gdf = get_gdf_from_query(query)
    h3_polygons = place_gdf.h3.polyfill_resample(resolution)
    return h3_polygons.drop(COLS_TO_DROP, axis=1)


def get_features_from_polygon(geometry, tags_dict):
    return ox.features.features_from_polygon(geometry, tags=tags_dict)


@get_sum_wrapper
def calculate_tag_area(geometry, tags_dict):
    try:
        features = clip_gdf_and_convert_crs(
            get_features_from_polygon(geometry, tags_dict)["geometry"]
        )
    except Exception as e:
        print(e)  # TODO: logger
        return 0
    return features.area


@get_sum_wrapper
def calculate_tag_len(geometry, tags_dict):
    try:
        features = clip_gdf_and_convert_crs(
            get_features_from_polygon(geometry, tags_dict)["geometry"]
        )
    except Exception as e:
        print(e)  # TODO: logger
        return 0
    return features.length


def calculate_tag_amt(geometry, tags_dict):
    try:
        features = get_features_from_polygon(geometry, tags_dict)
    except Exception as e:
        print(e)  # TODO: logger
        return 0
    return features.shape[0]


@get_sum_wrapper
def get_bike_paths_len(geometry, bike_paths):
    clipped_bike_paths = clip_gdf_and_convert_crs(bike_paths, geometry)
    return clipped_bike_paths.length
