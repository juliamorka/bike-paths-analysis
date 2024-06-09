import geopandas as gpd
import h3
import h3pandas
import osmnx as ox

from src.constants import COLS_TO_DROP
from src.gpd_utils import clip_gdf_and_convert_crs
from src.helpers import get_sum_wrapper


def get_gdf_from_query(query: str) -> gpd.GeoDataFrame:
    """
    Converts a query string into a GeoDataFrame using OSMnx geocoding.

    Parameters:
        query (str): The query string representing the place to geocode.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame representing the queried location.
    """
    return ox.geocode_to_gdf(query)


def get_h3_hexagons_gdf(query: str, resolution: int) -> gpd.GeoDataFrame:
    """
    Generates H3 hexagons for a given query at a specified resolution.

    Parameters:
        query (str): The query string representing the place to geocode.
        resolution (int): The H3 resolution level for generating hexagons.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the H3 hexagons.
    """
    place_gdf = get_gdf_from_query(query)
    h3_polygons = place_gdf.h3.polyfill_resample(resolution)
    return h3_polygons.drop(COLS_TO_DROP, axis=1)


def get_features_from_place(place_query, tags_dict) -> gpd.GeoDataFrame:
    """
    Retrieves features from a specified place based on OpenStreetMap tags.

    Parameters:
        place_query (str): The query string representing the place to search for features.
        tags_dict (dict): A dictionary of OpenStreetMap tags to filter the features.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the retrieved features.
    """
    return ox.features.features_from_place(place_query, tags_dict)


@get_sum_wrapper
def calculate_feat_area_within_polygon(polygon, features):
    """
    Calculates the total area of features within a polygon.

    Parameters:
        polygon (gpd.GeoDataFrame): The polygon GeoDataFrame used as the clipping boundary.
        features (gpd.GeoDataFrame): The features GeoDataFrame to be clipped.

    Returns:
        float: The sum of the areas of the features within the polygon.
    """
    return clip_gdf_and_convert_crs(features, polygon).area


@get_sum_wrapper
def calculate_feat_len_within_polygon(polygon, features):
    """
    Calculates the total length of features within a polygon.

    Parameters:
        polygon (gpd.GeoDataFrame): The polygon GeoDataFrame used as the clipping boundary.
        features (gpd.GeoDataFrame): The features GeoDataFrame to be clipped.

    Returns:
        float: The sum of the lengths of the features within the polygon.
    """
    return clip_gdf_and_convert_crs(features, polygon).length


def calculate_feat_amt_within_polygon(polygon, features):
    """
    Calculates the number of features within a polygon.

    Parameters:
        polygon (gpd.GeoDataFrame): The polygon GeoDataFrame used as the clipping boundary.
        features (gpd.GeoDataFrame): The features GeoDataFrame to be clipped.

    Returns:
        int: The number of features within the polygon.
    """
    return clip_gdf_and_convert_crs(features, polygon).shape[0]


def apply_calculations(
    polygons,
    kind,
    query,
    vars_dict=None,
    features=None,
    features_colnames=None,
):
    """
    Apply calculations to the polygons DataFrame based on the provided parameters.
    Use vars_dict parameter to retrieve features from OpenStreetMap API and features,
    and features_colnames to use self-defined feature (e.g. from other source)
    with a custom name.

    Parameters:
        polygons (DataFrame): DataFrame containing polygon geometries.
        kind (str): Type of calculation to perform. Options: 'amt', 'area', 'len'.
        query (str): Query string.
        vars_dict (dict, optional): Dictionary mapping with column names as keys
                                    and their definitions, using tags from OpenStreetMap,
                                    as values.
        features (list, optional): List of features in form of pd.Series.
        features_colnames (list, optional): List of column names for passed features.

    Raises:
        ValueError: If neither 'vars_dict' nor 'features' is provided.
    """
    if vars_dict is None and features is None:
        raise ValueError("Either 'vars_dict' or 'features' must be provided.")

    kind_func_mapping = {
        "amt": calculate_feat_amt_within_polygon,
        "area": calculate_feat_area_within_polygon,
        "len": calculate_feat_len_within_polygon,
    }

    func_to_use = kind_func_mapping[kind]
    if vars_dict:
        for tag_key, tags_dict in vars_dict.items():
            features = get_features_from_place(query, tags_dict)
            polygons[f"{tag_key}_{kind}"] = polygons["geometry"].apply(
                func_to_use, args=(features,)
            )
    else:
        for feature, feature_name in zip(features, features_colnames):
            polygons[feature_name] = polygons["geometry"].apply(
                func_to_use, args=(feature,)
            )
