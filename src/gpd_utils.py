import geopandas as gpd
from src.constants import DEFAULT_CRS


def convert_crs(gdf, crs=DEFAULT_CRS):
    return gdf.to_crs(crs)


def clip_gdf(to_clip, mask):
    return gpd.clip(to_clip, mask)


def clip_gdf_and_convert_crs(gdf, mask):
    return convert_crs(clip_gdf(gdf, mask))


def get_sum_wrapper(func):
    def inner(*args, **kwargs):
        pd_like_object_to_sum = func(*args, **kwargs)
        return pd_like_object_to_sum.sum()

    return inner
