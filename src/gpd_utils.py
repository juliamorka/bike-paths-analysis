import geopandas as gpd

from src.constants import DEFAULT_CRS


def clip_gdf_and_convert_crs(
    gdf: gpd.GeoDataFrame, mask: gpd.GeoDataFrame, crs: str = DEFAULT_CRS
) -> gpd.GeoDataFrame:
    """
    Clips a GeoDataFrame to the boundary of another GeoDataFrame and converts the CRS.

    Parameters:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to be clipped and converted.
        mask (gpd.GeoDataFrame): The GeoDataFrame providing the boundary for clipping.
        crs (str): The target CRS to convert the GeoDataFrame to. Defaults to "EPSG:3035".

    Returns:
        gpd.GeoDataFrame: The clipped and CRS-converted GeoDataFrame.
    """
    return gpd.clip(gdf, mask).to_crs(crs)
