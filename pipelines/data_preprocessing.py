import os

import geopandas as gpd
import luigi

from src.constants import (
    CONFIG_FILE_PATHS,
    DEFAULT_HEX_RESOLUTION,
    INPUT_DATA_DIR,
    INTERIM_DATA_DIR,
    OSMNX_QUERIES_DICT,
    OUTPUT_DATA_DIR,
    TARGET_FEATURE,
)
from src.helpers import get_json_configs
from src.osmnx_utils import apply_calculations, get_h3_hexagons_gdf
from src.transformations import get_predictors, get_transformed_predictors


class CreateDirectory(luigi.ExternalTask):
    directory = luigi.PathParameter()

    def output(self):
        return luigi.LocalTarget(self.directory)


class DownloadInput(luigi.ExternalTask):
    filepath = luigi.PathParameter()

    def requires(self):
        return CreateDirectory(INPUT_DATA_DIR)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.input().path, self.filepath))


class GetH3Hexagons(luigi.Task):
    place = luigi.Parameter()
    resolution = luigi.IntParameter()

    def requires(self):
        return CreateDirectory(INTERIM_DATA_DIR)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.input().path,
                f"{self.place.lower()}_h3_res_{self.resolution}_polygons.parquet",
            )
        )

    def run(self):
        place_h3_gdf = get_h3_hexagons_gdf(
            OSMNX_QUERIES_DICT[self.place], self.resolution
        )
        place_h3_gdf.to_parquet(self.output().path)


class FeatureEngineering(luigi.Task):
    place = luigi.Parameter()
    resolution = luigi.IntParameter()

    def requires(self):
        return {
            "output_dir": CreateDirectory(OUTPUT_DATA_DIR),
            "h3_polygons": GetH3Hexagons(self.place, self.resolution),
            "bike_paths": DownloadInput(f"{self.place}_bike_paths_extended.parquet"),
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.input()["output_dir"].path,
                f"{self.place.lower()}_h3_res_{self.resolution}_processed.parquet",
            )
        )

    def run(self):
        h3_polygons = gpd.read_parquet(self.input()["h3_polygons"].path)
        bike_paths = gpd.read_parquet(self.input()["bike_paths"].path)

        ox_query = OSMNX_QUERIES_DICT[self.place]

        amt_vars_dict, area_vars_dict, len_vars_dict = get_json_configs(
            CONFIG_FILE_PATHS
        )
        apply_calculations(h3_polygons, "amt", ox_query, vars_dict=amt_vars_dict)
        apply_calculations(h3_polygons, "area", ox_query, vars_dict=area_vars_dict)
        apply_calculations(h3_polygons, "len", ox_query, vars_dict=len_vars_dict)
        apply_calculations(
            h3_polygons,
            "len",
            ox_query,
            features=[bike_paths],
            features_colnames=[TARGET_FEATURE],
        )
        h3_polygons = get_transformed_predictors(
            h3_polygons, get_predictors(h3_polygons, TARGET_FEATURE)
        )

        h3_polygons.to_parquet(self.output().path)


class BikesDataPreprocessingPipeline(luigi.WrapperTask):
    def requires(self):
        yield FeatureEngineering("Amsterdam", DEFAULT_HEX_RESOLUTION)
        yield FeatureEngineering("Krakow", DEFAULT_HEX_RESOLUTION)
