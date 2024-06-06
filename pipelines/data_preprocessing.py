import luigi
from src.constants import (
    OSMNX_QUERIES_DICT,
    INPUT_DATA_DIR,
    INTERIM_DATA_DIR,
    OUTPUT_DATA_DIR,
)
from src.osmnx_utils import (
    get_h3_hexagons_gdf,
    calculate_tag_area,
    calculate_tag_len,
    calculate_tag_amt,
    get_bike_paths_len,
)
import os
import geopandas as gpd
from joblib import Parallel, delayed
from src.helpers import get_json_configs
from src.constants import CONFIG_FILE_PATHS


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

    def requires(self):
        return CreateDirectory(INTERIM_DATA_DIR)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(self.input().path, f"{self.place.lower()}_h3_polygons.parquet")
        )

    def run(self):
        place_h3_gdf = get_h3_hexagons_gdf(OSMNX_QUERIES_DICT[self.place])
        place_h3_gdf.to_parquet(self.output().path)


class FeatureEngineering(luigi.Task):
    place = luigi.Parameter()

    def requires(self):
        return {
            "output_dir": CreateDirectory(OUTPUT_DATA_DIR),
            "h3_polygons": GetH3Hexagons(self.place),
            "bike_paths": DownloadInput(f"{self.place}_bike_paths_extended.parquet"),
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.input()["output_dir"].path,
                f"{self.place.lower()}_h3_processed.parquet",
            )
        )

    def run(self):
        h3_polygons = gpd.read_parquet(self.input()["h3_polygons"].path)
        bike_paths = gpd.read_parquet(self.input()["bike_paths"].path)

        amt_vars_dict, area_vars_dict, len_vars_dict = get_json_configs(
            CONFIG_FILE_PATHS
        )

        with Parallel(n_jobs=-1, verbose=True) as parallel:
            for tag_key, tags_dict in amt_vars_dict.items():
                h3_polygons[f"{tag_key}_amt"] = parallel(
                    delayed(calculate_tag_amt)(polygon, tags_dict)
                    for polygon in h3_polygons["geometry"]
                )

            for tag_key, tags_dict in area_vars_dict.items():
                h3_polygons[f"{tag_key}_area"] = parallel(
                    delayed(calculate_tag_area)(polygon, tags_dict)
                    for polygon in h3_polygons["geometry"]
                )

            for tag_key, tags_dict in len_vars_dict.items():
                h3_polygons[f"{tag_key}_len"] = parallel(
                    delayed(calculate_tag_len)(polygon, tags_dict)
                    for polygon in h3_polygons["geometry"]
                )

            h3_polygons["bike_paths_len"] = parallel(
                delayed(get_bike_paths_len)(polygon, bike_paths)
                for polygon in h3_polygons["geometry"]
            )
        h3_polygons.to_parquet(self.output().path)


class BikesDataPreprocessingPipeline(luigi.WrapperTask):
    def requires(self):
        yield FeatureEngineering("Amsterdam")
        yield FeatureEngineering("Krakow")


# if __name__ == "__main__":
#     luigi.run()

print(__name__)
