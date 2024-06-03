import luigi
from src.constants import (
    OSMNX_QUERIES_DICT,
    INPUT_DATA_DIR,
    INTERIM_DATA_DIR,
)
from src.osmnx_utils import get_h3_hexagons_gdf
import os


class CreateDirectory(luigi.ExternalTask):
    directory = luigi.PathParameter()

    def output(self):
        return luigi.LocalTarget(self.directory)


class DownloadInput(luigi.ExternalTask):
    filepath = luigi.PathParameter()

    def requires(self):
        return CreateDirectory(INPUT_DATA_DIR)

    def output(self):
        return luigi.LocalTarget(self.filepath)


class GetH3Hexagons(luigi.Task):
    place = luigi.Parameter()

    def requires(self):
        return CreateDirectory(INTERIM_DATA_DIR)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.input().path, f"{str(self.place).lower()}_h3_polygons.parquet"
            )
        )

    def run(self):
        place_h3_gdf = get_h3_hexagons_gdf(OSMNX_QUERIES_DICT[self.place])
        place_h3_gdf.to_parquet(self.output().path)


class BikesDataPreprocessingPipeline(luigi.WrapperTask):
    def requires(self):
        yield GetH3Hexagons("Amsterdam")


if __name__ == "__main__":
    luigi.run()
