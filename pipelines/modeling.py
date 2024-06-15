import os

import geopandas as gpd
import luigi

from pipelines.common import CreateDirectory
from pipelines.data_preprocessing import FeatureEngineering
from src.constants import (
    DEFAULT_HEX_RESOLUTION,
    LINEAR_REGRESSION_FEATURES_NUM,
    MODEL_OUTPUT_DIR,
    TARGET_FEATURE,
)
from src.model_building import LinearRegressionModelBuilder
from src.transformations import get_predictors


class TrainAndTestLinearRegressionModel(luigi.Task):
    train_city = luigi.Parameter()
    test_city = luigi.Parameter()
    force_positive = luigi.BoolParameter()

    def requires(self):
        return {
            "output_dir": CreateDirectory(MODEL_OUTPUT_DIR),
            self.train_city: FeatureEngineering(
                self.train_city, DEFAULT_HEX_RESOLUTION
            ),
            self.test_city: FeatureEngineering(self.test_city, DEFAULT_HEX_RESOLUTION),
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.input()["output_dir"].path,
                "linear_regression",
                f"train_{self.train_city}_test_{self.test_city}_h{DEFAULT_HEX_RESOLUTION}"
                f"{'_only_pos' if self.force_positive else ''}",
            )
        )

    def run(self):
        train_city_data = gpd.read_parquet(self.input()[self.train_city].path)
        test_city_data = gpd.read_parquet(self.input()[self.test_city].path)

        lr = LinearRegressionModelBuilder(
            predictors=get_predictors(train_city_data, TARGET_FEATURE),
            target_feature=TARGET_FEATURE,
            save_dir=self.output().path,
        )

        lr.set_data(train_city_data, self.train_city)
        lr.split_data()
        lr.forward_feature_selection(num_features=LINEAR_REGRESSION_FEATURES_NUM)
        lr.build_model()
        lr.predict()
        if self.force_positive:
            lr.force_positive_predictions()
        lr.calculate_metrics_values()
        lr.plot_actuals_vs_predicted()

        lr.set_data(test_city_data, self.test_city)
        lr.split_data(test_size=1)
        lr.predict()
        if self.force_positive:
            lr.force_positive_predictions()
        lr.calculate_metrics_values()
        lr.plot_actuals_vs_predicted()

        lr.save_model_info()


class BikePathsLengthModelingPipeline(luigi.WrapperTask):
    def requires(self):
        yield TrainAndTestLinearRegressionModel("Amsterdam", "Krakow", False)
