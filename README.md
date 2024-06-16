# OSM Bike Paths ML Pipeline project
A data science project aimed at predicting bike path lengths within H3 hexagons in Amsterdam.

### :bicyclist: General project overview

This project consists of two parts (pipelines) - data preprocessing and modeling. 

The data preprocessing pipeline can be run separately from the modeling pipeline, for e.g. EDA purposes. 
The data will be aggregated and available as .parquet files, but model will not be built.

The modeling pipeline depends on the data preprocessing pipeline. If you run the modeling pipeline, 
it will trigger the data preprocessing pipeline automatically if it has not been executed before 
(i.e., if the .parquet with model's input data does not exist).

The described pipelines rely heavily on the Luigi package and its workflow. For further reference and detailed information, 
please see the [Luigi documentation](https://luigi.readthedocs.io/en/stable/).

### :bicyclist: Project structure 
```
.github/workflows
  └── formatting.yml             - GitHub Actions workflow for formatting
config
  ├── amt_columns_binning.json   - Configuration for binning columns related to amount
  ├── area_columns_binning.json  - Configuration for binning columns related to area
  └── len_columns_binning.json   - Configuration for binning columns related to length
data                             - Directory for data 
models                           - Directory for machine learning models and output related to them
pipelines
  ├── common.py                  - Common utilities and functions used across Luigi pipelines
  ├── data_preprocessing.py      - Luigi pipeline for data preprocessing
  └── modeling.py                - Luigi pipeline for model training and evaluation
src
  ├── constants.py               - Constants definition
  ├── gpd_utils.py               - Utility functions related to GeoPandas
  ├── helpers.py                 - Helper functions for general use
  ├── model_building.py          - Classes for building machine learning models
  ├── osmnx_utils.py             - Utility functions related to OSMnx (OpenStreetMap)
  └── transformations.py         - Functions for data transformations

```

### :bicyclist: External project requirements

Before running the project, download the data from [Data Repo](https://github.com/juliamorka/bike-paths-data) 
and store it under the `data/inputs` directory. It is recommended to copy the files manually, as cloning repository
within another one might cause unwanted issues.

### :bicyclist: Environment setup

To create the environment, when running the project for the first time, run below command from the main project directory:
```bash
conda env create -f env.yml
```
To update the environment run, run below command from the main project directory:
```bash
conda env update -f env.yml --prune
```
To activate the environment, run:
```bash
conda activate bpa-env
```
### :bicyclist: Instructions syntax
Detailed running instructions for both pipelines are described below. Their syntax uses square brackets ('[' and ']') when the flag is optional. 

### :bicyclist: Running data preprocessing pipeline with Luigi's local scheduler and default configuration (simple mode) 

Execute the following command in the terminal window:

```bash
python -m luigi --module pipelines.data_preprocessing BikesDataPreprocessingPipeline --workers {number_of_workers} --local-scheduler
```
To achieve faster computation times, it is recommended to run the pipeline with 2 workers for the current setup, 
allowing for parallel processing of 2 input files. Generally, the number of workers should match the number of input 
files being processed.

### :bicyclist: Running data preprocessing pipeline with Luigi's central scheduler and default configuration

To execute the data preprocessing pipeline using a central scheduler and visualize the dependency graph, follow these steps:

1. **Start the Luigi central scheduler**: Open a terminal window and run the following command:

```bash
luigid
```

2. **Run the Data Preprocessing Pipeline**: In a second terminal window, execute the following command:

```bash
python -m luigi --module pipelines.data_preprocessing BikesDataPreprocessingPipeline --workers 2
```

3. **Monitor the Dependency Graph while the pipeline runs**: The central scheduler starts on port `8082` of `localhost`. 
You can monitor the dependency graph by navigating to the following URL in your web browser:

```
http://localhost:8082
```

### :bicyclist: Running data preprocessing pipeline with custom configuration
```bash
python -m luigi --module pipelines.data_preprocessing BikesDataPreprocessingPipeline --hex-resolution {resolution} [--workers {number_of_workers}] [--local-scheduler]
```

Default hexagon resolution is set up to 8, it can be adjusted using --hex-resolution flag as shown in the above command.

### :bicyclist: Running modeling pipeline with default configuration (simple mode)

Execute the following command in the terminal window:

```bash
python -m luigi --module pipelines.modeling BikePathsLengthModelingPipeline [--workers {number_of_workers}] [--local-scheduler]
```
Recommended number of workers is the same as in the data preprocessing pipeline section (2) if data preprocessing
has not been run before, and the input data already exists, the number of workers should be 1. If central scheduler is used,
luigid has to be run in a separate terminal window, as described in the data preprocessing pipeline section.

### :bicyclist: Running Modeling Pipeline with MLFlow

To execute the modeling pipeline using MLFlow and monitor the experiments, follow these steps:

1. **Start the MLFlow Server**: In the first terminal window, run the following command:

```bash
mlflow server --host 127.0.0.1 --port 8080
```
   
2. **Run the Modeling Pipeline**: Open a second terminal window and execute the following command:

```bash
python -m luigi --module pipelines.modeling BikePathsLengthModelingPipeline --log-mlflow [--mlflow-experiment {experiment_name}] [--workers {number_of_workers}] [--local-scheduler]
```
You can choose your own MLFlow experiment name or stick with the default one.

3. **Monitor MLFlow Experiments**: The MLFlow server starts on port `8080` of `127.0.0.1`. 
You can monitor your MLFlow experiments by navigating to the following URL in your web browser:

```
http://127.0.0.1:8080
```

### :bicyclist: Running data preprocessing pipeline with custom configuration
```bash
python -m luigi --module pipelines.data_preprocessing BikePathsLengthModelingPipeline [--workers {number_of_workers}] [--train-city {city_name}] [--test-city {city_name}] [--hex-resolution {resolution}] [--num-features {n}] [--force-positive] [--log-mlflow] [--mlflow-experiment {experiment_name}] [--local-scheduler]
```

The custom configuration allows the user to:
- choose the city the model will be trained on (defaults to "Amsterdam")
- choose the city the already built model will calculate predictions for (defaults to "Krakow")
- choose hexagons resolution (defaults to 8)
- choose number of features chosen in forward feature selection (defaults to 6)
- choose if the negative predictions should be replaced with 0 (defaults to False)
- choose if forward feature selection should be logged with MLFlow 
- specify the name for MLFlow experiment used in the pipeline run

### :bicyclist: Running examples
1. Train model on Amsterdam, apply on Kraków, use 4 features, set hexagons resolution to default number, disallow parallel processing, use Luigi's central scheduler, no MLFlow logging:
```bash
# Terminal 1
luigid
```
```bash
# Terminal 2
python -m luigi --module pipelines.data_preprocessing BikePathsLengthModelingPipeline --num-features 4
```
2. Train model on Amsterdam, apply on Kraków, use default number of features, set hexagons resolution to 7, allow parallel processing, use Luigi's local scheduler, no MLFlow logging:
```bash
# Terminal 1
python -m luigi --module pipelines.data_preprocessing BikePathsLengthModelingPipeline --hex-resolution 7 --local-scheduler
```
3. Train model on Kraków, apply on Amsterdam, use default number of features and default hexagons resolution, disallow parallel processing, use Luigi's central scheduler, turn on MLFlow logging:
```bash
# Terminal 1
luigid
```
```bash
# Terminal 2
mlflow server --host 127.0.0.1 --port 8080
```
```bash
# Terminal 3
python -m luigi --module pipelines.data_preprocessing BikePathsLengthModelingPipeline --train-city Krakow --test-city Amsterdam --log-mlflow
```
### 🚴 Results
After running the data preprocessing pipeline, the features aggregated on H3 hexagonal level will be available under `data/output`.

After running the data modeling pipeline, the `models/linear-regression` will include a folder named according to the pipeline's run configuration. It will consist of:
* correlation heatmap of all predictors for train city
* corrleation heatmaps of predictors chosen in FFS process for both train and test city
* actual vs predicted plots for both cities
* .parquet files with predictions for both cities (this can be joined with features used for modeling and later explored by data scientist)
* .csv file with metrics for both cities (this format was chosen for easy evaluation)
* .pkl with a final model
This is an additional layer of results for the final model, while logs available in MLFlow focus more on FFS (experimental process searching for the best model).
