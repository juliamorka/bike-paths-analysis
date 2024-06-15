# bike-paths-analysis
A data science project aimed at predicting bike path lengths within H3 hexagons in Amsterdam.

### General project overview

This project consists of two parts (pipelines) - data preprocessing and modeling. 

The data preprocessing pipeline can be run separately from the modeling pipeline, for e.g. EDA purposes. 
The data will be aggregated and available as .parquet files, but model will not be built.

The modeling pipeline depends on the data preprocessing pipeline. If you run the modeling pipeline, 
it will trigger the data preprocessing pipeline automatically if it has not been executed before 
(i.e., if the .parquet with model's input data does not exist).

The described pipelines rely heavily on the Luigi package and its workflow. For further reference and detailed information, 
please see the [Luigi documentation](https://luigi.readthedocs.io/en/stable/).

### Environment setup

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

### Running data preprocessing pipeline with Luigi's local scheduler (simple mode)

Execute the following command in the terminal window:

```bash
python -m luigi --module pipelines.data_preprocessing BikesDataPreprocessingPipeline --workers {number_of_workers} --local-scheduler
```
To achieve faster computation times, it is recommended to run the pipeline with 2 workers for the current setup, 
allowing for parallel processing of 2 input files. Generally, the number of workers should match the number of input 
files being processed.

### Running data preprocessing pipeline with Luigi's central scheduler

To execute the data preprocessing pipeline using a central scheduler and visualize the dependency graph, follow these steps:

1. **Start the Luigi central scheduler**: Open a terminal window and run the following command:

```bash
luigid
```

2. **Run the Data Preprocessing Pipeline**: In a second terminal window, execute the following command:

```bash
python -m luigi --module pipelines.data_preprocessing --workers 2 BikesDataPreprocessingPipeline
```

3. **Monitor the Dependency Graph**: The central scheduler starts on port `8082` of `localhost`. 
You can monitor the dependency graph by navigating to the following URL in your web browser:

```
http://localhost:8082
```

### Running modeling pipeline (simple mode)

Execute the following command in the terminal window:

```bash
python -m luigi --module pipelines.modeling BikePathsLengthModelingPipeline --local-scheduler
```

### Running Modeling Pipeline with MLFlow

To execute the modeling pipeline using MLFlow and monitor the experiments, follow these steps:

1. **Start the MLFlow Server**: In the first terminal window, run the following command:

```bash
mlflow server --host 127.0.0.1 --port 8080
```
   
2. **Run the Modeling Pipeline**: Open a second terminal window and execute the following command:

```bash
python -m luigi --module pipelines.modeling BikePathsLengthModelingPipeline --local-scheduler
```

3. **Monitor MLFlow Experiments**: The MLFlow server starts on port `8080` of `127.0.0.1`. 
You can monitor your MLFlow experiments by navigating to the following URL in your web browser:

```
http://127.0.0.1:8080
```
