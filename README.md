# bike-paths-analysis
A data science project aimed at predicting bike path lengths within H3 hexagons in Amsterdam

`python -m luigi --module pipelines.data_preprocessing BikesDataPreprocessingPipeline --local-scheduler`

To see dependency graph:

`
luigid
`

Luigid starts central scheduler on 8082

`
python -m luigi --module main BikesDataPreprocessingPipeline
`

MLFlow:

`
mlflow server --host 127.0.0.1 --port 8080
`
