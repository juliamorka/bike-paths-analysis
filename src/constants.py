import os

COLS_TO_DROP = [
    "bbox_north",
    "bbox_south",
    "bbox_east",
    "bbox_west",
    "place_id",
    "osm_type",
    "osm_id",
    "lat",
    "lon",
    "class",
    "type",
    "place_rank",
    "importance",
    "addresstype",
    "name",
    "display_name",
    "index",
]
HEX_RESOLUTION = 7
OSMNX_QUERIES_DICT = {"Amsterdam": {"city": "Amsterdam"}, "Krakow": {"city": "Krakow"}}
DATA_DIR = "data"
CONFIG_DIR = "config"
INPUT_DATA_DIR = os.path.join("data", "inputs")
INTERIM_DATA_DIR = os.path.join("data", "interim")
OUTPUT_DATA_DIR = os.path.join("data", "output")
AMSTERDAM_INPUT_FILE_NAME = "amsterdam_bike_paths_extended.parquet"
CRACOW_INPUT_FILE_NAME = "krakow_bike_paths_extended.parquet"
AMSTERDAM_INPUT_FILE_PATH = os.path.join(INPUT_DATA_DIR, AMSTERDAM_INPUT_FILE_NAME)
CRACOW_INPUT_FILE_PATH = os.path.join(INPUT_DATA_DIR, CRACOW_INPUT_FILE_NAME)
CONFIG_FILE_PATHS = [
    os.path.join(CONFIG_DIR, file)
    for file in [
        "amt_columns_binning_config.json",
        "area_columns_binning_config.json",
        "len_columns_binning_config.json",
    ]
]
DEFAULT_CRS = 28992
