from lerobot.datasets.lerobot_dataset import LeRobotDataset
from clean_and_combine_datasets import DATASET_SCHEMA

ds = LeRobotDataset.create("float-lab/lesandwich2-groot", fps=60, features=DATASET_SCHEMA)
ds.pull_from_repo()
