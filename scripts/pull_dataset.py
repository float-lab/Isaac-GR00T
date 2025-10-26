from lerobot.datasets.lerobot_dataset import LeRobotDataset
from clean_and_combine_datasets import DATASET_SCHEMA
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset", "-d", type=str, default="float-lab/lesandwich2-groot")
parser.add_argument("--fps", "-f", type=int, default=60)
args = parser.parse_args()

ds = LeRobotDataset.create(args.dataset, fps=args.fps, features=DATASET_SCHEMA)
ds.pull_from_repo()
