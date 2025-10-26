from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    write_json,
    write_jsonlines,
    TASKS_PATH,
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    DEFAULT_FEATURES,
)
from pathlib import Path
import json
from argparse import ArgumentParser
import shutil
from tqdm import tqdm

import torchvision.transforms as transforms
import numpy as np

MODALITY_PATH = "meta/modality.json"

SO101_MODALITY = {
    "state": {
        "single_arm": {
          "start": 0,
          "end": 5
        },
        "gripper": {
          "start": 5,
          "end": 6
        }
    },
    "action": {
        "single_arm": {
          "start": 0,
          "end": 5
        },
        "gripper": {
          "start": 5,
          "end": 6
        }
    },
    "video": {
        "gripper_cam": {
            "original_key": "observation.images.main"
        },
        "front_cam": {
            "original_key": "observation.images.secondary_0"
        }
    },
    "annotation": {
        "human.task_description": {
            "original_key": "task_index"
        }
    }
}

def load_modality(local_dir: Path) -> dict:
    with open(local_dir / MODALITY_PATH) as f:
        return json.load(f)

def convert(lerobot_dataset, groot_dataset_repo_id, modality):
    groot_dataset = LeRobotDataset.create(
        repo_id=groot_dataset_repo_id,
        fps=lerobot_dataset.fps,
        features=lerobot_dataset.features,
        image_writer_threads=10,
        robot_type=lerobot_dataset.meta.robot_type
    )

    write_json(modality, groot_dataset.root / MODALITY_PATH)
    # shutil.copytree(
    #     lerobot_dataset.meta.root / "videos",
    #     groot_dataset.meta.root / "videos",
    # )
    # shutil.copytree(
    #     lerobot_dataset.meta.root / "data",
    #     groot_dataset.meta.root / "data",
    # )
    # shutil.copyfile(lerobot_dataset.meta.root / TASKS_PATH, groot_dataset.meta.root / TASKS_PATH)
    # shutil.copyfile(lerobot_dataset.meta.root / EPISODES_STATS_PATH, groot_dataset.meta.root / EPISODES_STATS_PATH)
    # shutil.copyfile(lerobot_dataset.meta.root / EPISODES_PATH, groot_dataset.meta.root / EPISODES_PATH)
    return groot_dataset

CAMERA_KEYS = [
    "observation.images.main",
    "observation.images.secondary_0",
]

def _copy_frame_essentials(frame):
    new_frame = {}
    for key in frame:
        if key != "task" and key not in DEFAULT_FEATURES:
            new_frame[key] = frame[key]
    
    return new_frame


def _apply_image_transforms(frame, tx = transforms.ToPILImage()):
    for key in CAMERA_KEYS:
        frame[key] = tx(frame[key])
    
    return frame

def process_frame(frame):
    new_frame = _copy_frame_essentials(frame)
    new_frame = _apply_image_transforms(new_frame)
    return new_frame


def main():
    parser = ArgumentParser()
    parser.add_argument("--lerobot-repo-id", "-l", type=str)
    parser.add_argument("--groot-lerobot-repo-id", "-g", type=str)
    parser.add_argument("--upload", "-u", action="store_true", default=False)

    args = parser.parse_args()

    lerobot_dataset = LeRobotDataset(args.lerobot_repo_id)
    groot_dataset = convert(lerobot_dataset, args.groot_lerobot_repo_id, SO101_MODALITY)

    last_episode_idx = None
    for frame in tqdm(lerobot_dataset):
        current_episode_idx = frame["episode_index"].item()
        if last_episode_idx is not None and current_episode_idx != last_episode_idx:
            groot_dataset.save_episode()
        
        last_episode_idx = current_episode_idx
        new_frame = process_frame(frame)
        groot_dataset.add_frame(new_frame, task=frame["task"], timestamp=frame["timestamp"])

    groot_dataset.save_episode()

    if args.upload:
        groot_dataset.push_to_hub()

if __name__ == "__main__":
    main()
