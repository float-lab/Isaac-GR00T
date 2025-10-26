from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    DEFAULT_FEATURES,
    write_json,
)

from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms
import numpy as np

RECORDED_DATASETS = [
    "markmau2/bread1",
    "markmau2/bread1.3",
    "markmau2/petty2.1",
    "markmau2/petty1",
    "markmau2/bread2.1",
    "markmau2/bread2",
]

DATASET_TYPE = {
    "markmau2/bread1": "bread1",
    "markmau2/bread1.3": "bread1",
    "markmau2/petty2.1": "patty1",
    "markmau2/petty1": "patty1",
    "markmau2/bread2.1": "bread2",
    "markmau2/bread2": "bread2",
}

SKIP_EPISODES = {
    "markmau2/bread1": [],
    "markmau2/petty1": [17],
    "markmau2/bread2": [15],
    "markmau2/petty2.1": [14],
}

CAMERA_KEYS = [
    "observation.images.main",
    "observation.images.secondary_0",
]


TASKS = {
    "bread1": "Pick bread and place it on the empty plate",
    "patty1": "Pick patty and place it on the bread on the plate",
    "bread2": "Pick bread and place it on the patty on the plate",
}


TASK_INDEX = {
    "bread1": 0,
    "patty1": 1,
    "bread2": 2,
}

def get_all_recorded_datasets(datasets_to_select: list[str] | None = None):
    datasets = {}
    if datasets_to_select is None:
        datasets_to_select = RECORDED_DATASETS
    else:
        assert set(datasets_to_select).intersection(set(RECORDED_DATASETS)) == set(datasets_to_select)

    print(f"Combining datasets: {datasets_to_select}")
    for repo_id in datasets_to_select:
        datasets[repo_id] = LeRobotDataset(repo_id=repo_id)
    
    return datasets


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


def copy_and_transform(func):
    def wrap(frame, ds_version):
        frame = func(frame, ds_version)
        new_frame = _copy_frame_essentials(frame)
        new_frame = _apply_image_transforms(new_frame)
        return new_frame
    
    return wrap


@copy_and_transform
def clean_bread1_dataset(frame: dict, ds_version: str="bread1"):
    if ds_version == "bread1":
        if frame["episode_index"].item() in [34, 33, 32, 31, 30]:
            frame["task"] = TASKS["patty1"]
            frame["task_index"] = torch.tensor(TASK_INDEX["patty1"])
        else:
            frame["task"] = TASKS["bread1"]
            frame["task_index"] = torch.tensor(TASK_INDEX["bread1"])
    
    elif ds_version == "bread1.3":
        frame["task"] = TASKS["bread1"]
        frame["task_index"] = torch.tensor(TASK_INDEX["bread1"])

    return frame


@copy_and_transform
def clean_bread2_dataset(frame: dict, ds_version="bread2"):
    if ds_version == "bread2":
        frame["task"] = TASKS["bread2"]
        frame["task_index"] = torch.tensor(TASK_INDEX["bread2"])

        if frame["episode_index"].item() > 15:
            frame["observation.state"][1] -= 2 * np.pi
            frame["observation.state"][4] -= 2 * np.pi 
    elif ds_version == "bread2.1":
        frame["task"] = TASKS["bread2"]
        frame["task_index"] = torch.tensor(TASK_INDEX["bread2"])
    return frame


@copy_and_transform
def clean_petty1_dataset(frame: dict, ds_version="patty1"):
    frame["task"] = TASKS["patty1"]
    frame["task_index"] = torch.tensor(TASK_INDEX["patty1"])

    return frame


DATASET_PROCESS_FUNC = {
    "bread1": clean_bread1_dataset,
    "patty1": clean_petty1_dataset,
    "bread2": clean_bread2_dataset,
}

DATASET_SCHEMA = {
    'action': {
        'dtype': 'float32', 
        'shape': (6,),
        'names': ['motor_1', 'motor_2', 'motor_3', 'motor_4', 'motor_5', 'motor_6']
    },
    'observation.state': {
        'dtype': 'float32',
        'shape': (6,),
        'names': ['motor_1', 'motor_2', 'motor_3', 'motor_4', 'motor_5', 'motor_6']
    },
    'timestamp': {
        'dtype': 'float32', 
        'shape': (1,), 
        'names': None
    },
    'episode_index': {
        'dtype': 'int64', 
        'shape': (1,), 
        'names': None
    },
    'frame_index': {
        'dtype': 'int64', 
        'shape': (1,), 
        'names': None
    },
    'task_index': {
        'dtype': 'int64', 
        'shape': (1,), 
        'names': None
    },
    'index': {
        'dtype': 'int64', 
        'shape': (1,), 
        'names': None
    },
    'observation.images.main': {
        'dtype': 'video',
        'shape': (1080, 1920, 3),
        'names': ['height', 'width', 'channel'],
        'info': {
            'video.fps': 60,
            'video.codec': 'avc1',
            'video.pix_fmt': 'yuv420p',
            'video.is_depth_map': False,
            'video.channels': 3,
            'has_audio': False
        }
    },
    'observation.images.secondary_0': {
        'dtype': 'video',
        'shape': (1080, 1920, 3),
        'names': ['height', 'width', 'channel'],
        'info': {
            'video.fps': 60,
            'video.codec': 'avc1',
            'video.pix_fmt': 'yuv420p',
            'video.is_depth_map': False,
            'video.channels': 3,
            'has_audio': False
        }
    },
}


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


def clean_and_combine(
    datasets: dict,
    new_repo_id: str,
    limit: int | None = None,
    do_train_test_split: bool = False,
):
    if limit is not None:
        assert limit > 0, f"Negative limit is not allowed. Got {limit}"
    
    train_dataset = None
    test_dataset = None
    full_dataset = None
    if do_train_test_split:
        train_dataset = LeRobotDataset.create(
            repo_id=f"{new_repo_id}__train",
            fps=60,
            features=DATASET_SCHEMA,
            image_writer_threads=10, 
            robot_type="so-101"
        )
        test_dataset = LeRobotDataset.create(
            repo_id=f"{new_repo_id}__test",
            fps=60,
            features=DATASET_SCHEMA,
            image_writer_threads=10, 
            robot_type="so-101"
        )
    else:
        full_dataset = LeRobotDataset.create(
            repo_id=new_repo_id,
            fps=60,
            features=DATASET_SCHEMA,
            image_writer_threads=10, 
            robot_type="so-101"
        )

    for ds_name, ds in datasets.items():
        print(f"Processing {ds_name}")

        ds_type = DATASET_TYPE[ds_name]
        ds_version = ds_name.split("/")[-1]
        process_func = DATASET_PROCESS_FUNC[ds_type]
        last_episode_idx = None
        test_episode_index = None

        if do_train_test_split:
            test_episode_index = torch.randint(0, ds.num_episodes, (1,)).item()
            print(f"Test Episode Index: {test_episode_index}")

        current_dataset = None
        for i, frame in enumerate(tqdm(ds)):
            if (limit is not None) and (i > limit):
                break
            current_episode_idx = frame["episode_index"].item()
            if current_episode_idx in SKIP_EPISODES.get(ds_name, []):
                continue

            if last_episode_idx is not None and current_episode_idx != last_episode_idx:
                current_dataset.save_episode()
            
            current_dataset = full_dataset
            if do_train_test_split and (current_episode_idx == test_episode_index):
                current_dataset = test_dataset
            elif do_train_test_split:
                current_dataset = train_dataset
            
            assert current_dataset is not None, "Cannot assign a dataset to the episode index"
            last_episode_idx = current_episode_idx
            new_frame = process_func(frame, ds_version)
            current_dataset.add_frame(new_frame, task=frame["task"], timestamp=frame["timestamp"])
        
        assert current_dataset is not None, "Cannot assign a dataset to the episode index"
        current_dataset.save_episode()

    return {
        "full_dataset": full_dataset,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--new-repo-id", "-n", type=str, default="float-lab/test")
    parser.add_argument("--upload", "-u", action="store_true", default=False)
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--train-test-split", "-t", action="store_true")
    parser.add_argument("--datasets", "-d", nargs="+", type=str, default=RECORDED_DATASETS)
    args = parser.parse_args()

    datasets = get_all_recorded_datasets(args.datasets)
    merged_clean_dataset = clean_and_combine(
        datasets, args.new_repo_id, args.limit, args.train_test_split
    )

    if args.upload:
        for ds_split in merged_clean_dataset:
            if merged_clean_dataset[ds_split] is None:
                continue
            write_json(SO101_MODALITY, merged_clean_dataset[ds_split].root / MODALITY_PATH)
            merged_clean_dataset[ds_split].push_to_hub()

if __name__ == "__main__":
    main()