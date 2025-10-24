from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms
import numpy as np

RECORDED_DATASETS = [
    "markmau2/bread1",
    "markmau2/petty1",
    "markmau2/bread2",
]

SKIP_EPISODES = {
    "markmau2/bread1": [],
    "markmau2/petty1": [17],
    "markmau2/bread2": [15],
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

def get_all_recorded_datasets():
    datasets = {}
    for repo_id in RECORDED_DATASETS:
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
    def wrap(frame):
        frame = func(frame)
        new_frame = _copy_frame_essentials(frame)
        new_frame = _apply_image_transforms(new_frame)
        return new_frame
    
    return wrap


@copy_and_transform
def clean_bread1_dataset(frame: dict):
    if frame["episode_index"].item() in [34, 33, 32, 31, 30]:
        frame["task"] = TASKS["patty1"]
        frame["task_index"] = torch.tensor(TASK_INDEX["patty1"])
    else:
        frame["task"] = TASKS["bread1"]
        frame["task_index"] = torch.tensor(TASK_INDEX["bread1"])
    
    return frame


@copy_and_transform
def clean_bread2_dataset(frame: dict):
    frame["task"] = TASKS["bread2"]
    frame["task_index"] = torch.tensor(TASK_INDEX["bread2"])

    if frame["episode_index"].item() > 15:
        frame["observation.state"][1] -= 2 * np.pi
        frame["observation.state"][4] -= 2 * np.pi 
    return frame


@copy_and_transform
def clean_petty1_dataset(frame: dict):
    frame["task"] = TASKS["patty1"]
    frame["task_index"] = torch.tensor(TASK_INDEX["patty1"])

    return frame


DATASET_PROCESS_FUNC = {
    "markmau2/bread1": clean_bread1_dataset,
    "markmau2/petty1": clean_petty1_dataset,
    "markmau2/bread2": clean_bread2_dataset,
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
            'has_audio': False
        }
    },
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
            repo_id=f"{new_repo_id}_train",
            fps=60,
            features=DATASET_SCHEMA,
            image_writer_threads=10, 
            robot_type="so-101"
        )
        test_dataset = LeRobotDataset.create(
            repo_id=f"{new_repo_id}_test",
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

        process_func = DATASET_PROCESS_FUNC[ds_name]
        last_episode_idx = None
        test_episode_index = None

        if do_train_test_split:
            test_episode_index = torch.randint(0, ds.num_episodes, (1,))[0]
        
        current_dataset = None
        for i, frame in enumerate(tqdm(ds)):
            if (limit is not None) and (i > limit):
                break
            current_episode_idx = frame["episode_index"].item()
            if current_episode_idx in SKIP_EPISODES[ds_name]:
                continue

            current_dataset = full_dataset
            if do_train_test_split and (current_episode_idx == test_episode_index):
                current_dataset = test_dataset
            elif do_train_test_split:
                current_dataset = train_dataset
            
            assert current_dataset is not None, "Cannot assign a dataset to the episode index"
            if last_episode_idx is not None and current_episode_idx != last_episode_idx:
                current_dataset.save_episode()
            
            last_episode_idx = current_episode_idx
            new_frame = process_func(frame)
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
    parser.add_argument("--train-test-split", action="store_true")
    args = parser.parse_args()

    datasets = get_all_recorded_datasets()
    merged_clean_dataset = clean_and_combine(
        datasets, args.new_repo_id, args.limit, args.train_test_split
    )

    if args.upload:
        for ds_split in merged_clean_dataset:
            if merged_clean_dataset[ds_split] is None:
                continue
            merged_clean_dataset[ds_split].push_to_hub()

if __name__ == "__main__":
    main()