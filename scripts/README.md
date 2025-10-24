# Using scripts for LeSandwich project

## clean_and_combine_datasets.py
- Combines `markmau2/bread1`, `markmau2/bread2` and `markmau2/petty1` datasets
- Cleans the task annotations in each dataset
- Removes bad episodes from the dataset
- Fixes the issue with `motor_2` and `motor_5` values in the last 10 episodes or `markmau2/bread2` dataset

## convert_to_groot_lerobot.py
Groot Lerobot dataset differs from Lerobot dataset [see docs here for details](getting_started/LeRobot_compatible_data_schema.md). This script converts the lerobot dataset to Groot format so that it can be used with groot_fintune scripts.

## pull_dataset.py
Helper script to pull a HF dataset and create a lerobot-friendly dataset with the pulled dataset

## Run finetuning
The experiment configs are store in [experiment_configs.py](scripts/experiment_configs.py). There are 4 main configs defined and each one corresponds to the fps at which actions are sampled from the dataset. These are:
1. lesandwich_60fps
2. lesandwich_30fps
3. lesandwich_20fps
4. lesandwich_10fps

Example command to start finetuning:
```
python scripts/gr00t_finetune.py lesandwich_60fps --no-tune-diffusion-model --batch-size=12 --lora-rank=128 --max-steps 20000 --output-dir /home/ajaya-rao/models/groot/lora-tuning-expts --report-to tensorboard
```
