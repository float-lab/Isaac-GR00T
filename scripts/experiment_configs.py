from dataclasses import dataclass
from typing import List, Literal

from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING

@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories, we assume all datasets have the same data config"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data configuration to use for training.
    Options:
    - Built-in configs: Use predefined config names like 'so100', 'fourier_gr1_arms_only', 'unitree_g1'.
    - External configs: Use 'module:ClassName' format to load custom configs from external files. e.g. 'my_dir.my_configs:RobotConfig'
    See gr00t/experiment/data_config.py for more details.
    """

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 12
    """Number of workers for data loading per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    """Video backend to use for training. [torchcodec, decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""



LESANDWICH_60FPS_CONFIG = ArgsConfig(
    dataset_path = [
        "/home/ajaya-rao/.cache/huggingface/lerobot/float-lab/lesandwich2-groot"
    ],
    output_dir = "/home/ajaya-rao/models/default",
    data_config = "lesandwich_60_fps",
    tune_llm = False,
    tune_visual = False,
    tune_projector = True,
    tune_diffusion_model= False,
    resume = False,
    report_to="wandb",
    embodiment_tag="new_embodiment",
)

LESANDWICH_30FPS_CONFIG = ArgsConfig(
    dataset_path = [
        "/home/ajaya-rao/.cache/huggingface/lerobot/float-lab/lesandwich2-groot"
    ],
    output_dir = "/home/ajaya-rao/models/default",
    data_config = "lesandwich_30_fps",
    tune_llm = False,
    tune_visual = False,
    tune_projector = True,
    tune_diffusion_model= False,
    resume = False,
    report_to="wandb",
    embodiment_tag="new_embodiment",
)

LESANDWICH_20FPS_CONFIG = ArgsConfig(
    dataset_path = [
        "/home/ajaya-rao/.cache/huggingface/lerobot/float-lab/lesandwich2-groot"
    ],
    output_dir = "/home/ajaya-rao/models/default",
    data_config = "lesandwich_20_fps",
    tune_llm = False,
    tune_visual = False,
    tune_projector = True,
    tune_diffusion_model= False,
    resume = False,
    report_to="wandb",
    embodiment_tag="new_embodiment",
)

LESANDWICH_10FPS_CONFIG = ArgsConfig(
    dataset_path = [
        "/home/ajaya-rao/.cache/huggingface/lerobot/float-lab/lesandwich2-groot"
    ],
    output_dir = "/home/ajaya-rao/models/default",
    data_config = "lesandwich_10_fps",
    tune_llm = False,
    tune_visual = False,
    tune_projector = True,
    tune_diffusion_model= False,
    resume = False,
    report_to="wandb",
    embodiment_tag="new_embodiment",
)

CONFIGS = dict(
    lesandwich_60fps = (
        "LeSandwich Training on 60fps data.",
        LESANDWICH_60FPS_CONFIG
    ),
    lesandwich_30fps = (
        "LeSandwich Training on 30fps data.",
        LESANDWICH_30FPS_CONFIG
    ),
    lesandwich_20fps = (
        "LeSandwich Training on 20fps data.",
        LESANDWICH_20FPS_CONFIG
    ),
    lesandwich_10fps = (
        "LeSandwich Training on 10fps data.",
        LESANDWICH_10FPS_CONFIG
    )
)