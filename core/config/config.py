import yaml
from pathlib import Path
import logging

class FireClassificationConfig:
    def __init__(self):
        config_path = Path(__file__).parent / "config.yaml"
        with config_path.open("r") as file:
            self.config = yaml.safe_load(file)

    def get_config(self, *keys):
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:     %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger()
model_config = FireClassificationConfig()

# Global variable to store root_path override (via command-line args)
_root_path_override = None

# Global seed set by set_seed() — used by dataloaders
_global_seed = 42

def set_global_seed(seed: int):
    global _global_seed
    _global_seed = seed

def get_global_seed() -> int:
    return _global_seed

log_info = log.info
log_warning = log.warning
log_error = log.error
get_logger = lambda: log

def init_results() -> None:
    with open("final_results.txt", "w", encoding="utf-8") as f:
        f.write("| Modelo                  | F1-Score | MaxBoxAcc (IoU 0.5)     | AccV       | AccT       | Thr  |\n")
        f.write("| ----------------------- | -------- | ----------------------- | ---------- | ---------- | ---- |\n")


def write_results(model_name: str, f1: float, val_acc: float, train_acc: float, thr: str = "", boxacc: str = "-----------------------") -> None:
    with open("final_results.txt", "a", encoding="utf-8") as f:
        line = f"| {model_name:<23} | {f1:.4f} | {boxacc:.8f} | {val_acc:.4f} | {train_acc:.4f} | {thr} |\n"
        f.write(line)


def get_config(*keys):
    return model_config.get_config(*keys)

def set_root_path(root_path: str = None):
    """
    Set root_path override for dataset directory.

    Args:
        root_path: Path to override dataset directory (e.g., '/path/to/data')
                  If None, uses config.yaml default
    """
    global _root_path_override
    _root_path_override = root_path
    if root_path:
        log_info(f"Root path override set to: {root_path}")
    else:
        log_info("Root path override cleared, using config.yaml default")

def get_root_path():
    """
    Get the root_path for dataset, respecting command-line override.

    Returns:
        Root path as string. Priority:
        1. Command-line override (--root-path argument)
        2. config.yaml data_path for active dataset
        3. Fallback: "data"
    """
    global _root_path_override

    # If override is set via command-line, use it
    if _root_path_override is not None:
        return _root_path_override

    # Otherwise use config.yaml
    root_path = get_active_dataset_config('data_path') or "data"
    return root_path

def get_active_dataset_config(key: str = None):
    """
    Get configuration for the active dataset

    Args:
        key: Specific key to get from dataset config (e.g., 'num_classes', 'f1_average')
             If None, returns the entire dataset config

    Returns:
        Dataset configuration value or entire config dict

    Example:
        num_classes = get_active_dataset_config('num_classes')  # Returns 2 or 10
        f1_avg = get_active_dataset_config('f1_average')  # Returns 'binary' or 'weighted'
    """
    active_dataset = get_config("active_dataset")
    if active_dataset is None:
        log_warning("No active_dataset defined in config, using 'fire' as default")
        active_dataset = "fire"

    dataset_config = get_config("datasets", active_dataset)

    if dataset_config is None:
        log_error(f"Dataset configuration '{active_dataset}' not found in config")
        return None

    if key is None:
        return dataset_config

    return dataset_config.get(key)


def get_augmentation_config(key: str = None):
    """
    Get augmentation configuration

    Args:
        key: Specific key to get from augmentation config (e.g., 'supervised', 'standard', 'normalization')
             If None, returns the entire augmentation config

    Returns:
        Augmentation configuration value or entire config dict

    Example:
        aug_config = get_augmentation_config()  # Returns entire augmentation config
        supervised = get_augmentation_config('supervised')  # Returns supervised augmentation config
        norm = get_augmentation_config('normalization')  # Returns normalization values
    """
    aug_config = get_config("augmentation")

    if aug_config is None:
        log_warning("No 'augmentation' section found in config, using defaults")
        return None

    if key is None:
        return aug_config

    return aug_config.get(key)

