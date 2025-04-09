from .data_path import YAML_FILE
import yaml

def load_config(config_path = YAML_FILE):
    """Loads the configuration from the YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config