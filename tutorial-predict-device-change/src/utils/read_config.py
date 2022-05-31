from box import Box
from typing import Text


def get_config(config_file_path: Text):
    with open(config_file_path) as f:
        config = Box.from_yaml(f)
    return config
