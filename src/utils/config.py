"""Configuration management for AML Monitoring project."""

import os
import yaml
from pathlib import Path


class Config:
    """Load and manage project configuration from YAML file."""

    _instance = None
    _config = None

    def __new__(cls, config_path=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path=None):
        if self._config is None:
            if config_path is None:
                config_path = self._find_config()
            self.load(config_path)

    @staticmethod
    def _find_config():
        current = Path(__file__).resolve().parent
        for _ in range(5):
            candidate = current / "configs" / "config.yaml"
            if candidate.exists():
                return str(candidate)
            current = current.parent
        raise FileNotFoundError("config.yaml not found")

    def load(self, config_path):
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)
        self._project_root = str(Path(config_path).resolve().parent.parent)

    @property
    def project_root(self):
        return self._project_root

    def get(self, *keys, default=None):
        if len(keys) == 1 and "." in keys[0]:
            keys = keys[0].split(".")
        result = self._config
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result

    def get_path(self, *keys):
        relative = self.get(*keys)
        if relative is None:
            return None
        return os.path.join(self._project_root, relative)

    @property
    def raw(self):
        return self._config
