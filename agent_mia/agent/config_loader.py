# agent/config_loader.py
import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.path = Path(config_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def get(self, key: str, default=None):
        parts = key.split(".")
        value = self.cfg
        for p in parts:
            value = value.get(p, {})
        return value if value else default

