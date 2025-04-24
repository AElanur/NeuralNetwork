import pandas as pd
import yaml
from pathlib import Path

def load_config():
    root = Path(__file__).parent.parent
    with open(root/"config/config.yaml") as f:
        return yaml.safe_load(f)

def load_fruit_data():
    cfg = load_config()

    df = pd.read_csv(
        Path(__file__).parent.parent/cfg["data"]["csv_path"],
        usecols=cfg["data"]["features"] + [cfg["data"]["target"]]
    )

    x = df[cfg["data"]["features"]].values
    y = df[cfg["data"]["target"]].values

    return x, y, cfg