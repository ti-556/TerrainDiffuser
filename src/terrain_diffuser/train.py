import argparse
from terrain_diffuser.training.trainer import run_from_yaml

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to YAML config")
    run_from_yaml(p.parse_args().config)