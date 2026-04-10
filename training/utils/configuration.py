from pathlib import Path

import yaml


DEFAULT_GLOBAL_CONFIG_PATH = Path("./config/global_configuration.yaml")


def load_global_configuration(config_path=DEFAULT_GLOBAL_CONFIG_PATH):
	config_path = Path(config_path)

	if not config_path.exists():
		raise FileNotFoundError(f"Global configuration not found at: {config_path}")

	with config_path.open("r", encoding="utf-8") as file:
		config = yaml.safe_load(file) or {}

	model_config = config.get("model_architecture", {})
	if "head_size" not in model_config and model_config.get("heads_num"):
		model_config["head_size"] = model_config["embedding_dim"] // model_config["heads_num"]
		config["model_architecture"] = model_config

	return config
