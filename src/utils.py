import logging
import os
from datetime import datetime
from omegaconf import OmegaConf
import logging
import argparse
from config import ExperimentConfig, MODEL_CONFIG_REGISTRY

def setup_logger(save_dir: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'train_{current_time}.log'

    formatter = logging.Formatter(
        '%(asctime)s|'
        '%(filename)s:%(lineno)d|'
        '%(levelname)s|'
        '%(message)s'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        os.path.join(save_dir, log_filename),
        encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def get_config():
    parser = argparse.ArgumentParser(description="Configuration for SLU model")
    parser.add_argument("--config", type=str, required=False, default=None,
                        help="Path to the config file.")
    args, _ = parser.parse_known_args()

    # configuration from file
    file_conf = OmegaConf.load(args.config) if args.config else OmegaConf.create()
    model_type = file_conf.get("model", {}).get("type", "base")

    if model_type not in MODEL_CONFIG_REGISTRY:
        raise ValueError(f"Not supported model: {model_type}")

    TargetModelConfigClass = MODEL_CONFIG_REGISTRY[model_type]

    # basic scheme
    base_conf = OmegaConf.structured(ExperimentConfig)
    base_conf.model = TargetModelConfigClass()

    # mergine three confs
    final_conf = OmegaConf.merge(base_conf, file_conf)

    return final_conf

def get_test_config(config):
    parser = argparse.ArgumentParser(description="Configuration for evaluation")    
    parser.add_argument("--ckpt", type=str, required=True, default="",
                        help="Path to a ckpt file.")
    parser.add_argument("--epoch", type=int, default=1,
                        help="Epoch number (optional)")
    args, _ = parser.parse_known_args()

    # create OmegaConf for evaluation
    eval_cli_conf = OmegaConf.create({"eval": vars(args)})
    final_conf = OmegaConf.merge(config, eval_cli_conf)

    return final_conf