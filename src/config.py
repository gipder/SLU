from dataclasses import dataclass, field
from typing import Any, Dict, Type

#
# ExperimentConfig Experiment Configuration
# |
# |-- model: ModelConfig
# |-- data: DataConfig
# |-- eval: EvalConfig
@dataclass
class EvalConfig:
    ckpt: str = ""
    epoch: int = -1
    batch: int = 4


@dataclass
class ModelConfig:
    type: str = "base"


@dataclass
class SLUConfig(ModelConfig):
    type: str = "slu_v0"
    llm_name: str = "microsoft/deberta-v3-base"
    upsample_factor: int = 8
    d_model: int = 768
    dropout: float = 0.1
    use_log_probs: bool = True
    blank_token_id: int = 0
    vocab_size: int = 650


@dataclass
class DataConfig:
    train_input_files: str = "data/STOP_text/low.eval.asr"
    valid_input_files: str = "data/STOP_text/low.eval.asr"
    test_input_files: str = "data/STOP_text/low.eval.asr"

    train_output_files: str = "data/STOP_text/low.eval.slu"
    valid_output_files: str = "data/STOP_text/low.eval.slu"
    test_output_files: str = "data/STOP_text/low.eval.slu"    

    shuffle_in_train: bool = True    
    shuffle_in_eval: bool = False

    bpe_model_path: str = "data/STOP_text/bpe_650.model"
    max_length: int = 256
    blank: str = "<blk>"
    MASK: str = "[MASK]"


@dataclass
class ExperimentConfig:    
    start_epoch: int = 1
    end_epoch: int = 100
    exp_name: str = "DeBERTa_CTC_SLU"
    save_dir: str = "./exp/ctc_loss"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    num_workers: int = 4

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]):
        data_cfg = DataConfig(**cfg_dict.get('data', {}))

        model_dict = cfg_dict.get("model", {})
        model_type = model_dict.get("model_type", "base")

        config_cls = MODEL_CONFIG_REGISTRY.get(model_type)
        if config_cls is None:
            raise ValueError(f"Unknown model type: {model_type}")

        # create an model cfg from the class
        model_cfg = config_cls(**model_dict)
        exp_args = {k: v for k, v in cfg_dict.items() if k not in ['data', 'model']}

        return cls(data=data_cfg, model=model_cfg, **exp_args)


MODEL_CONFIG_REGISTRY: Dict[str, Type[ModelConfig]] = {
    "base": ModelConfig,
    "slu_v0" : SLUConfig,    
}