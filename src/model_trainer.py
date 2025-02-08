import re
import os
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
import wandb
from calculator_stopping_criteria import CalculatorStoppingCriteria as CSC
from dotenv import load_dotenv
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datetime import datetime


class LLModelTrainer:

    BASE_MODEL_NAME: str = "Qwen/Qwen2.5-Math-7B"
    HF_USER: str = "brickster241"
    GSM8K_SYS_PROMPT: str
    Stopping_Criteria: CSC
    LLM_Tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer
    QUANT_4_BIT: bool = True
    Quant_Config: BitsAndBytesConfig

    # LORA Paramaters
    Lora_Config: LoraConfig
    LORA_R: int = 8
    LORA_ALPHA: int = 16
    TARGET_MODULES: list[str] = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    LORA_DROPOUT: float = 0.1

    # HyperParamater Optimization
    Train_Parameters: SFTConfig
    EPOCHS: int = 1
    BATCH_SIZE: int = 2
    GRADIENT_ACCUMULATION_STEPS: int = 1
    LEARNING_RATE: float = 1e-4
    LR_SCHEDULER_TYPE: str = "cosine"
    WARMUP_RATIO: float = 0.03
    OPTIMIZER = "adamw_bnb_8bit"
    STEPS: int = 100
    SAVE_INTERVAL: int = 1000
    LOG_TO_WANDB: bool = True

    # HF NAMES
    RUN_NAME: str = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
    PROJECT_RUN_NAME: str = f"Qwen2.5-Math-7B-QLora-4Bit-GSM8K-SFT-{RUN_NAME}"
    HUB_MODEL_NAME: str = f"{HF_USER}/{PROJECT_RUN_NAME}"

    def __init__(self):
        load_dotenv()
        pass

    def load_tokenizer(self):
        self.LLM_Tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_MODEL_NAME, trust_remote_code=True
        )
        self.LLM_Tokenizer.pad_token = self.LLM_Tokenizer.eos_token
        self.LLM_Tokenizer.padding_side = "right"
        self.Stopping_Criteria = CSC(self.LLM_Tokenizer)
        self.BaseModel = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_NAME,
            quantization_config=self.Quant_Config,
            device_map="auto",
        )

    def login_hf_hub(self):
        """
        Logs In to HuggingFace.
        """
        login(token=os.getenv("HF_TOKEN"), add_to_git_credential=True)

    def set_quant_config(self):
        """
        Sets the Quantization Configuration for the HF Model.
        """
        if self.QUANT_4_BIT:
            self.Quant_Config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            self.Quant_Config = BitsAndBytesConfig(
                load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16
            )

    def set_lora_config(self):
        self.Lora_Config = LoraConfig(
            lora_alpha=self.LORA_ALPHA,
            lora_dropout=self.LORA_DROPOUT,
            r=self.LORA_R,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.TARGET_MODULES,
        )

    def set_training_params(self):
        self.Train_Parameters = SFTConfig(
            output_dir=self.PROJECT_RUN_NAME,
            num_train_epochs=self.EPOCHS,
            per_device_train_batch_size=self.BATCH_SIZE,
            per_device_eval_batch_size=1,
            eval_strategy="no",
            gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
            optim=self.OPTIMIZER,
            save_steps=self.SAVE_INTERVAL,
            save_total_limit=10,
            logging_steps=self.STEPS,
            learning_rate=self.LEARNING_RATE,
            weight_decay=0.001,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=self.WARMUP_RATIO,
            group_by_length=True,
            lr_scheduler_type=self.LR_SCHEDULER_TYPE,
            report_to="wandb" if self.LOG_TO_WANDB else None,
            run_name=self.RUN_NAME,
            max_seq_length=512,
            dataset_text_field="text",
            save_strategy="steps",
            hub_strategy="every_save",
            push_to_hub=True,
            hub_model_id=self.HUB_MODEL_NAME,
            hub_private_repo=True,
            seed=3407,
        )
        pass

    def login_wandb(self):
        """
        Logs in to Weight and Biases and configures it to record against our project.
        """
        wandb.login()
        os.environ["WANDB_PROJECT"] = "Qwen2.5-Math7B-GSM8K"
        os.environ["WANDB_LOG_MODEL"] = "checkpoint" if self.LOG_TO_WANDB else "end"
        os.environ["WANDB_WATCH"] = "gradients"
