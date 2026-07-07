"""
• Ability to configure and control the various model and optimizer hyperparameters.
• Memory-efficient loading of large training and validation datasets with np.memmap.
• Serializing checkpoints to a user-provided path.
• Periodically logging training and validation performance (e.g., to console and/or an external
service like Weights and Biases).
"""
import torch
from bpe_tokenizer import BPE_Tokenizer
from modules import TransformerLM, load_checkpoint, save_checkpoint, get_device, AdamW, data_load
import wandb
import numpy as np
"""
# Saving & logging
CKPT_SAVE_PATH: str
WANDB_PROJECT_NAME: str
WANDB_RUN_NAME: str

# Optimizer states
lr: float
beta1: float
beta2: float
iter: int

# Training Data

DATA_PATH: str
VOCAB_SIZE: int

# Model Configuration

num_heads: int
num_layers: int
d_model: int
d_ff: int
context_length: int
"""

WANDB_PROJECT_NAME = "basic-lm"
WANDB_RUN_NAME = "test"

TRAIN_DATA_PATH = "data/owt_train.txt"
VALID_DATA_PATH = 'data/owt_valid.txt'

def encode_to_bin(tokenizer, input_path, output_path, buffer_size= 1024**2):
    buffer = []
    with open(input_path, "r",  encoding='utf-8') as f, open(output_path, 'wb') as out:
        for token_id in tokenizer.encode_iterable(f):
            buffer.append(token_id)
            if len(buffer) >= buffer_size:
                np.array(buffer, dtype=np.uint16).tofile(out)
                buffer.clear()
        if buffer:
            np.array(
                buffer,
                dtype=np.uint16
            ).tofile(out)

def get_tokenizer(data_path:str, vocab_size:int, special_tokens: list[str] = ['<|endoftext|>']):
    tokenizer = BPE_Tokenizer()
    tokenizer.train_from_file(file_path=data_path, vocab_size=vocab_size, special_tokens=special_tokens)
    return tokenizer

def get_config() -> dict:
    return {
        'train_data_path': TRAIN_DATA_PATH,
        'valid_data_path': VALID_DATA_PATH,
        
        'optim':{
            'lr': 1e-2,
            'beta1': 0.9,
            'beta2': 0.99,
        },
        
        'model':{
            'vocab_size': 10000,
            'num_heads': 12,
            'num_layers': 24,
            'd_model': 512,
            'd_ff': 1280,
            'context_length': 10000
        },

        'step': 0,
    }

def main():
    config = get_config()
    tk = get_tokenizer(VALID_DATA_PATH, config['model']['vocab_size'])
    
    encode_to_bin(tk, TRAIN_DATA_PATH, 'train.bin')
    encode_to_bin(tk, VALID_DATA_PATH, 'valid.bin')
    train_data = np.memmap(filename=TRAIN_DATA_PATH, dtype=np.uint16, mode='r')
    valid_data = np.memmap(filename=VALID_DATA_PATH, dtype=np.uint16, mode='r')
    
    device = get_device()
    wandb.init(project=WANDB_PROJECT_NAME, name=WANDB_RUN_NAME, config=config)
    model = TransformerLM(**config['model'], device=device)
    optim = AdamW(model.state_dict(), **config['optim'])
    a,b = data_load(token_ids=train_data, batch_size=256, context_length=1024, device_str=None) 
    print(a)
    

if __name__ == "__main__":
    main()