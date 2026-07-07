"""
• Ability to configure and control the various model and optimizer hyperparameters.
• Memory-efficient loading of large training and validation datasets with np.memmap.
• Serializing checkpoints to a user-provided path.
• Periodically logging training and validation performance (e.g., to console and/or an external
service like Weights and Biases).
"""
import torch
from bpe_tokenizer import BPE_Tokenizer
from modules import *
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
                np.array(buffer, dtype=np.int32).tofile(out)
                buffer.clear()
        if buffer:
            np.array(
                buffer,
                dtype=np.int32
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
            'context_length': 1024
        },

        'train_step': 65536,
        'batch_size': 16
    }

VAL_PER_STEP = 100
SAVE_PER_STEP = 1000
SAVE_PATH = 'ckpt/save.ckpt'

RESUME=False

def main():
    config = get_config()
    tk = get_tokenizer(TRAIN_DATA_PATH, config['model']['vocab_size'])
    
    encode_to_bin(tk, TRAIN_DATA_PATH, 'train.bin')
    encode_to_bin(tk, VALID_DATA_PATH, 'valid.bin')
    train_data = np.memmap(filename='train.bin', dtype=np.int32, mode='r')
    valid_data = np.memmap(filename='valid.bin', dtype=np.int32, mode='r')
    
    device = get_device()
    wandb.init(project=WANDB_PROJECT_NAME, name=WANDB_RUN_NAME, config=config)
    
    model = TransformerLM(**config['model'], device=device)
    optim = AdamW(model.parameters(), **config['optim'])
    if RESUME == False:
        step = 1
    else:
        step = load_checkpoint(SAVE_PATH, model, optim)

    target_step = config['train_step']
    while step <= target_step:
        optim.zero_grad()
        token, gt = data_load(token_ids=train_data, batch_size=config['batch_size'], context_length=config['model']['context_length'], device_str=None) 
        pred_logit = model(token)
        loss = cross_entropy(pred_logit, gt)
        loss.backward()
        optim.step()
        wandb.log(data={
            "train_loss": float(loss)
        }, step = step)
        if step % VAL_PER_STEP == 0:
            val, val_gt = data_load(token_ids=valid_data, batch_size=config['batch_size'], context_length=config['model']['context_length'], device_str=None)
            pred_logit = model(val)
            loss = cross_entropy(pred_logit, val_gt) 
            wandb.log(data={
                'val_loss': float(loss)
            }, step=step)
        if step % SAVE_PER_STEP == 0:
            save_checkpoint(model, optim, step, SAVE_PATH)
        step += 1

if __name__ == "__main__":
    main()