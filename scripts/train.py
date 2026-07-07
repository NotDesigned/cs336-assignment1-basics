"""
• Ability to configure and control the various model and optimizer hyperparameters.
• Memory-efficient loading of large training and validation datasets with np.memmap.
• Serializing checkpoints to a user-provided path.
• Periodically logging training and validation performance (e.g., to console and/or an external
service like Weights and Biases).
"""
import torch
from cs336_basics.bpe_tokenizer import BPE_Tokenizer
from cs336_basics.modules import *
import wandb
import numpy as np
import argparse
from tqdm import tqdm

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--weight-decay", default=0.1, type=float)

    parser.add_argument("--vocab-size", default=16384, type=int)
    parser.add_argument("--num-heads", default=12, type=int)
    parser.add_argument("--num-layers", default=24, type=int)
    parser.add_argument("--d-model", default=512, type=int)
    parser.add_argument("--context-length", default=1024, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--train-step", default=65536, type=int)
    
    parser.add_argument("--val-every", default=100, type=int)
    parser.add_argument("--save-every", default=1000, type=int)
    parser.add_argument("--save", default='ckpt/save.ckpt', type=str)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--train-data", default='data/TinyStoriesV2-GPT4-train.txt',type=str)
    parser.add_argument("--val-data", default='data/TinyStoriesV2-GPT4-valid.txt',type=str)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default='basic_lm', type=str)
    parser.add_argument("--wandb-run", default='', type=str) # Gen by random by default
    
    args = parser.parse_args()
    args.d_ff = round((8 / 3 * args.d_model) / 64) * 64
    return args

def main():
    
    args = parse_args() 
    tk = get_tokenizer(args.train_data, args.vocab_size)
    
    encode_to_bin(tk, args.train_data, 'train.bin')
    encode_to_bin(tk, args.val_data, 'valid.bin')
    train_data = np.memmap(filename='train.bin', dtype=np.int32, mode='r')
    valid_data = np.memmap(filename='valid.bin', dtype=np.int32, mode='r')
    
    device = get_device()
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=None if args.wandb_run == '' else args.wandb_run)
    
    model = TransformerLM(
        vocab_size=args.vocab_size, 
        num_heads=args.num_heads,
        num_layers=args.num_layers, 
        d_model=args.d_model, 
        d_ff=args.d_ff,
        content_length=args.context_length,
        device=device
    )
    optim = AdamW(model.parameters(), args.lr, (args.beta1, args.beta2), args.weight_decay)
    if args.resume:
        step = load_checkpoint(args.save, model, optim)
    else:
        step = 1

    target_step = args.train_step
    for step in tqdm(range(1, target_step+1), desc="Training Step"):
        optim.zero_grad()
        token, gt = data_load(token_ids=train_data, batch_size=args.batch_size, context_length=args.context_length, device_str=None) 
        pred_logit = model(token)
        loss = cross_entropy(pred_logit, gt)
        loss.backward()
        optim.step()
        if args.use_wandb:
            wandb.log(data={
                "train_loss": float(loss)
            }, step = step)
        if step % args.val_every == 0:
            val, val_gt = data_load(token_ids=valid_data, batch_size=args.batch_size, context_length=args.context_length, device_str=None)
            pred_logit = model(val)
            loss = cross_entropy(pred_logit, val_gt) 
            if args.use_wandb:
                wandb.log(data={
                    'val_loss': float(loss)
                }, step=step)
        if step % args.save_every == 0:
            save_checkpoint(model, optim, step, args.save)
        step += 1

if __name__ == "__main__":
    main()