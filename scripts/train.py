"""
• Ability to configure and control the various model and optimizer hyperparameters.
• Memory-efficient loading of large training and validation datasets with np.memmap.
• Serializing checkpoints to a user-provided path.
• Periodically logging training and validation performance (e.g., to console and/or an external
service like Weights and Biases).
"""
from contextlib import nullcontext
import time
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from cs336_basics.tokenizer import BPE_Tokenizer
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
    print("Training tokenizer")
    tokenizer.train_from_file(file_path=data_path, vocab_size=vocab_size, special_tokens=special_tokens)
    print("Training finished")
    return tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--lr-min", default=1e-5, type=float)
    parser.add_argument("--warmup-steps", default=100, type=int)

    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--weight-decay", default=0.1, type=float)

    parser.add_argument("--vocab-size", default=16384, type=int)
    parser.add_argument("--num-heads", default=16, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=512, type=int)
    parser.add_argument("--context-length", default=256, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    # parser.add_argument("--train-step", default=65536, type=int)
    parser.add_argument("--grad-clip-norm", default=0.1, type=float)
    
    parser.add_argument("--val-every", default=100, type=int)
    parser.add_argument("--save-every", default=1000, type=int)
    parser.add_argument("--save-dir", default='save', type=str)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--train-tokenizer", action='store_true')
    parser.add_argument("--train-data", default='data/TinyStoriesV2-GPT4-train.txt',type=str)
    parser.add_argument("--val-data", default='data/TinyStoriesV2-GPT4-valid.txt',type=str)
    parser.add_argument("--total-token", default=327680000, type=int)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default='basic_lm', type=str)
    parser.add_argument("--wandb-run", default='', type=str) # Gen by random by default

    parser.add_argument("--profile", action="store_true")
    
    args = parser.parse_args()
    args.d_ff = round((8 / 3 * args.d_model) / 64) * 64
    print(args)
    return args

def accounting(args):
    params, flops, _, _ = calc_params_flops(
        V=args.vocab_size,
        S=args.context_length,
        L=args.num_layers,
        D=args.d_model,
        H=args.num_heads,
        D_=args.d_ff
    )
    return params, flops * args.batch_size * 3

def main():
    
    args = parse_args() 

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    tk_save_path = os.path.join(args.save_dir, "tokenizer.json")
    pt_save_path = os.path.join(args.save_dir, "save.pt")
    print(f"{tk_save_path}, {os.path.exists(tk_save_path)}")
    
    if not os.path.exists(tk_save_path) or args.train_tokenizer:
        print("Training tokenizer from scratch")
        tk = get_tokenizer(args.val_data, args.vocab_size)
        tk.save(tk_save_path)
    else:
        print("Resume tokenizer")
        tk = BPE_Tokenizer().load(os.path.join(args.save_dir, "tokenizer.json"))

    if not os.path.exists("train.bin"):
        encode_to_bin(tk, args.train_data, 'train.bin')
        encode_to_bin(tk, args.val_data, 'valid.bin')
        print("Encode data to bin")
    else:
        print("Use existing data bin")

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
    prof = profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        schedule=schedule(
            wait=5,
            warmup=5,
            active=10,
            repeat=1
        ),
        on_trace_ready=tensorboard_trace_handler("log/profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) if args.profile else nullcontext() 
    # optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    optim = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    if args.resume:
        step = load_checkpoint(pt_save_path, model, optim) + 1
        print(f"Resume from step {step}")
    else:
        step = 1
    params, flops = accounting(args)

    print(f"Params: {params/1000**2} M, FLOPs: {flops/1000**3} GFLOPs")
    
    target_step = args.total_token / args.batch_size / args.context_length
    print(f"Target Step: {target_step}")

    peak = 56.28e12 # For my RTX5080
    bar = tqdm(range(step, target_step+1), desc="Training Step", initial=step - 1)
    with prof:
        for step in bar:
            token, gt = data_load(token_ids=train_data, batch_size=args.batch_size, context_length=args.context_length, device_str=None, device=device)

            lr = get_lr_(step, args.lr_min, args.lr, args.warmup_steps, args.train_step)
            for group in optim.param_groups:
                group["lr"] = lr

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            t0 = time.perf_counter()
            optim.zero_grad()
            pred_logit = model(token)
            loss = cross_entropy(pred_logit, gt)
            loss.backward()
            grad_clip(model.parameters(), max_l2_norm=args.grad_clip_norm)
            optim.step()
            
            if args.profile:
                prof.step()
                if step >= 25:
                    break
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            t1 = time.perf_counter()
            dt = t1 - t0
            MFU = flops / dt / peak
            bar.set_postfix({"MFU":MFU, "loss": float(loss)})
            
            if args.use_wandb:
                wandb.log(data={
                    "train_loss": float(loss)
                }, step = step)
                
            if step % args.val_every == 0:
                with torch.no_grad():
                    model.eval()
                    val, val_gt = data_load(token_ids=valid_data, batch_size=args.batch_size, context_length=args.context_length, device_str=None, device=device)
                    pred_logit = model(val)
                    loss = cross_entropy(pred_logit, val_gt) 
                    if args.use_wandb:
                        wandb.log(data={
                            'val_loss': float(loss)
                        }, step=step)
                model.train()

            if step % args.save_every == 0:
                save_checkpoint(model, optim, step, pt_save_path)
                print(f"Save step {step}")
                
    if args.profile:
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=20,
            )
        )

if __name__ == "__main__":
    main()