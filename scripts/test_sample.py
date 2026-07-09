import torch
from cs336_basics.tokenizer import *
from cs336_basics.modules import *
from cs336_basics.sample import test_completion
from train import parse_args

def main():
    prompt = str(input("Prompt:"))
    tk = BPE_Tokenizer().load("save/tokenizer.json")
    device = torch.device("cuda")
    
    tokens = tk.encode(prompt)
    tokens = torch.as_tensor(tokens, device=device)
    args = parse_args()
    
    model = TransformerLM(
        vocab_size=args.vocab_size, 
        num_heads=args.num_heads,
        num_layers=args.num_layers, 
        d_model=args.d_model, 
        d_ff=args.d_ff,
        content_length=args.context_length,
        device=device
    )
    optim = AdamW(model.parameters())
    load_checkpoint("save/save.pt", model, optim)

    generation: str = test_completion(model, prompt, tk, max_generated=100, temperature=0.1, p=0.5)
    print(generation)

main()