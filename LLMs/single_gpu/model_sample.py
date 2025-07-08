r'''
This script loads a pre-trained LLM model and generates text from it.
It offers better argument parsing and a more robust way to load the model.

To run this script:
python Single\ GPU/sample.py --model_path flash_llm_model.pt --max_new_tokens 500 --start_text "Hello, world" --device cuda
'''

import sys
import os
import argparse
import torch
import tiktoken
from time import time

# Add the parent directory to the sys.path to find the 'models' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the LLM model definition (assuming it's in models/flash_llm.py)
class config:
    pass

def parse_args():
    """Parses command-line arguments for the sampling script."""
    parser = argparse.ArgumentParser(description='Load a pre-trained LLM model and sample text.')
    parser.add_argument('--model_path',     type=str,   default="flash_llm_model.pt",help='Path to the pre-trained model checkpoint.')
    parser.add_argument('--max_new_tokens', type=int,   default=500,    help='Maximum number of new tokens to generate.')
    parser.add_argument('--temperature',    type=float, default=1.0,    help='Sampling temperature. Higher values make output more random.')
    parser.add_argument('--top_k',          type=int,   default=None,   help='Top-k sampling: sample from the top k most probable tokens.')
    parser.add_argument('--start_text',     type=str,   default="\n",   help='Initial text to prime the generation.')
    parser.add_argument('--device',         type=str,   default='cuda' if torch.cuda.is_available() else 'cpu',help='Device to use for generation (cpu or cuda).')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model with torch.compile()')
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine the device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    model = torch.load(args.model_path, map_location=device, weights_only=False)

    if args.compile:
        print("\ncompiling model...")
        model = torch.compile(model)
    model.eval() # Set model to evaluation mode

    enc = tiktoken.get_encoding('gpt2')

    start_ids = enc.encode(args.start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    t = time()
    with torch.no_grad():
        generated_ids = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    dt = time() - t

    generated_text = enc.decode(generated_ids[0].tolist())

    print(f'\n\n{generated_text}')
    print(f'\n\n--------------------------------------\n\nTime taken to generate = {dt:.2f}s')

if __name__ == '__main__':
    main()