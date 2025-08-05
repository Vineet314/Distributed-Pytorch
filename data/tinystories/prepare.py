'''This script prepares the requested dataset. Needs to be run only once for a dataset.'''

import os
import tiktoken
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

n_proc = os.cpu_count() // 2
DATASET_NAME = "roneneldan/TinyStories"
EOT_TOKEN_ID = 50256 # End-of-text token for gpt2

def tokenize_and_save():
    """
    Downloads, tokenizes, and saves the TinyStories dataset.
    This version uses the `map` method for efficient tokenization.
    """

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, cache_dir='./cache', num_proc=n_proc)

    split_dataset = dataset['train'].train_test_split(test_size=0.01, seed=1729, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # Rename 'test' to 'val'

    enc = tiktoken.get_encoding("gpt2")

    def tokenize(example):
        """Tokenization function to be applied with `map`."""
        ids = enc.encode_ordinary(example['text'])
        ids.append(EOT_TOKEN_ID)  # Add End-of-Text token
        return {'ids': ids, 'len': len(ids)}

    # Tokenize parallel
    tokenized = split_dataset.map(
        tokenize,
        remove_columns=['text'],
        desc="Tokenizing",
        num_proc=n_proc)

    # Save the tokenized datasets to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        file_path = f'{split}.bin'
        dtype = np.uint16 # GPT-2 tokenizer has 50257 tokens

        # Write to binary file
        print(f"Saving {split} split to {file_path}")
        with open(file_path, 'wb') as f:
            for example in tqdm(dset, desc=f"Writing {split}", total=len(dset)):
                f.write(np.array(example['ids'], dtype=dtype).tobytes())

        print(f"Saved {arr_len:,} tokens to {file_path}")

if __name__ == "__main__":
    tokenize_and_save()
    print("Data preparation complete.")