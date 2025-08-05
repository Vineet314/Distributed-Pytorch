'''This script prepares the Tiny Shakespeare dataset. Needs to be run only once.'''

import tiktoken
import requests 
import numpy as np

def tokenize_and_save():

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)
        text = response.text
        print(f"Downloaded {len(text):,} characters.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")
        return # Exit the function if download fails


    enc = tiktoken.get_encoding("gpt2")
    print("Encoding the entire dataset with the gpt2 tokenizer...")
    tokens = enc.encode(text)
    print(f"Total tokens in dataset: {len(tokens):,}")
    tokens = np.array(tokens, dtype=np.uint16)

    n = int(0.9 * len(tokens))
    train_data = tokens[:n]
    val_data = tokens[n:]

    data_splits = {'train': train_data, 'val': val_data}
    for split, data in data_splits.items():
        file_path = f'{split}.bin'
        print(f"Saving {split} data to {file_path}...")
        with open(file_path, 'wb') as f:
            f.write(data.tobytes())
        print(f"Saved {len(data):,} tokens.")

if __name__ == "__main__":
    tokenize_and_save()
    print("\nData preparation complete. You now have train.bin and val.bin.")