import os
import sys

import numpy as np
import tiktoken
from datasets import load_dataset

OUT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/edu_fineweb"
SIZES = [("val", 10_000_000), ("train", 50_000_000)]

os.makedirs(OUT, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]
docs = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
docs = iter(docs)

for i, (split, n) in enumerate(SIZES):
    buf = np.empty(n, dtype=np.uint16)
    filled = 0
    while filled < n:
        batch = [next(docs)["text"] for _ in range(256)]
        for toks in enc.encode_ordinary_batch(batch):
            toks = [eot] + toks  # documents are separated by <|endoftext|>
            take = min(len(toks), n - filled)
            buf[filled : filled + take] = toks[:take]
            filled += take
            if filled == n:
                break
    path = os.path.join(OUT, f"edufineweb_{split}_{i:06d}.npy")
    np.save(path, buf)
    print(f"wrote {path}: {n:,} tokens")