import os

import numpy as np
import torch


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderFineWeb:
    def __init__(self, B, T, process_rank, num_processes, data_path, split="train"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get the shard filenames
        shards = os.listdir(data_path)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_path, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if process_rank == 0:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def state_dict(self):
        # position relative to this rank's offset, so it's the same on every rank
        rank_offset = self.B * self.T * self.process_rank
        return {
            "shard": self.current_shard,
            "position": self.current_position - rank_offset,
        }

    def load_state_dict(self, state):
        self.current_shard = state["shard"]
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = state["position"] + self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        # measured from rank 0's position so that all ranks move on together
        rank_0_position = self.current_position - B * T * self.process_rank
        if rank_0_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


import tiktoken


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, data_path, split="train"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        with open(data_path, "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = torch.tensor(enc.encode(text))
        # first 90% is train, last 10% is val
        n_train = int(0.9 * len(tokens))
        self.tokens = tokens[:n_train] if split == "train" else tokens[n_train:]
        print(f"Loaded {len(self.tokens)} tokens for split {split}.")
        assert len(self.tokens) >= B * T * num_processes + 1, (
            f"{split} split is too small for B * T * world_size = {B * T * num_processes}"
        )

        # state
        self.reset()

    def reset(self):
        self.current_position = self.B * self.T * self.process_rank

    def state_dict(self):
        # position relative to this rank's offset, so it's the same on every rank
        rank_offset = self.B * self.T * self.process_rank
        return {"shard": 0, "position": self.current_position - rank_offset}

    def load_state_dict(self, state):
        self.current_position = state["position"] + self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B * T * self.num_processes

        # measured from rank 0's position so that all ranks wrap together
        rank_0_position = self.current_position - B * T * self.process_rank
        if rank_0_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.reset()
        return x, y
