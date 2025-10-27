import pickle

import regex as re
import os
import multiprocessing as mp
import time
from collections import defaultdict, Counter
from typing import BinaryIO

def BPE_trainer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:
    #validate input
    assert len(special_tokens) + 256 <= vocab_size

    #initialize vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    #initialize merges
    merges = []

    #initialize pair counters
    pair_counter = Counter()

    #initialize pre_tokens and pair_counter
    pre_tokens = pre_tokenize_all(input_path,special_tokens,8)
    for pre_token_seq, freq in pre_tokens.items():
        for a, b in zip(pre_token_seq, pre_token_seq[1:]):
            pair_counter[(a, b)] += freq

    #update merging process
    while len(vocab) < vocab_size:
        #found the best merging choice based on (freq, (a, b))
        best_merge = max(pair_counter.items(), key = lambda x: (x[1],x[0]))[0]
        a, b = best_merge
        #update learned merge list
        merges.append(best_merge)
        #update vocab dict & concatenate two bytes
        vocab[len(vocab)] = a + b

        #apply the merge (update pre_token and pair_counter)
        pre_tokens, pair_counter = apply_merge(pre_tokens, pair_counter, best_merge)

    #save to disk
    with open("./tokenizer/TS_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("./tokenizer/TS_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    return vocab, merges

def apply_merge(pre_tokens, pair_counter, best_merge):
    a, b = best_merge
    merged = a + b
    new_pre_tokens = defaultdict(int)

    for pre_token_seq, freq in pre_tokens.items():
        new_token_seq = []
        i = 0
        seq_len = len(pre_token_seq)

        #check every byte in every pre_token_seq
        while i < seq_len:
            if i < seq_len -1 and (pre_token_seq[i], pre_token_seq[i+1]) == best_merge:
                left = pre_token_seq[i-1] if i > 0 else None
                right = pre_token_seq[i+2] if i < seq_len - 2 else None

                if left is not None:
                    pair_counter[(left, a)] -= freq
                    pair_counter[(left, merged)] = pair_counter.get((left, merged), 0) + freq
                if right is not None:
                    pair_counter[(b,right)] -= freq
                    pair_counter[(merged, right)] = pair_counter.get((merged, right), 0) + freq

                #replace a, b by merging into ab
                new_token_seq.append(merged)
                i+=2
            else:
                new_token_seq.append(pre_token_seq[i])
                i+=1

        new_token_seq = tuple(new_token_seq)
        new_pre_tokens[new_token_seq] += freq

    # Clean up zero or negative entries from pair_counter
    del pair_counter[best_merge]
    to_delete = [p for p, c in pair_counter.items() if c <= 0]
    for p in to_delete:
        del pair_counter[p]

    # return updated vocab and counters
    return new_pre_tokens, pair_counter

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize_chunks(
    file_path: str,
    start,
    end,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    pretoken = defaultdict(int)
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        text = chunk

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    #special_token_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    special_token_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    segments = re.split(special_token_pattern, text)

    for segment in segments:
        matches = re.finditer(PAT, segment)
        for match in matches:
            token = match.group(0)
            token_bytes = token.encode("utf-8")
            key = tuple(bytes([b]) for b in token_bytes)
            pretoken[key] += 1
    return pretoken

def pre_tokenize_all(
    file_path: str,
    #"../data/TinyStoriesV2-GPT4-valid.txt",
    special_tokens: list[str],
    # = ["<|endoftext|>"],
    num_processes=8,
) -> dict[tuple[bytes], int]:

    global_pretoken= defaultdict(int)

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    args_list = [(file_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    # Parallelizing pre-tokenization
    start_time = time.perf_counter()
    with mp.Pool(num_processes) as pool:
        pretoken_chunks = pool.starmap(pre_tokenize_chunks, args_list)

    #End time of all pre-tokenization
    end_time = time.perf_counter()
    print("Time of pre-tokenization:", end_time - start_time)

    # Merging all workers' vocab into global one
    for pretoken in pretoken_chunks:
        for key, value in pretoken.items():
            assert b"<|" not in key, "Pre-tokenization error"
            global_pretoken[key] += value
    return global_pretoken

if __name__ == "__main__":
    BPE_trainer(
        "../data/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"],)