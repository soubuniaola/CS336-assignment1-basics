
from collections.abc import Iterable, Iterator
import pickle as pk
import regex

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.byte2id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        #load vocab
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pk.load(vocab_file)

        #load merges
        with open(merges_filepath, "rb") as merges_file:
            merges = pk.load(merges_file)

        #return a BPETokenizer
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        encoded = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_token_patter = "|".join(regex.escape(tok) for tok in sorted_special_tokens)
            segments = regex.split(f"({special_token_patter})", text)
        else:
            segments = [text]

        for segment in segments:
            if not segment:
                continue
            elif self.special_tokens and segment in self.special_tokens:
                encoded += [self.byte2id[segment.encode("utf-8")]]
                continue
            matches = regex.finditer(PAT, segment)
            for match in matches:
                text_bytes = [bytes([b]) for b in match.group(0).encode("utf-8")] ### IMPORTANT
                for (a, b) in self.merges:
                    if len(text_bytes) == 1:
                        break
                    new_bytes = []
                    i = 0
                    while i < len(text_bytes):
                        if i < len(text_bytes) - 1 and (text_bytes[i], text_bytes[i + 1]) == (a, b):
                            new_bytes.append(a + b)
                            i += 2
                        else:
                            new_bytes.append(text_bytes[i])
                            i += 1
                    text_bytes = new_bytes
                current_encoded = [self.byte2id[b] for b in text_bytes]
                encoded += current_encoded
        return encoded


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        decoded = b''.join(self.vocab.get(id, b"\xef\xbf\xbd") for id in ids)
        return decoded.decode("utf-8","replace")


if __name__ == "__main__":
    btok = BPETokenizer.from_files("./tokenizer/vocab.pkl", "./tokenizer/merges.pkl",special_tokens=["<|endoftext|>"])
    encoded = btok.encode("hello, this is zehan's <|endoftext|> debugging phase <|endoftext|><|endoftext|>")
