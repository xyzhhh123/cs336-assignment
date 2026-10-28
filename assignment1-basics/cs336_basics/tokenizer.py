from .pretokenization import PAT
import re
import regex
from collections.abc import Iterator

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens) -> None:
        self.vocab = vocab
        self.token_ids = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        
        self.pat = regex.compile(PAT)
        
        if special_tokens:
            sorted_tokens = sorted(special_tokens, key= lambda x: len(x), reverse=True)
            self.pattern = "(" + "|".join(re.escape(token) for token in sorted_tokens) + ")"
            self.special_tokens = set(special_tokens)
        else:
            self.pattern = ""
            self.special_tokens = set()

    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        cur = list(tokens)
        while True:
            new_token_list = []
            min_rank = float("inf")
            locate = 0
            for i in range(len(cur) - 1):
                pair = (cur[i], cur[i+1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    locate = i
                    min_rank = rank
            if min_rank == float("inf"):
                return cur
            
            merged_token = cur[locate] + cur[locate+1]
            index = 0
            while index < len(cur):
                if index != locate and index != locate + 1:
                    new_token_list.append(cur[index])
                    index += 1
                elif index == locate:
                    new_token_list.append(merged_token)
                    index += 2
            cur = new_token_list

    def _encode_chunk_stream(self, text: str) -> Iterator[int]:
        if self.pattern:
            parts = re.split(self.pattern, text)
            chunks = [p for p in parts if p]
        else:
            chunks = [text]

        for chunk in chunks:
            if chunk in self.special_tokens:
                tok_b = chunk.encode("utf-8")
                yield self.token_ids[tok_b]
            else:
                for match in self.pat.finditer(chunk):
                    sub_str = match.group()
                    byte_tokens = [bytes([b]) for b in sub_str.encode("utf-8")]
                    merged_tokens = self._apply_merges(byte_tokens)
                    for t in merged_tokens:
                        yield self.token_ids[t]

    def encode(self, text) -> list[int]:
        return list(self._encode_chunk_stream(text))

    def decode(self, ids) -> str:
        raw = b"".join(self.vocab[idx] for idx in ids)
        return raw.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable) -> Iterator[int]:
        buffer = ""
        for item in iterable:
            work_text = buffer + item
            if not self.pattern:
                yield from self._encode_chunk_stream(work_text)
                buffer = ""
                continue
            
            parts = re.split(self.pattern, work_text)
            chunks = [p for p in parts if p]
            if not chunks:
                buffer = ""
                continue

            to_process = chunks[:-1]
            buffer = chunks[-1]

            for c in to_process:
                yield from self._encode_chunk_stream(c)
        
        if buffer:
            yield from self._encode_chunk_stream(buffer)
