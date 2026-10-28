import regex as re
from collections import Counter
import os
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path, vocab_size, special_tokens):
    word_counts = Counter()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(file=f, desired_num_chunks=4, split_special_token=b'<|endoftext|>')
        for start, end in zip(boundaries,boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start)
            text = chunk.decode("utf-8")
            segments = split_on_special_tokens(text, special_tokens)
            counts = count_pretokens(segments)
            word_counts.update(counts)
            
    # segments = split_on_special_tokens(text, special_tokens)
    # word_counts = count_pretokens(segments)
    vocab = build_initial_vocab(special_tokens)
    merges = []
    indexing = build_pair_index(word_counts)
    pairs = count_pairs(word_counts)
    while len(vocab) < vocab_size:
        if not pairs:
            break
        best = select_best_pair(pairs)
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]
        # if len(merges) < 5:
        #     print("word_counts 总词数：", len(word_counts))
        #     count = sum(1 for tup in word_counts if best in zip(tup,tup[1:]))
        #     print("word_counts 中含 best 的 键数：", count)
        #     print("两者相除比例：", count / len(word_counts))
        apply_merge(word_counts, best, indexing, pairs)
    return vocab, merges

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

def split_on_special_tokens(text, special_tokens):
    if not special_tokens:
        return [text]
    special_token_list = []
    for special_token in special_tokens:
        special_token_list.append(re.escape(special_token))
    special_token_str = "|".join(special_token_list)
    return re.split(special_token_str,text)
    
def count_pretokens(segments):
    word_counts = {}
    for segment in segments:
        for match in re.finditer(PAT,segment):
            word = match.group()
            tup = tuple(bytes([b]) for b in word.encode("utf-8"))
            word_counts[tup] = word_counts.get(tup, 0) + 1
    return word_counts

def build_initial_vocab(special_tokens):
    vocab = {}
    for key in range(0,256):
        vocab[key] = bytes([key])
    
    key = len(vocab)
    for special_token in special_tokens:
        vocab[key] = special_token.encode("utf-8")
        key += 1
    return vocab

def count_pairs(word_counts):
    pairs_counts = {}
    for tup, freq in word_counts.items():
        for index in range(len(tup) - 1):
            pair = (tup[index], tup[index + 1])
            pairs_counts[pair] = pairs_counts.get(pair, 0) + freq
    return pairs_counts

def select_best_pair(pairs_counts):
    return max(pairs_counts, key = lambda pair: [pairs_counts[pair], pair])

def build_pair_index(word_counts):
    indexing = {}
    for word in word_counts:
        for pair in zip(word,word[1:]):
            if pair not in indexing:
                indexing[pair] = set()
            indexing[pair].add(word)
    return indexing 

def apply_merge(word_counts, best_pair, indexing, pairs_counts):
    affect_words = list(indexing.get(best_pair, set()))

    for word in affect_words:
        freq = word_counts.pop(word)
        new_key_list = []
        index = 0
        while index < len(word):
            if index != len(word) - 1:
                if (word[index], word[index + 1]) != best_pair:
                    new_key_list.append(word[index])
                else:
                    new_key_list.append(word[index] + word[index + 1])
                    index += 1
            else:
                new_key_list.append(word[index])
            index += 1
        
        new_word = tuple(new_key_list)
        word_counts[new_word] = word_counts.get(new_word, 0) + freq

        for pair in zip(word, word[1:]):
            if pair in indexing:
                indexing[pair].discard(word)
                if not indexing[pair]:
                    del indexing[pair]
        
        for pair in zip(new_word, new_word[1:]):
            if pair not in indexing:
                indexing[pair] = set()
            indexing[pair].add(new_word)


        old_pair_counts = Counter(zip(word, word[1:]))
        new_pair_counts = Counter(zip(new_word, new_word[1:]))
        
        all_pairs = set(old_pair_counts.keys()).union(new_pair_counts.keys())
        
        for pair in all_pairs:
            old_c = old_pair_counts[pair] # 在旧词中出现了几次
            new_c = new_pair_counts[pair] # 在新词中出现了几次
            
            # 净变化量 = 新出现次数 - 旧出现次数
            diff = new_c - old_c
            
            if diff != 0:
                pairs_counts[pair] = pairs_counts.get(pair, 0) + (diff * freq)
                if pairs_counts[pair] == 0:
                    del pairs_counts[pair]
        