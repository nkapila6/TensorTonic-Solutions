from typing import Dict, List

import numpy as np


class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        words = sum([w.split() for w in texts], [])
        unique_words = sorted(set(words))
        unique_words = self.special_tokens + unique_words
        self.word_to_id = {word: id for id, word in enumerate(unique_words)}
        self.id_to_word = {id: word for id, word in enumerate(unique_words)}
        self.vocab_size += len(unique_words)

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        enc = []
        for word in text.split():
            enc.append(self.word_to_id.get(word, self.word_to_id[self.unk_token]))

        # return [self.word_to_id[w] for w in text.split(" ")]
        return enc

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        return " ".join([self.id_to_word[idx] for idx in ids])
