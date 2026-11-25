"""Token offsets for measuring and cutting text by token count."""

import re
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenOffsets:
    """Character span of each token in a text. Lets callers count tokens in any span cheaply."""

    starts: list[int]
    ends: list[int]

    def __len__(self) -> int:
        return len(self.starts)

    def count(self, char_start: int, char_end: int) -> int:
        """Number of tokens lying wholly inside the character span."""
        return max(0, bisect_right(self.ends, char_end) - bisect_left(self.starts, char_start))


class TokenCounter(ABC):
    @abstractmethod
    def offsets(self, text: str) -> TokenOffsets: ...


class HuggingFaceTokenCounter(TokenCounter):
    def __init__(self, name: str, revision: str) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_pretrained(name, revision=revision)
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def offsets(self, text: str) -> TokenOffsets:
        spans = self._tokenizer.encode(text, add_special_tokens=False).offsets
        return TokenOffsets([start for start, _ in spans], [end for _, end in spans])


class RegexTokenCounter(TokenCounter):
    """Dependency-free tokenizer for tests: words and single punctuation marks."""

    _TOKEN = re.compile(r"\w+|[^\w\s]")

    def offsets(self, text: str) -> TokenOffsets:
        spans = [match.span() for match in self._TOKEN.finditer(text)]
        return TokenOffsets([start for start, _ in spans], [end for _, end in spans])
