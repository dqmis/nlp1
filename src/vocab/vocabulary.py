# Here we first define a class that can map a word to an ID (w2i)
# and back (i2w).

from collections import Counter, OrderedDict


class _OrderedCounter(Counter, OrderedDict):  # type: ignore
    """Counter that remembers the order elements are first seen"""

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self) -> tuple:
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""

    def __init__(self) -> None:
        self.freqs = _OrderedCounter()
        self.w2i: dict = {}
        self.i2w: list = []

    def count_token(self, t: str) -> None:
        self.freqs[t] += 1

    def add_token(self, t: str) -> None:
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self, min_freq: int = 0) -> None:
        """
        min_freq: minimum number of occurrences for a word to be included
                in the vocabulary
        """
        self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
        self.add_token("<pad>")  # reserve 1 for <pad> (discussed later)

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)
