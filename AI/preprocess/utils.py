from torch import nn


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x) + x


class SeqTokenizer:
    def __init__(self, alphabet: str) -> None:
        self.ascii_code = np.frombuffer(alphabet.encode(), dtype=np.int8)
        self.int2idx = np.empty(self.ascii_code.max() + 1, dtype=int)
        for i, c in enumerate(self.ascii_code):
            self.int2idx[c] = i

    def __call__(self, seq: str) -> np.ndarray:
        return self.int2idx[np.frombuffer(seq.encode(), dtype=np.int8)]
