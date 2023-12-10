import torch
from torch import nn

from src.vocab.vocabulary import Vocabulary


class NLPModel(nn.Module):
    def freeze_embeddings(self) -> None:
        self.embed.weight.requires_grad = False

    def unfreeze_embeddings(self) -> None:
        self.embed.weight.requires_grad = True


class BOW(NLPModel):
    """A simple bag-of-words model"""

    def __init__(
        self,
        vocab_size: int,
        vocab: Vocabulary,
        embedding_dim: int = 300,
    ) -> None:
        super(BOW, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeds = self.embed(inputs)
        logits: torch.Tensor = embeds.sum(1) + self.bias

        return logits


class CBOW(NLPModel):
    """A simple bag-of-words model"""

    def __init__(
        self,
        vocab_size: int,
        vocab: Vocabulary,
        embedding_dim: int = 300,
        output_dim: int = 5,
    ) -> None:
        super(CBOW, self).__init__()
        self.vocab = vocab

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, output_dim, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeds = self.embed(inputs)

        out = self.linear1(embeds)
        out = out.sum(1)

        return out


class DeepCBOW(NLPModel):
    """A simple bag-of-words model"""

    def __init__(
        self,
        vocab_size: int,
        vocab: Vocabulary,
        embedding_dim: int = 300,
        hidden_dim: int = 100,
        output_dim: int = 5,
    ):
        super(DeepCBOW, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeds = self.embed(inputs)

        out = self.model(embeds)
        out = out.sum(1)

        return out


class PTDeepCBOW(NLPModel):
    def __init__(
        self,
        vocab_size: int,
        vocab: Vocabulary,
        embedding_dim: int = 300,
        hidden_dim: int = 100,
        output_dim: int = 5,
    ):
        super(PTDeepCBOW, self).__init__(
            vocab_size, embedding_dim, hidden_dim, output_dim, vocab
        )
