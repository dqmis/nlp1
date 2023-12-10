from typing import Optional

import numpy as np
import torch

from src.models.models import BOW, CBOW, DeepCBOW, NLPModel
from src.vocab.vocabulary import Vocabulary

MODELS = {
    "BOW": BOW,
    "CBOW": CBOW,
    "DeepCBOW": DeepCBOW,
}


def init_model(
    model_name: str, embeddings: Optional[np.ndarray], vocab: Vocabulary
) -> NLPModel:
    vocab_size = len(vocab.w2i)
    model = MODELS[model_name](vocab_size=vocab_size, vocab=vocab)
    if embeddings is not None:
        model.embed.weight.data.copy_(torch.from_numpy(embeddings))
