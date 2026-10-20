"""Smoke-test the sentence-transformers model on six short passages."""

from importlib.metadata import version

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PASSAGES = [
    "Cedar trees are evergreen conifers.",
    "Embeddings map text to numeric vectors.",
    "Retrieval finds passages relevant to a query.",
    "NumPy provides efficient multidimensional arrays.",
    "This model runs locally without an API key.",
    "The smoke test explicitly uses the CPU.",
]


def main() -> None:
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    embeddings = model.encode(PASSAGES, convert_to_numpy=True)

    assert model.device.type == "cpu"
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(PASSAGES), 384)
    assert np.isfinite(embeddings).all()

    print(f"model={MODEL_NAME}")
    print(f"device={model.device.type}")
    print(f"passages={len(PASSAGES)}")
    print(f"embedding_shape={embeddings.shape}")
    print(f"python_packages: sentence-transformers={version('sentence-transformers')}, "
          f"numpy={np.__version__}, torch={torch.__version__}")
    print("status=OK")


if __name__ == "__main__":
    main()

