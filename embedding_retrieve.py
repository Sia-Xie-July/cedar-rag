"""Inspect and use real MiniLM embeddings for the six Cedar Lab passages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from retrieve import Chunk, load_chunks


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DISPLAYED_COORDINATES = 12
NUMERICAL_TOLERANCE = 1e-6


@dataclass(frozen=True)
class EmbeddingResult:
    chunk: Chunk
    score: float
    document_index: int


class EmbeddingRetriever:
    """Load one model and compute the document matrix once per program run."""

    def __init__(self) -> None:
        self.chunks = load_chunks()
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        passage_texts = [chunk.text for chunk in self.chunks]
        self.document_embeddings = self.model.encode_document(
            passage_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def search(self, question: str) -> tuple[np.ndarray, list[EmbeddingResult]]:
        question_embedding = self.model.encode_query(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores = self.document_embeddings.astype(np.float64) @ question_embedding.astype(
            np.float64
        )
        ranked = sorted(
            (
                EmbeddingResult(chunk, float(scores[index]), index)
                for index, chunk in enumerate(self.chunks)
            ),
            key=lambda result: (-result.score, result.chunk.chunk_id),
        )
        return question_embedding, ranked


def format_coordinates(vector: np.ndarray, count: int = DISPLAYED_COORDINATES) -> str:
    return "[" + ", ".join(f"{float(value):.4f}" for value in vector[:count]) + "]"


def render_report(retriever: EmbeddingRetriever, questions: list[str], k: int) -> str:
    lines: list[str] = []
    documents = retriever.document_embeddings

    lines.append("CEDAR-RAG embedding retriever")
    lines.append(f"model={MODEL_NAME}")
    lines.append(f"device={retriever.model.device.type}")
    lines.append(
        "versions: "
        f"python={platform.python_version()}, "
        f"sentence-transformers={version('sentence-transformers')}, "
        f"numpy={np.__version__}, torch={torch.__version__}, "
        f"transformers={version('transformers')}"
    )
    lines.append(f"document_matrix_shape={documents.shape}")
    lines.append("Document vectors (first 12 coordinates):")
    for chunk, vector in zip(retriever.chunks, documents, strict=True):
        vector_length = float(np.linalg.norm(vector.astype(np.float64)))
        approximately_unit = bool(np.isclose(vector_length, 1.0, atol=NUMERICAL_TOLERANCE))
        lines.append(
            f"{chunk.chunk_id} | {chunk.filename} | length={vector_length:.8f} | "
            f"approximately_1={approximately_unit} | first_12={format_coordinates(vector)}"
        )
    lines.append(
        "Display note: only the first 12 coordinates are printed; all 384 coordinates "
        "are used in retrieval."
    )

    for question_number, question in enumerate(questions, 1):
        question_embedding, ranked_all = retriever.search(question)
        selected = ranked_all[:k]
        question_length = float(np.linalg.norm(question_embedding.astype(np.float64)))
        approximately_unit = bool(
            np.isclose(question_length, 1.0, atol=NUMERICAL_TOLERANCE)
        )
        lines.append("")
        lines.append(f"Question {question_number}: {question}")
        lines.append(f"question_vector_shape={question_embedding.shape}")
        lines.append(
            f"question_vector_length={question_length:.8f} | "
            f"approximately_1={approximately_unit}"
        )
        lines.append(f"question_first_12={format_coordinates(question_embedding)}")
        lines.append("All similarity scores (descending):")
        for result in ranked_all:
            lines.append(
                f"{result.chunk.chunk_id} | {result.chunk.filename} | "
                f"score={result.score:.8f}"
            )

        lines.append(f"Selected passages (top k={k}, with no relevance threshold):")
        for result in selected:
            lines.append(f"ID: {result.chunk.chunk_id}")
            lines.append(f"Filename: {result.chunk.filename}")
            lines.append(f"Text: {result.chunk.text}")

        if question_number == 1:
            top = ranked_all[0]
            top_vector = documents[top.document_index]
            coordinate_products = question_embedding.astype(np.float64) * top_vector.astype(
                np.float64
            )
            first_subtotal = float(np.sum(coordinate_products[:DISPLAYED_COORDINATES]))
            remaining_subtotal = float(np.sum(coordinate_products[DISPLAYED_COORDINATES:]))
            full_similarity = float(np.sum(coordinate_products))
            document_length = float(np.linalg.norm(top_vector.astype(np.float64)))
            direct_cosine = full_similarity / (question_length * document_length)
            difference = abs(top.score - direct_cosine)
            agrees = bool(difference <= NUMERICAL_TOLERANCE)

            lines.append("Q1 top-passage coordinate-product inspection:")
            lines.append(f"top_id={top.chunk.chunk_id}")
            lines.append(
                "first_12_coordinate_products="
                + "["
                + ", ".join(
                    f"{float(value):.8f}"
                    for value in coordinate_products[:DISPLAYED_COORDINATES]
                )
                + "]"
            )
            lines.append(f"first_12_subtotal={first_subtotal:.8f}")
            lines.append(f"remaining_372_subtotal={remaining_subtotal:.8f}")
            lines.append(f"full_similarity_dot_product={full_similarity:.8f}")
            lines.append(f"direct_cosine_from_vector_lengths={direct_cosine:.8f}")
            lines.append(f"absolute_score_cosine_difference={difference:.10f}")
            lines.append(
                f"scores_agree_within_tolerance_{NUMERICAL_TOLERANCE:g}={agrees}"
            )

    return "\n".join(lines) + "\n"


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("k must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve Cedar Lab passages using real normalized MiniLM embeddings."
    )
    parser.add_argument(
        "questions",
        nargs="+",
        help="one or more quoted questions; document vectors are reused for all of them",
    )
    parser.add_argument("-k", "--k", type=positive_integer, default=2)
    parser.add_argument(
        "--evidence",
        type=Path,
        help="also save the exact printed report to this path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = EmbeddingRetriever()
    report = render_report(retriever, args.questions, args.k)
    print(report, end="")
    if args.evidence is not None:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

