"""A deliberately small keyword retriever for the Cedar Lab passages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


DOCS_DIRECTORY = Path(__file__).parent / "docs"
DOCUMENTS = ("access.txt", "training.txt", "booking.txt")
STOPWORDS = {
    "a", "an", "the", "is", "are", "to", "of", "in", "at", "for",
    "and", "or", "may", "can", "i", "it", "be",
}
WORD_PATTERN = re.compile(r"[a-z0-9]+")
LABELED_PARAGRAPH_PATTERN = re.compile(r"^\[([^]]+)]\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    filename: str
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: int
    matched_words: tuple[str, ...]


def content_words(text: str) -> set[str]:
    """Lowercase, extract alphanumeric words, and remove only STOPWORDS."""
    words = WORD_PATTERN.findall(text.lower())
    return {word for word in words if word not in STOPWORDS}


def load_chunks(docs_directory: Path = DOCS_DIRECTORY) -> list[Chunk]:
    """Load each labeled line, storing its label separately from its text."""
    chunks: list[Chunk] = []
    for filename in DOCUMENTS:
        path = docs_directory / filename
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            match = LABELED_PARAGRAPH_PATTERN.fullmatch(line)
            if match is None:
                raise ValueError(f"{path}:{line_number}: expected '[ID] paragraph text'")
            chunk_id, paragraph_text = match.groups()
            chunks.append(Chunk(chunk_id, filename, paragraph_text))
    return chunks


def score_chunk(chunk: Chunk, query_words: set[str]) -> ScoredChunk:
    """Count distinct non-stopword query terms shared with the chunk text."""
    matched_words = tuple(sorted(query_words & content_words(chunk.text)))
    return ScoredChunk(chunk, len(matched_words), matched_words)


def retrieve(question: str, k: int = 2) -> tuple[list[ScoredChunk], list[ScoredChunk]]:
    """Score every chunk, rank it, and select at most k positive results."""
    query_words = content_words(question)
    scored = [score_chunk(chunk, query_words) for chunk in load_chunks()]
    ranked = sorted(scored, key=lambda result: (-result.score, result.chunk.chunk_id))
    selected = [result for result in ranked if result.score > 0][:k]
    return ranked, selected


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("k must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve Cedar Lab passages by keyword overlap.")
    parser.add_argument("question", help="question to search for")
    parser.add_argument("-k", "--k", type=positive_integer, default=2,
                        help="maximum selected chunks (default: 2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranked, selected = retrieve(args.question, args.k)

    print("All chunk scores (ranked):")
    for result in ranked:
        matches = ", ".join(result.matched_words) if result.matched_words else "(none)"
        print(f"{result.chunk.chunk_id} | {result.chunk.filename} | "
              f"score={result.score} | matched={matches}")

    print()
    if not selected:
        print("No matching passages.")
        return

    print(f"Selected passages (up to k={args.k}):")
    for result in selected:
        print(f"ID: {result.chunk.chunk_id}")
        print(f"Filename: {result.chunk.filename}")
        print(f"Text: {result.chunk.text}")


if __name__ == "__main__":
    main()

