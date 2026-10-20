"""Tests for the Cedar Lab keyword retriever."""

import unittest

from retrieve import content_words, load_chunks, retrieve


class RetrieverTests(unittest.TestCase):
    def test_loads_six_chunks_with_separate_metadata(self) -> None:
        chunks = load_chunks()
        self.assertEqual(6, len(chunks))
        self.assertEqual(("A1", "access.txt"), (chunks[0].chunk_id, chunks[0].filename))
        self.assertFalse(chunks[0].text.startswith("[A1]"))

    def test_removes_exact_stopwords_without_stemming(self) -> None:
        self.assertEqual({"operate", "operating"}, content_words("May I operate or be operating?"))

    def test_q1_ranking_and_selection(self) -> None:
        ranked, selected = retrieve(
            "May a trained graduate student operate the laser alone at 19:00?"
        )
        self.assertEqual(
            [("A1", 3), ("A2", 2), ("B1", 1), ("T1", 1), ("T2", 1), ("B2", 0)],
            [(result.chunk.chunk_id, result.score) for result in ranked],
        )
        self.assertEqual(["A1", "A2"], [result.chunk.chunk_id for result in selected])

    def test_no_positive_scores(self) -> None:
        _, selected = retrieve("zebra xylophone")
        self.assertEqual([], selected)


if __name__ == "__main__":
    unittest.main()

