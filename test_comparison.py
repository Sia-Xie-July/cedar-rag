"""Check saved measurements and the evidence boundary without downloading a model."""

import csv
import hashlib
import json
from pathlib import Path
import re
import unittest

from retrieve import load_chunks, retrieve


ROOT = Path(__file__).parent
EVIDENCE = ROOT / "evidence"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ComparisonEvidenceTests(unittest.TestCase):
    def test_judgments_preserve_original_measurements(self):
        raw = read_rows(EVIDENCE / "results-unjudged.csv")
        audited = read_rows(ROOT / "results.csv")
        self.assertEqual(["Q1", "Q2", "Q3", "Q4"], [r["question_id"] for r in raw])
        self.assertEqual(len(raw), len(audited))
        judgment_fields = {"keyword_evidence_complete", "embedding_evidence_complete", "observation"}
        for before, after in zip(raw, audited):
            for field in before:
                if field in judgment_fields:
                    self.assertEqual("", before[field])
                    self.assertTrue(after[field])
                else:
                    self.assertEqual(before[field], after[field])
        self.assertEqual("absent", audited[-1]["required_ids"])
        self.assertEqual("false", audited[-1]["keyword_evidence_complete"])
        self.assertEqual("false", audited[-1]["embedding_evidence_complete"])

    def test_rankings_match_report_and_keyword_execution(self):
        report = (EVIDENCE / "retrieval-runs.txt").read_text()
        questions = re.split(r"\nQuestion \d+: ", report)[1:]
        rows = read_rows(ROOT / "results.csv")
        self.assertEqual(4, len(questions))
        for section, row in zip(questions, rows):
            question = section.splitlines()[0]
            _, selected = retrieve(question, k=2)
            self.assertEqual(row["keyword_top_ids"], ";".join(r.chunk.chunk_id for r in selected))
            self.assertEqual(row["keyword_scores"], ";".join(str(r.score) for r in selected))
            scores = re.findall(r"^([ATB]\d) \| [^\n]+ \| score=(-?\d+\.\d+)$", section, re.M)
            self.assertEqual(6, len(scores))
            self.assertEqual(row["embedding_top_ids"], ";".join(key for key, _ in scores[:2]))
            self.assertEqual(row["embedding_scores"], ";".join(value for _, value in scores[:2]))

    def test_answer_handoffs_and_traces(self):
        rows = {r["question_id"]: r for r in read_rows(ROOT / "results.csv")}
        chunks = {c.chunk_id: c.text for c in load_chunks()}
        trace = json.loads((EVIDENCE / "answer-generation.json").read_text())
        self.assertEqual(3, len(trace["runs"]))
        for run in trace["runs"]:
            prompt = (EVIDENCE / run["prompt_file"]).read_bytes()
            answer = (EVIDENCE / run["answer_file"]).read_bytes()
            self.assertEqual(run["prompt_sha256"], hashlib.sha256(prompt).hexdigest())
            self.assertEqual(run["answer_sha256"], hashlib.sha256(answer).hexdigest())
            passages = prompt.decode().split("Supplied passages:\n", 1)[1].strip().split("\n\n")
            ids = rows[run["run"][:2].upper()]["embedding_top_ids"].split(";")
            self.assertEqual([f"[{key}] {chunks[key]}" for key in ids], passages)
            self.assertEqual(0, run["tool_calls"])
            self.assertEqual("turn.completed", run["events"][-1]["type"])
            for event in run["events"]:
                if "item" in event:
                    self.assertEqual("agent_message", event["item"]["type"])


if __name__ == "__main__":
    unittest.main()
