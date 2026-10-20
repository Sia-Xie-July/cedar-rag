# Evidence provenance

Tasks 3–4 were run locally on 2026-10-20 for Tong Zhou and Sia Xie.
No embedding scores or vectors were fabricated or borrowed for these tasks.

- `embedding-inspection.txt` is the original Task 2 report inherited from the
  team's repository at commit `ecde9ec`; its Python version is 3.13.7.
- `embedding-inspection-task3.txt` is a new independent local CPU Q1 inspection;
  the Tasks 3–4 Python version is 3.12.1. The old report was preserved.
- `retrieval-runs.txt` is actual output produced by `compare_retrieval.py` for
  all four questions; its embedding section includes vectors and all six scores.
- `results-unjudged.csv` preserves the rankings before evidence judgments.
  The root `results.csv` adds subsequent Codex-assisted judgments for student review.
- `k-experiment.txt` records the actual Q1 k=1, 2, and 3 reruns. The interpretation
  and coverage comparison are in the root README.
- The Q1/Q4 prompt files contain only the selected evidence, never the full corpus.
  The answer files are actual, unedited outputs from separate fresh ephemeral
  Codex CLI conversations. Q1 has one preserved failure and one revision;
  Q4 passed on its first attempt. The CLI used its default model selection.
- `answer-generation.json` retains generation events, hashes, working-directory
  labels, and the restrictions used. It shows that all answer conversations
  completed with no tool calls. `task3-environment.txt` records the local runtime
  and downloaded embedding-model revision.

Reruns may have small numeric differences across hardware or library versions.
Review these audits yourselves and write your own individual reflections before submission.
