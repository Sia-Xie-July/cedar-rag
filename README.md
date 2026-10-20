# CEDAR-RAG

A small teaching project comparing exact keyword overlap with real MiniLM
embeddings, followed by a manual handoff for cited answers and claim audits.
Partners: **Tong Zhou and Sia Xie**.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export HF_HOME="$PWD/.cache/huggingface"
```

The first embedding run downloads the model and needs internet access. Later
runs can set `HF_HUB_OFFLINE=1`. Keyword retrieval needs only the standard library.
The local Tasks 3–4 run used Python 3.12.1, sentence-transformers 6.1.0,
NumPy 2.5.4, PyTorch 2.14.1, and transformers 5.20.2 on CPU.
`requirements.txt` pins the installed packages; `evidence/task3-environment.txt`
records the model revision. The original Task 2 environment remains in
`INSTALLED_VERSIONS.txt` (Python 3.13.7).

## Run it

From the project directory, run a question with the default `k=2`:

```bash
python3 retrieve.py "May a trained graduate student operate the laser alone at 19:00?"
```

Set `k` explicitly:

```bash
python3 retrieve.py "May a trained graduate student operate the laser alone at 19:00?" --k 2
```

Run the tests:

```bash
python3 -m unittest -v
```

## How it works

Each nonempty labeled line in `docs/access.txt`, `docs/training.txt`, and
`docs/booking.txt` is one chunk. The loader separates the bracketed ID and
filename from the paragraph, so only paragraph text is searched.

The question and paragraph are lowercased. The regular expression
`[a-z0-9]+` extracts alphanumeric words, and exactly the configured stopwords
are removed. Sets make query words distinct. A chunk's score is the size of
the intersection between its words and the query words. Results sort by score
descending, then chunk ID ascending. Only positive-scoring chunks are selected,
up to `k`.

Key code locations in `retrieve.py`:

- Lines 41–54 load chunks and separate ID, filename, and paragraph text.
- Lines 57–60 score one chunk and retain its matched words.
- Lines 63–69 tokenize the query, score all chunks, rank them, and select up to `k`.

Because `19:00`, `08:00`, and `22:00` are split into alphanumeric words, the
word `00` is a literal match in Q1. `student` does not match `students`, and
`operate` does not match `operating`, because there is no stemming.

## Q1 actual output

Command:

```bash
python3 retrieve.py "May a trained graduate student operate the laser alone at 19:00?" --k 2
```

Output:

```text
All chunk scores (ranked):
A1 | access.txt | score=3 | matched=00, graduate, trained
A2 | access.txt | score=2 | matched=laser, trained
B1 | booking.txt | score=1 | matched=laser
T1 | training.txt | score=1 | matched=laser
T2 | training.txt | score=1 | matched=trained
B2 | booking.txt | score=0 | matched=(none)

Selected passages (up to k=2):
ID: A1
Filename: access.txt
Text: General laboratory access hours are 08:00 to 22:00 for trained graduate students. Access outside these hours requires written approval from the lab manager.
ID: A2
Filename: access.txt
Text: The laser room requires two trained people to be present whenever the laser is operating, including during general laboratory access hours.
```

This output intentionally stops at retrieval; it does not generate an answer to
the question.

## Embedding retriever

The embedding version uses the prepared virtual environment and the real
`sentence-transformers/all-MiniLM-L6-v2` model on the CPU:

```bash
source .venv/bin/activate
python embedding_retrieve.py "May a trained graduate student operate the laser alone at 19:00?"
```

To reproduce Q1 and save the same report that is printed:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python embedding_retrieve.py \
  "May a trained graduate student operate the laser alone at 19:00?" \
  --k 2 --evidence evidence/embedding-inspection.txt
```

More than one quoted question can be supplied in one run. The six document
vectors are computed only once and reused:

```bash
.venv/bin/python embedding_retrieve.py "first question" "second question" --k 2
```

### Encoding and similarity in plain language

The model changes each paragraph into a list of 384 numbers called a vector.
The program gives the model only the paragraph text; the ID and filename stay
attached as descriptive information but are not encoded. It uses
`encode_document()` for all six paragraphs at once, producing a 6-by-384
matrix. It uses `encode_query()` for each question, producing one 384-number
vector. Both calls use the same loaded model and ask for normalized NumPy
arrays.

Normalization makes each vector's length approximately 1. To compare a
question with a paragraph, the program multiplies corresponding coordinates
and adds all 384 products. This dot product is the similarity score. For
unit-length vectors it is also cosine similarity; the report calculates cosine
from the measured lengths and checks that the values agree within `1e-6`.
Only 12 coordinates are displayed to keep the inspection readable, but all 384
are used to rank passages. Individual coordinates are deliberately not given
semantic names.

Key code locations in `embedding_retrieve.py`:

- Lines 33–42 load one CPU model and encode the six text-only passages once.
- Lines 44–50 encode each question with the same model.
- Lines 51–60 calculate dot products and rank by score, then ID.
- Lines 84–94 inspect document vectors and their lengths.
- Lines 124–156 inspect Q1's coordinate products and verify cosine agreement.

The complete unedited Q1 report is in
`evidence/embedding-inspection.txt`. Its actual top two IDs are `A2` and `T1`;
the ranking was not adjusted.

## Task 3: comparison and k experiment

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv/bin/python compare_retrieval.py
.venv/bin/python -m unittest -v
```

For a first run with no cached model, omit `HF_HUB_OFFLINE=1`.
The script loads one CPU model, encodes the six documents once, and reuses them
for Q1–Q4 and the Q1 k experiment. It imports the existing retrievers without
changing their scoring rules. IDs and filenames remain metadata. Assertions
check finite vectors, (6, 384) document and (384,) query shapes, unit lengths,
and agreement between dot products and direct cosine within 1e-6.

`evidence/retrieval-runs.txt` contains all six rankings, keyword matches,
embedding vectors, versions, and selected passages for each question.
`evidence/results-unjudged.csv` preserves measurements with the three judgment
fields blank. `results.csv` contains the subsequent evidence judgments; rerunning
the script does not overwrite that file or the saved answers. Scores and IDs
use semicolons in corresponding rank order; scores from different methods have
different scales and must not be compared numerically.

| Question | Keyword top 2 | Embedding top 2 | Required evidence coverage: keyword / embedding |
| --- | --- | --- | --- |
| Q1 | A1, A2 | A2, T1 | 2/2 / 1/2 |
| Q2 | A2, B1 | A2, T1 | 1/1 / 1/1 |
| Q3 | T1, A2 | T1, B1 | 1/1 / 1/1 |
| Q4 | A2, B1 | T1, A2 | Not applicable: answer absent |

Q1 demonstrates incomplete retrieval: A2 supports the operating condition,
but the embedding top two omit A1's time range. The keyword score for A1 is 3:
the distinct matches are `00`, `graduate`, and `trained` (repetition adds nothing).
For Q2, A2 ranks first with embedding score 0.58519333 and keyword score 2
(`laser`, `present`). For Q3, T1 ranks first with embedding score 0.74202738,
matching the intended renewal rule despite the question's paraphrase; keyword
retrieval also succeeds with `laser` and `must`. These examples do not establish
that embeddings always outperform word overlap.

Q4's nearest embedding passages are T1 (0.34772122) and A2 (0.32082713).
Neither provides a wavelength, and none of the six sources specifies one.
The embedding retriever always returns the top k, with no relevance threshold;
a positive score measures alignment rather than answerability. Both completeness
fields are therefore false, and no coverage ratio is assigned to Q4.

`evidence/k-experiment.txt` records actual Q1 reruns at k=1, 2, and 3.
They return A2; A2,T1; and A2,T1,B1, respectively. All have coverage 1/2.
T1 adds the current-training condition. B1 adds a two-hour booking limit and
calendar procedure, which do not resolve the question's solo-operation or
access-time evidence gap. A1 ranks fourth (0.37424891), so increasing k to 3
still does not recover all required evidence. This is a four-question classroom
check, not a general performance estimate.

## Task 4: evidence-only answers and audits

The script writes `evidence/q1-prompt.txt` and `evidence/q4-prompt.txt`, each
containing only the handout's answer instruction, the question, and its two
selected embedding passages. Each was submitted to a fresh ephemeral Codex
conversation from an empty directory, with no project attachment, shell tools
disabled, and browsing disabled. The event trace confirms no tool calls.
Generation is a separate manual handoff, not part of the retrieval program.
To repeat it, paste each prompt into a separate fresh conversation with no
project files and audit the resulting answer against exactly those passages.

- `evidence/q1-answer-original.md`: initial response, preserved unedited.
- `evidence/q1-prompt-revised.txt` and `evidence/q1-answer-revised.md`: one
  instruction revision and one fresh rerun with the same A2 and T1 passages.
- `evidence/q1-answer-audit.md`: claim checks and the original failure: it
  omitted the missing access-hour range. The revision explicitly acknowledges it.
- `evidence/q4-answer-original.md` and `evidence/q4-answer-audit.md`: the answer
  says wavelength is unspecified; its initial audit passes, so no revision.
- `evidence/answer-generation.json`: generation events and input/output hashes.

The final Q1 answer prohibits solo operation based on A2 and describes the
training condition from T1. It cannot establish whether 19:00 is within the
permitted access hours. Q4 cannot establish a wavelength. Missing information
is stated explicitly rather than filled in from other sources.

## Individual reflections

### Tong Zhou

In this workflow, a question is converted into an embedding, compared with
document vectors, and paired with the top two passages to generate a cited
answer. Q4 showed me that similarity does not establish answerability. The
embedding retriever selected T1 and A2, but these passages describe training
and staffing requirements, not the laser’s wavelength. The answer therefore
correctly stated that the wavelength could not be determined from the supplied
evidence. Retrieving related text is only one step; each answer still needs an
evidence check.

Next, I would test additional answerable and unanswerable questions using the
same six passages. I would compare the current prompt with one requiring
explicit identification of missing evidence, checking whether it reduces
unsupported answers without rejecting questions the passages can answer.

## Sia Xie

In this project, a question is encoded and then compared with the six passages. The retriever ranks them and then pass the top passages to the answer step. Later, the answer step cites the chunk IDs that support each claim. 

One limitation I noticed from Q1: the embedding retriever found the laser staffing rule but ranked the general access-hours rule fourth, so the top-two context could not establish whether 19:00 was permitted. This showed me that a plausible similarity ranking could also omit necessary evidence. 

For next potential steps, I would test a hybrid ranking that combines embedding similarity with keyword overlap, then compare coverage and citation accuracy on both answerable and unanswerable questions.