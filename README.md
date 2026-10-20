# CEDAR-RAG

A small teaching project that retrieves Cedar Lab passages with exact keyword
overlap. Despite the project name, this version does not generate answers and
does not use semantic search, stemming, or synonyms. It uses only Python's
standard library.

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

