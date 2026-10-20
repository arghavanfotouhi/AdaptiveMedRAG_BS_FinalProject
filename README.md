# Adaptive-k Retrieval for Medical RAG

Evidence-sufficiency-based adaptive document retrieval for medical question answering, built on top of [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG) and its iterative variant **i-MedRAG**.

Instead of feeding every question a fixed number of retrieved documents (e.g. top-32), **Adaptive-k** retrieves documents in stages and lets an LLM-based *evidence judge* decide, after each stage, whether the evidence collected so far is already sufficient. Retrieval stops as soon as it is, so easy questions use few documents and hard questions get more.

> Bachelor's final project, Computer Engineering (Artificial Intelligence track), Iran University of Science and Technology.

---

## Highlights

- **Adaptive document count.** `k` is chosen per question from a schedule such as `(4, 8, 16, 32)`.
- **Structured evidence judge.** The judge builds a list of option-independent *information requirements*, marks each as `RESOLVED` / `PARTIALLY_RESOLVED` / `UNRESOLVED`, and links each to the supporting documents. The STOP/CONTINUE decision is computed deterministically in Python, never taken from the LLM.
- **Three stopping logics:** evidence sufficiency, stability/stagnation of the evidence state, and saturation of document novelty.
- **Works in two frameworks:** plain MedRAG (one retrieval per question) and i-MedRAG (every follow-up sub-query uses adaptive retrieval).

## Results

Evaluated on 1,000 questions from MedQA-USMLE, with MedCPT as retriever and the Textbooks corpus. The baseline is Fixed-k with k = 32.

| Framework | Method | Accuracy | Avg. documents |
|-----------|--------|---------:|---------------:|
| MedRAG    | Fixed-k (32) | 76.50% | 32 |
| MedRAG    | **Adaptive-k** | **76.90%** | **11.72** (-63.4%) |
| i-MedRAG  | Fixed-k (32) | 86.00% | - |
| i-MedRAG  | **Adaptive-k** | **86.80%** | about -72% vs. fixed |

A paired McNemar test found no significant accuracy difference between the two methods in either framework, and a Tango non-inferiority test confirmed that Adaptive-k does not lose more than 2 percentage points of accuracy. In other words, retrieval volume drops sharply without a noticeable loss in answer quality. The full analysis (k distributions, stop-reason breakdowns, success and failure cases) is in the project report.

## How it works

For a question with options, `adaptive_retrieve` performs one initial top-`max_k` retrieval (with embeddings), then walks through the `k_schedule`. At each `k` it asks the judge to re-evaluate **the full evidence set** against a single, persistent requirement list.

**The judge** (`judge_evidence_sufficiency`) returns requirements with `id`, `text`, `critical`, `status`, `supporting_documents`, `justification` and `option_leakage`, plus a `supported_option`. The prompt enforces that requirements describe *what must be known*, never *what the answer is*, so the judge cannot smuggle an answer choice into its own checklist. Supporting document numbers are validated in code: a "resolved" requirement without valid grounding is downgraded to `UNRESOLVED`. If the judge response cannot be parsed, the system never stops; it continues with more evidence.

**Stopping logics**, checked after every round:

1. **Clean STOP (evidence sufficiency).** Every critical requirement is fully `RESOLVED` and, for multiple-choice questions, the evidence supports exactly one option.
2. **Plateau STOP (stability).** No critical requirement is fully `UNRESOLVED`, the supported option is the same in two consecutive rounds, and the requirement/evidence snapshot did not change between them.
3. **Stagnation STOP.** At least one critical requirement already has grounding, and either the requirement snapshot is identical to the previous round, or the mean novelty of newly added documents did not grow by more than 5% (while no critical requirement is fully unresolved).

**Novelty** of a document is `1 - max cosine_similarity` against all documents ranked before it (`compute_document_novelty`). It is never used as a standalone sufficiency signal, only as a trend across rounds.

If no logic fires, the largest `k` in the schedule is used.

**In i-MedRAG**, each follow-up sub-query is a free-form question rather than a multiple-choice one, so it is judged with `options=None` (option discrimination is skipped). The selected snippets are then passed to the standard answer prompt, so the answer format is identical to the non-adaptive path.

## Repository structure

```
.
├── adaptive_retrieval.py   # Evidence judge, novelty computation, adaptive_retrieve() driver
├── medrag.py               # MedRAG / i-MedRAG with adaptive retrieval integrated
├── ...                     # Remaining MedRAG files (see Installation)
└── README.md
```

| File | Contents |
|------|----------|
| `adaptive_retrieval.py` | `JUDGE_SYSTEM_PROMPT`, `judge_evidence_sufficiency()`, `adaptive_retrieve()`, `compute_document_novelty()` |
| `medrag.py` | `MedRAG` class with `medrag_answer()`, `_adaptive_medrag_answer()` and `i_medrag_answer(adaptive=...)` |

## Installation

This project extends the original MedRAG code base. Besides the two files above, it relies on the following MedRAG modules, which must be present in the project (`src/utils.py`, `src/template.py`, `config.py`):

- `RetrievalSystem`, `DocExtracter` (in `src/utils.py`)
- the prompt templates (in `src/template.py`)
- `config` (in `config.py`)

Adaptive retrieval additionally needs `RetrievalSystem.retrieve_with_embeddings(question, k, rrf_k)`, which returns `(snippets, scores, ids, embeddings)`. It is supported only for **exactly one dense retriever and one corpus** (e.g. MedCPT + Textbooks), because novelty is computed from the retriever's embeddings.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt   # torch, transformers, openai>=1.0, tiktoken, tqdm, numpy, ...
```

Follow the MedRAG instructions to download or build the corpus (Textbooks) and the MedCPT index into `./corpus`.

### LLM configuration

The code talks to an OpenAI-compatible endpoint. The experiments used **ArvanCloud** with `gpt-4o-mini`
Standard OpenAI / Azure OpenAI is also supported through `OPENAI_API_KEY`, `OPENAI_API_TYPE`, `OPENAI_API_VERSION`, and `OPENAI_ENDPOINT`.

## Usage

### Adaptive retrieval with MedRAG

```python
from medrag import MedRAG
from adaptive_retrieval import adaptive_retrieve

medrag = MedRAG(
    llm_name="ArvanCloud/gpt-4o-mini",
    rag=True,
    retriever_name="MedCPT",
    corpus_name="Textbooks",
)

question = "A 45-year-old man presents with ..."
options = {"A": "...", "B": "...", "C": "...", "D": "..."}

# 1. Pick the documents adaptively
snippets, scores, selected_k, history = adaptive_retrieve(
    medrag,
    question,
    options=options,
    k_schedule=(4, 8, 16, 32),
    verbose=False,
)

# 2. Answer with the selected documents
answer, _, _ = medrag.medrag_answer(
    question,
    options=options,
    k=max(selected_k, 1),
    snippets=snippets,
)
print(selected_k, answer)
```

### Adaptive retrieval with i-MedRAG

```python
medrag = MedRAG(
    llm_name="ArvanCloud/gpt-4o-mini",
    rag=True,
    follow_up=True,                 # enables i-MedRAG
    retriever_name="MedCPT",
    corpus_name="Textbooks",
    adaptive=True,                  # adaptive retrieval for every follow-up query
    adaptive_k_schedule=(4, 8, 16, 32),
)

answer, messages = medrag.answer(question=question, options=options, k=32)

# Per-sub-query logs: selected k, judge calls, stop reason, full history
for entry in medrag.last_adaptive_log:
    print(entry["query"], entry["selected_k"], entry["judge_calls"], entry["stop_reason"])
```

You can also toggle it per call: `medrag.answer(..., adaptive=False)` runs the fixed-k baseline with the same object.

### Inspecting a run

`adaptive_retrieve` returns `(selected_snippets, selected_scores, selected_k, history)`. Each `history` item (one per evaluated `k`) contains:

`k`, `new_documents`, `document_ids`, `mean/min/max_new_novelty`, `judge_decision`, `supported_option`, `requirements`, `missing_requirements`, `stop_reason`, `reason`, `stagnated`, `stagnation_reason`.

Set `verbose=True` to print the judge's requirement table at every round.

## Limitations

- Requires a dense retriever whose embeddings are exposed (single retriever + single corpus).
- The 5% novelty-trend rule is an operational heuristic for detecting saturation, not a statistical proof that new documents are redundant.
- Evaluated on MedQA-USMLE (4-option) with one retriever, one corpus, and one LLM.

## Acknowledgments

This work builds on:

- *Benchmarking Retrieval-Augmented Generation for Medicine* (MedRAG), Xiong et al.
- *Improving Retrieval-Augmented Generation in Medicine with Iterative Follow-up Questions* (i-MedRAG), Xiong et al.
- MedCPT retriever and the MedQA-USMLE dataset.

## Author

**Arghavan Fotouhi**, supervised by Dr. Seyed Saleh Etemadi, School of Computer Engineering, Iran University of Science and Technology.
