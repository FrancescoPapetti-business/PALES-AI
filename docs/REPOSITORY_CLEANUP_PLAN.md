# Repository Cleanup Plan

This document tracks the work needed to turn the original thesis repository into a clean portfolio repository.

## Completed in `portfolio-cleanup`

- Added a professional English `README.md`
- Presented the project as **ClassyFarm RAG Assistant** while preserving the original PALES-AI codename
- Added `.env.example` for reproducible local setup
- Updated `.gitignore` so real secrets stay private while `.env.example` is versioned
- Added architecture documentation in `docs/ARCHITECTURE.md`

## Recommended Next Steps

### 1. Repository Naming

Rename the GitHub repository from `PALES-AI` to something clearer for recruiters, for example:

- `classyfarm-rag-assistant`
- `classyfarm-rag-chatbot`
- `classyfarm-support-assistant`

The README already uses **ClassyFarm RAG Assistant** as the public-facing project name.

### 2. Root Folder Cleanup

Several experimental scripts currently live in the repository root. A cleaner structure would move them into dedicated folders:

- `tests/` for repeatable tests
- `experiments/` for thesis experiments and debug scripts
- `scripts/` for reusable operational scripts
- `docs/` for reports and written explanations

This should be done in a local Git workspace so imports and test paths can be verified safely.

### 3. Evaluation Artifacts

The repository includes many validation outputs. They are useful because they show the evaluation process, but the final portfolio version should keep only the most representative files in the main branch.

Recommended structure:

```text
results/
├── ragas_summary.csv
├── simple_rag_comparison.json
└── README.md
```

Historical CSV reports can be archived or moved to a release asset if they are not needed for day-to-day review.

### 4. Source Data Policy

The included PDFs and FAQ files are public ClassyFarm resources and are kept to make the project reproducible. Generated assets such as SQLite databases, FAISS indexes, local datasets, logs, and caches should remain ignored because they can be rebuilt.

### 5. Local Verification

Once Git is available locally, run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.rebuild_knowledge_base
uvicorn backend.main:app --reload
```

Then in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Finally verify:

- `GET http://localhost:8000/health`
- `POST http://localhost:8000/api/v1/chat`
- `GET http://localhost:3000`

### 6. License Decision

Before presenting the repository as reusable open source, choose and add a license. If the repository is only for portfolio viewing, explicitly state that reuse is not licensed yet.
