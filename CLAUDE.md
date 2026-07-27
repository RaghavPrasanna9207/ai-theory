# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is a personal study-notes repository for learning AI/ML, following the book *Hands-On Machine Learning* (HOML). It contains no application code, build system, or tests — it's a mix of Markdown notes and Jupyter notebooks.

## Structure

All content lives in `Notes/`, organized by HOML chapter:

- `The ML Landscape - HOML1.md` — Markdown notes only (chapter 1 has no code exercises)
- `E2E Project - HOML2.md` + `02_end_to_end_machine_learning_project.ipynb` — chapter 2
- `Classification - HOML3.md` + `03_classification.ipynb` — chapter 3
- `Training Models - HOML4.md` + `04_training_linear_models.ipynb` — chapter 4

Each chapter pairs a `.md` file (conceptual notes, definitions, Q&A) with a numbered `.ipynb` notebook (code exercises), when the chapter has code content. Later chapters should follow this same naming pattern: `<Chapter Topic> - HOML<N>.md` and `0<N>_<snake_case_topic>.ipynb`.

## Working in this repo

- There is no build, lint, or test tooling — changes are notes and notebook edits, not software changes.
- Notebooks use a standard `python3` kernel but there is no `requirements.txt`/environment file in the repo; assume dependencies (numpy, pandas, scikit-learn, matplotlib, etc.) are installed in whatever environment the user runs Jupyter with.
- Markdown notes use bullet-heavy, terse shorthand with **bold** for key terms and *italics* for defined jargon (e.g., *data drift*, *catastrophic forgetting*). Match this style when adding to existing notes rather than writing prose paragraphs.
- Each `.md` note file ends with a "Questions and Answers" section answering the book's end-of-chapter review questions — keep this section when editing a chapter's notes.
