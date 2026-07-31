# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is a personal study-notes repository for learning AI/ML, following the book *Hands-On Machine Learning* (HOML) by Aurélien Géron. It contains no application code, build system, or tests — it's a mix of Markdown notes, Jupyter notebooks, and reference PDFs of book chapters.

## Current focus

Notes already exist for chapters 1-4. The active task/focus of this repo is writing notes for the next three chapters:

- Chapter 9 — Introduction to Artificial Neural Networks
- Chapter 10 — Neural Nets with PyTorch
- Chapter 11 — Training Deep Neural Networks

**These notes are the user's first and only pass at this material** — time constraints mean there's no lecture, course, or other resource backing them up. This changes the bar for chapters 9-11 relative to the earlier ones:

- **Cover everything.** Don't skip sections of the chapter because they seem minor — if the book explains it, the notes explain it.
- **Jargon-free.** Every technical term gets defined in plain language the first time it's used, even if that means being more verbose than the chapter 1-4 notes.
- **Descriptive, not just terse bullets.** The reader should finish a section with no open questions about *why* something works, not just *what* it's called. Prefer a full explanatory sentence or two over a clipped fragment when the concept isn't obvious.
- This is a deliberate departure from the tighter shorthand used in chapters 1-4 (e.g., "Feed data → Train model → Solution"). Don't retroactively rewrite the old notes to match — only new chapters (9+) follow this fuller style.

Still follow the formatting conventions described below (bold/italics, bullet structure, formula blocks, closing Q&A section) — the shift is in depth and explanation, not in format.

## Structure

- `Notes/` — Markdown notes, one file per chapter: `<Chapter Topic> - HOML<N>.md`
- `Notebooks/` — Aurélien Géron's official Jupyter notebooks (code exercises), one per chapter: `0<N>_<snake_case_topic>.ipynb`
- `PDFs/` — Source PDFs of individual book chapters: `HOML_<N>.pdf`. These are the primary source when writing notes — the notebooks contain code but not the chapter's explanatory prose. Currently only chapters 9-11 have PDFs in the repo.

## Working in this repo

- There is no build, lint, or test tooling — changes are notes and notebook edits, not software changes.
- Notebooks use a standard `python3` kernel but there is no `requirements.txt`/environment file in the repo; assume dependencies (numpy, pandas, scikit-learn, matplotlib, pytorch, etc.) are installed in whatever environment the user runs Jupyter with.
- Markdown notes use bullet-heavy shorthand with **bold** for key terms and *italics* for defined jargon (e.g., *data drift*, *catastrophic forgetting*). For chapters 1-4, match the existing terse style when editing. For chapters 9+, follow the fuller, more explanatory style described above instead.
- Each `.md` note file ends with a "Questions and Answers" section answering the book's end-of-chapter review questions — keep this section when editing a chapter's notes, and make answers equally self-contained/descriptive for chapters 9+.
