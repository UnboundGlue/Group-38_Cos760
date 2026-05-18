# COS760 Final Report — Task Assignment & TODO

**Deadline: 27 May 2026 (Report + Code) | 15 June 2026 (Video) | 17 June 2026 (Live Presentation)**

---

## Rubric Weights

| Criterion | Weight |
|---|---|
| Problem Statement | 5 |
| Introduction | 5 |
| Literature Survey | 10 |
| Approach/Methodology/Data | 10 |
| Experiments & Results | 10 |
| Conclusion & Discussion | 10 |
| Writing & References | 10 |
| Presentation & Q&A | 30 |
| Project Repo | 10 |

---

## Task Assignments
Take alook in the Docs/Final-Project/COS760 Project Rubric v 1.1.xlsx - Sheet1-2.pdf to go so how much detail you must put into you assigned sections.

### "The Writer / Framing"

- [ ] (Mekhail) Write the **Problem Statement** (research questions, sub-questions, connection to topic)
- [ ] (Mekhail) Write the **Introduction** (motivation, background, guide to rest of document, happens third last in the list of things to do)
- [ ] (Mekhail) Write the **Conclusion & Discussion** (summarise findings, limitations, future work, happens second last in the list of things to do)
- [ ] (Mekhail) Write the **Abstract** (max 200 words — write this last)
- [ ] Ensure narrative flow and consistency across all sections

### "The Researcher / Literature"

- [ ] (Mekhail) Write the **Literature Survey** (prior work on authorship attribution, stylometry, subword tokenisation, CNN/LSTM for text classification, explainability, identify gaps. This was mostly done in the Proposal)
- [ ] (Mekhail)Compile all **References** in `custom.bib` (proper BibTeX entries, DOIs where possible)

### "The Engineer / Experiments"

#### Approach/Methodology/Data section

- [ ] (Ruan) Describe the **dataset** (Chanchal AuthorIdentification, number of authors, tweets per author, train/val/test split ratios)
- [ ] (Ruan) Describe **preprocessing** (URL/mention removal, whitespace normalisation — `src/preprocessing.py`)
- [ ] (Mekhail) Describe **subword tokenisation** (BPE/WordPiece via HuggingFace tokenizers, vocab size — `src/tokeniser.py`)
- [ ] (Ruan) Describe **CNN-LSTM architecture** (embedding layer, parallel CNN filters, LSTM layers, dropout, linear classifier — `src/model.py`)
- [ ] (Thomas) Describe **baseline feature extractors** (BoW, TF-IDF, char n-grams, word n-grams — `src/features.py`)
- [ ] (Thomas) Describe **baseline classifiers** (SVM, Logistic Regression)
- [ ] (Ruan) List key **hyperparameters** (lr, batch size, epochs, patience, embed dim, num filters, LSTM hidden, dropout, max seq len, label smoothing, weight decay)
- [ ] Create **architecture diagram** (pipeline figure: text → preprocess → tokenise → CNN → LSTM → classifier)

#### Experiments & Results section 

- [ ] Run final experiments: `python -m experiments.run_cnn_lstm --fetch-dataset`
- [ ] Run baselines: `python -m experiments.run_baselines --fetch-dataset`
- [ ] (Thomas) Report **CNN-LSTM metrics** (accuracy, precision, recall, macro-F1)
- [ ] (Thomas) Report **baseline metrics** (all 4 feature types × 2 classifiers)
- [ ] (Thomas) Create **comparison table** (CNN-LSTM vs all baselines)
- [ ] (Thomas) Create **confusion matrix** figure (CNN-LSTM)
- [ ] (Thomas) Report **per-class F1** scores (highlight best/worst authors)
- [ ] (Mekhail) Describe **SHAP/LIME explainability** results (what subword patterns distinguish authors — `src/explainability.py`)
- [ ] (Mekhail) Discuss **misclassification analysis** (which authors get confused, why)
- [ ] (Thomas) Create **training curve** figure (loss/F1 over epochs, if available)

#### Figures & Tables

- [ ] (Ruan) Architecture diagram
- [ ] (Ruan) Dataset statistics table (authors, samples, split sizes)
- [ ] (Thomas) Results comparison table
- [ ] (Thomas) Confusion matrix heatmap
- [ ] (Mekhail) SHAP/LIME example visualisation (if space permits)

#### Project Repo (Ruan)

- [ ] Ensure README.md is complete and accurate
- [ ] Verify `requirements.in` / `requirements-locked.txt` are up to date
- [ ] Confirm experiments are reproducible from a clean clone
- [ ] Clean up any unused files or notebooks
- [ ] Add any missing code comments/docstrings

### Shared (All 3 of us)

- [ ] **Presentation slides** — divide the 5-min talk (~1.5 min each)
- [ ] **Recorded video** (max 5 min) — due 15 June
- [ ] **Live presentation** — 17 June (all must attend Q&A)
- [ ] Review each other's sections before merging
- [ ] Complete **Peer + Self Evaluation** form (released after submission)
- [ ] Help prepare **presentation slides**
- [ ] Final **proofread** of entire document (grammar, spelling, ACL formatting compliance)

---

## Timeline

| Date | Milestone |
|---|---|
| **27 May** | **Submit report (PDF) + code (.zip)** |
| **15 June** | **Submit recorded video** |
| **17 June** | **Live presentation (08:00–16:00 online)** |

---

## Submission Checklist

- [ ] Report is **max 6 pages** (including references)
- [ ] Uses **ACL LaTeX template** (switch to `\usepackage[final]{acl}`)
- [ ] Group name + all member names in the document
- [ ] PDF named: `Group<number>_u<student_number>.pdf`
- [ ] Submit report PDF via **ClickUp**
- [ ] Code .zip with README.md submitted via **Google Form**
- [ ] Video (max 5 min) submitted via **Google Form**

---

## LaTeX Quick Reference

### Compile to PDF (local)

```bash
cd Docs/Final-Project/acl-style-files-master
pdflatex acl_latex.tex
bibtex acl_latex
pdflatex acl_latex.tex
pdflatex acl_latex.tex
```

Or with `latexmk`:
```bash
latexmk -pdf acl_latex.tex
```

### Overleaf (recommended for collaboration)

Template: https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj

### Key edits to `acl_latex.tex`

1. Change `\usepackage[review]{acl}` → `\usepackage[final]{acl}`
2. Set `\title{Neural Authorship Attribution with Subword Embeddings and CNN-LSTM}`
3. Set all 3 authors with `\author{Name1 \\ Affiliation \And Name2 \\ Affiliation \And Name3 \\ Affiliation}`
4. Replace body with: Abstract, Introduction, Literature Survey, Methodology, Experiments & Results, Conclusion
5. Add references to `custom.bib`, cite with `\citet{key}` or `\citep{key}`
