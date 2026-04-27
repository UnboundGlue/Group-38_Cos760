# Neural Authorship Attribution with Subword Embeddings and CNN-LSTM

This project investigates **authorship attribution** on social media text using subword tokenisation (BPE/WordPiece) combined with a CNN-LSTM neural architecture. The model learns author-specific stylometric fingerprints from subword token sequences and classifies documents to their authors. Results are compared against traditional baselines (Bag-of-Words, TF-IDF, character n-grams, word n-grams), and SHAP/LIME explainability is integrated to interpret model decisions and analyse misclassifications.

## How it works

```
Raw text → Preprocessor → SubwordTokeniser (BPE/WordPiece)
                                    ↓
                          CNN (multi-scale n-gram filters)
                                    ↓
                          LSTM (sequential encoding)
                                    ↓
                          Linear classifier → author prediction
                                    ↓
                          SHAP / LIME → error analysis report
```

Baseline classifiers (SVM, Logistic Regression) are trained on the same data using traditional feature extractors for direct comparison.

## Project structure

```
(repository root)
├── data/                   # Optional: local dataset CSVs (see data/README.md)
├── src/
│   ├── dataset.py          # DatasetLoader — CSV/JSON loading, stratified splits
│   ├── preprocessing.py    # Preprocessor — URL/mention removal, whitespace normalisation
│   ├── tokeniser.py        # SubwordTokeniser — BPE/WordPiece via HuggingFace tokenizers
│   ├── features.py         # BaselineFeatureExtractor — BoW, TF-IDF, n-grams
│   ├── model.py            # CNNLSTMModel — parallel CNN + stacked LSTM
│   ├── models.py           # Shared dataclasses (ModelConfig, MetricsDict, etc.)
│   ├── trainer.py          # Trainer — training loop, early stopping, checkpointing
│   ├── evaluate.py         # evaluate() — accuracy, F1, confusion matrix
│   └── explainability.py   # ExplainabilityModule — SHAP, LIME, error analysis
├── experiments/
│   ├── run_cnn_lstm.py              # End-to-end CNN-LSTM pipeline (CLI)
│   ├── run_baselines.py             # Baseline-only pipeline (CLI)
│   ├── dry_run_cnn_lstm_synthetic.py   # Task 7 smoke: model + random token IDs only
│   ├── dry_run_cnn_lstm_real_text_stub.py  # one forward: real data + char stub ids
│   └── validate_cnn_lstm_real_stub.py  # few epochs: train/val metrics vs random (stub ids)
├── tests/                  # Unit tests and property-based tests (hypothesis)
├── artifacts/
│   ├── tokeniser.json      # Saved tokeniser vocabulary (generated)
│   └── checkpoints/        # Model checkpoints (generated)
├── results/
│   └── metrics.json        # Evaluation results (generated)
├── requirements.in         # Direct dependencies (no version pins; edit this list)
├── requirements.txt        # `-r requirements.in` (so `pip install -r requirements.txt` works)
└── requirements-locked.txt # Exact versions from `pip freeze` (optional; generated)
```

## Dataset format

The dataset must be a CSV or JSON file with at least two columns/fields for **text** and **author**. Column names are matched case-insensitively (e.g. `text` / `Text` and `author` / `Author`).

**Reference dataset (Chanchal et al.):** Use `DatasetLoader.load(path, fetch_if_missing=True)` or `ensure_authoridentification_dataset()` in `src/dataset.py` to shallow-clone the [AuthorIdentification](https://github.com/chanchalIITP/AuthorIdentification) repo into `data/AuthorIdentification/` if needed (Git on `PATH`), or run `python data/fetch_dataset.py` / experiments with `--fetch-dataset` — see `data/README.md`. Those CSVs use headers `Text` and `Author`; tweet cells are often stored as Python byte-string literals (`b'…'`), which `DatasetLoader` decodes to normal Unicode strings.

**CSV (lowercase example):**
```csv
text,author
"Hello world, this is a post.",alice
"Another message here.",bob
```

**CSV (Chanchal-style headers are also accepted):**
```csv
Text,Author
"Hello world.",61771813
```

**JSON** (list of records; keys may be `text`/`author` or `Text`/`Author`):
```json
[
  {"text": "Hello world, this is a post.", "author": "alice"},
  {"text": "Another message here.", "author": "bob"}
]
```

Each author must have at least 10 samples for the default stratified split (configurable `min_samples` threshold).

## Setup

Python **3.10+** is recommended (3.11+ works well with current `torch` wheels).

### 1. Create and activate a virtual environment

**Windows (PowerShell), from the repository root:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

The `.venv` directory is listed in `.gitignore` so it is not committed.

### 2. Install dependencies

Direct dependencies are listed in **`requirements.in`** without pinned versions. Install them into the active venv:

```bash
pip install -r requirements.in
```

Alternatively, `requirements.txt` includes `requirements.in`, so this is equivalent:

```bash
pip install -r requirements.txt
```

This matches **Task 1** in the project plan: `torch`, `tokenizers`, `scikit-learn`, `numpy`, `pandas`, `shap`, `lime`, `hypothesis`, and `pytest`.

### 3. Locking versions with `pip freeze` (optional but recommended for reproducibility)

**Do not hand-edit version numbers** in a large requirements file. Instead:

1. After `pip install -r requirements.in` (and once tests or training run successfully), capture the **exact** environment:

   ```bash
   pip freeze > requirements-locked.txt
   ```

2. **When to run `pip freeze`:**
   - After you add or remove a package in `requirements.in` and confirm everything still works.
   - Before a milestone, release, or demo so teammates and CI can reinstall the same stack.
   - When someone reports “works on my machine” issues; compare or refresh the lock.

3. **To reproduce a locked environment** (e.g. on another machine or in CI):

   ```bash
   pip install -r requirements-locked.txt
   ```

The repository may include an up-to-date **`requirements-locked.txt`** generated this way; treat it as **machine-generated**. If it is missing or outdated, create it locally with the steps above.

**Note:** `pip freeze` lists every package in the venv, including transitive dependencies. Keep the venv **only** for this project when generating the lock so the file stays minimal and relevant.

## Running experiments

All commands below assume your shell’s **current directory is the repository root** (the folder that contains `src/`, `tests/`, and `experiments/`).

### Train and evaluate the CNN-LSTM model

**`--dataset` is optional:** it defaults to the Chanchal **`200_tweets_per_user.csv`** slice (`DEFAULT_CHANCHAL_200_CSV` in `src/dataset.py`) so each author has more training text. Pass `--dataset` only to use another CSV/JSON (e.g. the smaller `50_tweets_per_user.csv` for fast experiments). `--fetch-dataset` clones the AuthorIdentification repo if the file is missing.

**GPU:** Training picks **batch size** and **DataLoader workers** from your hardware (`--batch-size 0`, `--num-workers -1` by default; see `src/training_hardware.py`). Install a **CUDA-enabled** PyTorch from the [Get Started](https://pytorch.org/get-started/locally/) page (not the default CPU-only `pip install torch`). **NVIDIA RTX 50-series (Blackwell, e.g. RTX 5070)** needs a build with **CUDA 12.8** in the wheel name (e.g. `+cu128`); older `+cu126` wheels do not include **sm_120** kernels. Example for this repo’s venv:

```bash
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then check: `python -c "import torch; print(torch.__version__, torch.cuda.is_available()); x=torch.randn(2,2,device='cuda'); print(x.device)"` — you should see `cuda:0` and no “not compatible with sm_120” warning.

```bash
python -m experiments.run_cnn_lstm --fetch-dataset
```

Smaller slice (faster, harder learning):

```bash
python -m experiments.run_cnn_lstm --fetch-dataset \
  --dataset data/AuthorIdentification/Dataset/Dataset_with_varying_number_of_tweets/50_tweets_per_user.csv
```

This runs the full pipeline: preprocessing → BPE tokenisation → CNN-LSTM training → evaluation → baseline comparison. Results are saved to `results/metrics.json`. Defaults favour accuracy: 200-tweets data, larger model (256/384), **max sequence length 384**, **balanced class weights** in the loss, label smoothing, weight decay, and **ReduceLROnPlateau** (default) on validation macro-F1. Optional: ``--lr-schedule cosine_restarts`` (warm restarts) to experiment; it can be slower in early epochs and is not always better.

To **validate `CNNLSTMModel` only** on random token IDs (no dataset, no `SubwordTokeniser` — useful before Task 4 is integrated):

```bash
python -m experiments.dry_run_cnn_lstm_synthetic
```

To run **load → preprocess → split** on a **real CSV** and one forward on **`CNNLSTMModel`** using a **temporary character-based id stub** (still **not** BPE; Task 4 not required):

```bash
python -m experiments.dry_run_cnn_lstm_real_text_stub --dataset data/AuthorIdentification/Dataset/.../50_tweets_per_user.csv --fetch-dataset
```

To **train for a few epochs** on real stub-encoded text and print **val accuracy / macro-F1 vs. random chance** (stronger check that CNN-LSTM + optimization behave):

```bash
python -m experiments.validate_cnn_lstm_real_stub --epochs 5 --fetch-dataset
# Faster CPU smoke:  python -m experiments.validate_cnn_lstm_real_stub --max-train-samples 1000 --epochs 3
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | Chanchal `200_tweets…` (see `DEFAULT_CHANCHAL_200_CSV`) | Path to CSV or JSON; omit for the default file |
| `--seed` | `42` | Random seed for reproducibility |
| `--epochs` | `100` | Maximum training epochs |
| `--patience` | `16` | Early-stopping patience |
| `--batch-size` | `0` (auto) | Training batch size; `0` picks from GPU memory (or 32 on CPU) |
| `--num-workers` | `-1` (auto) | DataLoader workers; `-1` uses up to 4 on Windows / 8 elsewhere (0 = main process only) |
| `--lr` | `0.0008` | Adam learning rate |
| `--vocab-size` | `10000` | BPE vocabulary size |
| `--embed-dim` | `256` | Embedding dimension |
| `--num-filters` | `256` | CNN filters per kernel size |
| `--lstm-hidden` | `384` | LSTM hidden state size |
| `--lstm-layers` | `2` | Number of stacked LSTM layers |
| `--dropout` | `0.35` | Dropout rate |
| `--max-seq-len` | `384` | Maximum token sequence length |
| `--weight-decay` | `1e-4` | Adam L2 weight decay |
| `--label-smoothing` | `0.05` | Label smoothing (use `0` to turn off) |
| `--lr-schedule` | `plateau` | `plateau` (default, shrink LR when val F1 stalls), `cosine_restarts` (opt-in), or `none` |
| `--cosine-t0` | `8` | Epochs per cycle before first restart (only for `cosine_restarts`) |
| `--class-weight` | `balanced` | `balanced` = inverse-frequency loss weights; use `none` for uniform |

**Example with custom settings:**
```bash
python -m experiments.run_cnn_lstm \
  --dataset data/posts.csv \
  --seed 0 \
  --epochs 30 \
  --patience 3 \
  --vocab-size 5000
```

**Outputs:**
- `results/metrics.json` — CNN-LSTM and baseline metrics
- `artifacts/tokeniser.json` — trained BPE tokeniser
- `artifacts/checkpoints/best_model.pt` — best model checkpoint (by validation macro-F1)

### Run baselines only

```bash
python -m experiments.run_baselines --fetch-dataset
```

Trains SVM and Logistic Regression classifiers on BoW, TF-IDF, character n-gram, and word n-gram features (same default `--dataset` as `run_cnn_lstm`).

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | Chanchal `200_tweets…` (see `DEFAULT_CHANCHAL_200_CSV`) | Path to CSV or JSON; omit for the default file |
| `--seed` | `42` | Random seed |
| `--output` | `results/metrics.json` | Output path for metrics JSON |

## Running tests

```bash
python -m pytest tests/ -v
```

The test suite includes unit tests and property-based tests (via `hypothesis`) covering all components. 133 tests total.

To run a specific test file:
```bash
python -m pytest tests/test_pipeline.py -v
```

## Outputs explained

`results/metrics.json` contains:
```json
{
  "cnn_lstm": {
    "accuracy": 0.87,
    "precision_macro": 0.86,
    "recall_macro": 0.85,
    "f1_macro": 0.85,
    "f1_per_class": {"0": 0.91, "1": 0.79, ...},
    "confusion_matrix": [...]
  },
  "baselines": {
    "bow":       {"svm": {...}, "logreg": {...}},
    "tfidf":     {"svm": {...}, "logreg": {...}},
    "char_ngram":{"svm": {...}, "logreg": {...}},
    "word_ngram":{"svm": {...}, "logreg": {...}}
  }
}
```
