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
Repository/
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
│   ├── run_cnn_lstm.py     # End-to-end CNN-LSTM pipeline (CLI)
│   └── run_baselines.py    # Baseline-only pipeline (CLI)
├── tests/                  # Unit tests and property-based tests (hypothesis)
├── artifacts/
│   ├── tokeniser.json      # Saved tokeniser vocabulary (generated)
│   └── checkpoints/        # Model checkpoints (generated)
├── results/
│   └── metrics.json        # Evaluation results (generated)
└── requirements.txt
```

## Dataset format

The dataset must be a CSV or JSON file with at least two columns/fields:

**CSV:**
```csv
text,author
"Hello world, this is a post.",alice
"Another message here.",bob
```

**JSON** (list of records):
```json
[
  {"text": "Hello world, this is a post.", "author": "alice"},
  {"text": "Another message here.", "author": "bob"}
]
```

Each author must have at least 10 samples (configurable minimum threshold).

## Setup

```bash
cd Repository
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Running experiments

All commands should be run from inside the `Repository/` directory.

### Train and evaluate the CNN-LSTM model

```bash
python -m experiments.run_cnn_lstm --dataset path/to/dataset.csv
```

This runs the full pipeline: preprocessing → BPE tokenisation → CNN-LSTM training → evaluation → baseline comparison. Results are saved to `results/metrics.json`.

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | Path to CSV or JSON dataset |
| `--seed` | `42` | Random seed for reproducibility |
| `--epochs` | `50` | Maximum training epochs |
| `--patience` | `5` | Early-stopping patience |
| `--batch-size` | `64` | Training batch size |
| `--lr` | `0.001` | Adam learning rate |
| `--vocab-size` | `10000` | BPE vocabulary size |
| `--embed-dim` | `128` | Embedding dimension |
| `--num-filters` | `128` | CNN filters per kernel size |
| `--lstm-hidden` | `256` | LSTM hidden state size |
| `--lstm-layers` | `2` | Number of stacked LSTM layers |
| `--dropout` | `0.5` | Dropout rate |
| `--max-seq-len` | `256` | Maximum token sequence length |

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
python -m experiments.run_baselines --dataset path/to/dataset.csv
```

Trains SVM and Logistic Regression classifiers on BoW, TF-IDF, character n-gram, and word n-gram features.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | Path to CSV or JSON dataset |
| `--seed` | `42` | Random seed |
| `--output` | `results/metrics.json` | Output path for metrics JSON |

## Running tests

```bash
# From the workspace root (one level above Repository/)
python -m pytest Repository/tests/ -v
```

The test suite includes unit tests and property-based tests (via `hypothesis`) covering all components. 133 tests total.

To run a specific test file:
```bash
python -m pytest Repository/tests/test_pipeline.py -v
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
