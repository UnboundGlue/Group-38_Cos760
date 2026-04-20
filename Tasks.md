Here's a complete step-by-step guide, organised into 3 workstreams your team can divide up.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## Step 0 — Setup (everyone does this)

```bash
cd /Users/mekhail/Documents/University/Cos760/Project/Repository
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. Each team member sets this up on their own machine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## Step 1 — Prepare a dataset (everyone needs this)

You need a CSV or JSON file with text and author columns. Each author must have at least 10 samples.

Example CSV (data/posts.csv):
```bash
text,author
"Hello world, this is a post.",alice
"Another message here.",bob
```

Put it somewhere accessible, e.g. data/posts.csv inside the repo. 
Everyone uses the same dataset file.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## Step 2 — Run tests (verify everything works)

From the Repository/ directory:

```bash
python -m pytest tests/ -v
```

This runs all 133 tests (unit + property-based via hypothesis). To run a specific module's tests:

```bash
# Individual test files
python -m pytest tests/test_dataset.py -v
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_tokeniser.py -v
python -m pytest tests/test_features.py -v
python -m pytest tests/test_model.py -v
python -m pytest tests/test_trainer.py -v
python -m pytest tests/test_evaluate.py -v
python -m pytest tests/test_explainability.py -v
python -m pytest tests/test_pipeline.py -v

# Property-based tests only
python -m pytest tests/test_properties_*.py -v
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## Step 3 — Train the CNN-LSTM model (main experiment)

```bash
python -m experiments.run_cnn_lstm --dataset data/posts.csv
```

This runs the full pipeline: preprocessing → BPE tokenisation → CNN-LSTM training → evaluation → baseline comparison.

With custom settings:
```bash
python -m experiments.run_cnn_lstm \
  --dataset data/posts.csv \
  --seed 42 \
  --epochs 30 \
  --patience 3 \
  --batch-size 64 \
  --lr 0.001 \
  --vocab-size 5000 \
  --embed-dim 128 \
  --num-filters 128 \
  --lstm-hidden 256 \
  --lstm-layers 2 \
  --dropout 0.5 \
  --max-seq-len 256
```

Outputs:
- results/metrics.json — all metrics (CNN-LSTM + baselines)
- artifacts/tokeniser.json — trained BPE vocabulary
- artifacts/checkpoints/best_model.pt — best model checkpoint (by validation macro-F1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## Step 4 — Run baselines only

```bash
python -m experiments.run_baselines --dataset data/posts.csv
```

Trains SVM and Logistic Regression on BoW, TF-IDF, character n-grams, and word n-grams. Useful for comparison without retraining the neural model.

Options: --seed 42, --output results/metrics.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## Step 5 — Check results

Results land in results/metrics.json. It contains accuracy, precision, recall, macro-F1, per-class F1, and confusion matrices for both the CNN-LSTM and all baselines.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## Suggested team split (3 people)

| Person | Responsibility | Commands |
|--------|---------------|----------|
| Person 1 — Data & Baselines | Source/prepare the dataset, run baseline experiments, compare baseline results | run_baselines, review metrics.json baselines section, run test_dataset, test_preprocessing, test_features |
| Person 2 — CNN-LSTM Training | Run the main CNN-LSTM experiment, tune hyperparameters (epochs, lr, vocab-size, etc.), analyse neural model results | run_cnn_lstm with various flags, run test_tokeniser, test_model, test_trainer, test_evaluate |
| Person 3 — Explainability & Analysis | Run SHAP/LIME analysis, interpret misclassifications, compile final comparison of CNN-LSTM vs baselines | Run test_explainability, test_pipeline, analyse metrics.json outputs, review confusion matrices |

All three should run python -m pytest tests/ -v first to confirm the codebase works on their machine before starting their individual tasks.
