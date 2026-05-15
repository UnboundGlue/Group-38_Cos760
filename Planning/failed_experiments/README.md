# Failed / retired experiments

Code and notes for approaches we **do not** ship in the main pipeline.

## Optuna hyperparameter search (`tune_cnn_lstm.py`)

**Status:** Retired from `experiments/` (no longer a supported entry point).

**What it did:** Spawned `python -m experiments.run_cnn_lstm` as a subprocess per Optuna trial, optimising validation macro-F1 over vocabulary size, sequence length, LR, dropout, weight decay, label smoothing, and LR schedule. It added optional metadata JSON into `training.json` for reproducibility.

**Why we dropped it from the product path**

1. **Cost vs benefit** — On this task and split, the pipeline was already near a stable optimum with hand-tuned defaults and a seeded first trial. Sequential full retrains (tokenizer + CNN–LSTM from scratch each time) made wall time **many times** a single production run, without meaningfully beating the baseline F1.
2. **Metric mismatch** — Tuning used single-seed headline metrics; ensemble runs (multiple training seeds + vote) behave differently, so “improvement” during search did not always translate to the headline numbers we care about for reporting.
3. **Platform friction** — Windows + subprocess + matplotlib live plots + occasional non-zero exit codes after successful training added noise and confusion without improving science outcomes.
4. **Simpler alternatives** — Manual or small grid sweeps via `run_cnn_lstm` flags, FastText embedding init, ensembles, and patience/epoch adjustments address the same goals with less moving parts.

**If you want to revive it:** The script lives here with its previous dependencies (`optuna`). You would need to reinstall `optuna`, restore any CLI hooks you want in `run_cnn_lstm.py` / `run_artifact.py` (we removed `--skip-baselines`, `--experiment-metadata-json`, and `experiment_metadata` in `training.json`), and maintain the tuner yourself.
