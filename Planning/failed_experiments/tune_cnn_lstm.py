"""ARCHIVED — see ``Planning/failed_experiments/README.md``. Not part of the supported pipeline.

Optuna hyperparameter search for the CNN-LSTM pipeline (subprocess trials).

Each trial shells out to::

    python -m experiments.run_cnn_lstm ...

**Objective:** maximise validation macro-F1 reported in metrics JSON
(``cnn_lstm_splits.validation.f1_macro``), matching the headline val metric after
loading the best checkpoint.

Trials pass ``--skip-baselines`` and ``--no-promote-best`` for speed and so tuning
does not overwrite ``artifacts/best_model_bundle``. Use ``--experiment-metadata-json``
so each run's ``training.json`` records Optuna trial ids; after the study completes,
the **best** trial's ``training.json`` is patched with ``optuna_study_best``.

Defaults run **full training per trial** (``epochs=100`` / ``patience=24``, matching
``run_cnn_lstm`` defaults) so reported best params reflect realistic training. Lower
``--epochs`` only when you explicitly want a quick exploratory study.

Example::

    python -m experiments.tune_cnn_lstm --fetch-dataset --n-trials 20 --split-seed 42 --seed 42

Outputs under ``artifacts/tuning/<study_name>_<UTC>/``: SQLite study DB,
``trial_<n>_metrics.json`` copies, ``study_best_summary.json`` (best params,
seeds, reproduce CLI).

Final reporting run (baselines + promote)::

    # Command is printed at end of tuning and stored in study_best_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import FrozenTrial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _safe_filename_tag(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE).strip("_")
    return (s[:48] if s else "study").lower()


def _trial_run_label(study_name: str, trial_number: int) -> str:
    return f"optuna_{_safe_filename_tag(study_name)}_trial{trial_number}"


def _latest_matching_run_dir(repo: Path, run_label_prefix: str) -> Path | None:
    pattern = str(repo / "artifacts" / "runs" / f"{run_label_prefix}_*")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        return None
    return Path(matches[0])


def _read_validation_f1(metrics_path: Path) -> float:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    try:
        return float(data["cnn_lstm_splits"]["validation"]["f1_macro"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot read validation F1 from {metrics_path}") from exc


def _patch_best_training_json(
    *,
    training_json: Path,
    study: optuna.Study,
    best_trial: FrozenTrial,
    study_out_dir: Path,
    reproduce_argv: list[str],
) -> None:
    payload = json.loads(training_json.read_text(encoding="utf-8"))
    em = payload.setdefault("experiment_metadata", {})
    em["optuna_study_best"] = {
        "study_name": study.study_name,
        "study_direction": study.direction.name if study.direction else None,
        "storage_parent": str(study_out_dir.resolve()),
        "n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "best_trial_number": best_trial.number,
        "best_validation_f1_macro_objective": best_trial.value,
        "best_params": best_trial.params,
        "best_user_attrs": dict(best_trial.user_attrs),
        "reproduce_cli_argv": reproduce_argv,
        "written_iso_utc": datetime.now(timezone.utc).isoformat(),
    }
    training_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Patched best trial training.json: %s", training_json)


def _build_subprocess_argv(
    *,
    repo: Path,
    trial: optuna.Trial,
    args: argparse.Namespace,
    study_out_dir: Path,
    experiment_md_path: Path,
    save_run_label: str,
    metrics_out: Path,
) -> list[str]:
    vocab_size = trial.suggest_categorical(
        "vocab_size", [8192, 10_000, 12_288]
    )
    max_seq_len = trial.suggest_categorical("max_seq_len", [256, 384])
    lr = trial.suggest_float("lr", 3e-4, 2e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.20, 0.45)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 3e-3, log=True)
    label_smoothing = trial.suggest_categorical(
        "label_smoothing", [0.0, 0.02, 0.05]
    )
    lr_schedule = trial.suggest_categorical(
        "lr_schedule", ["plateau", "cosine_restarts"]
    )

    argv = [
        sys.executable,
        "-m",
        "experiments.run_cnn_lstm",
        "--dataset",
        args.dataset,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--split-seed",
        str(args.split_seed),
        "--batch-size",
        str(args.batch_size),
        "--vocab-size",
        str(vocab_size),
        "--max-seq-len",
        str(max_seq_len),
        "--embed-dim",
        str(args.embed_dim),
        "--lr",
        str(lr),
        "--dropout",
        str(dropout),
        "--weight-decay",
        str(weight_decay),
        "--label-smoothing",
        str(label_smoothing),
        "--lr-schedule",
        lr_schedule,
        "--metrics-out",
        str(metrics_out.resolve()),
        "--save-run",
        save_run_label,
        "--no-promote-best",
        "--skip-baselines",
        "--experiment-metadata-json",
        str(experiment_md_path.resolve()),
    ]

    if args.no_live_plot:
        argv.append("--no-live-plot")

    if not args.allow_compile:
        argv.append("--no-compile")

    if lr_schedule == "cosine_restarts":
        cosine_t0 = trial.suggest_int("cosine_t0", 4, 16)
        argv.extend(["--cosine-t0", str(cosine_t0)])

    if args.fetch_dataset:
        argv.append("--fetch-dataset")

    if args.fasttext_vec:
        argv.extend(["--fasttext-vec", args.fasttext_vec])
        argv.extend(["--fasttext-limit", str(args.fasttext_limit)])
        if args.freeze_pretrained:
            argv.append("--freeze-pretrained")

    if args.no_amp:
        argv.append("--no-amp")
    if args.no_compile:
        argv.append("--no-compile")

    return argv


def _argv_for_best_params_cli(
    *,
    params: dict[str, Any],
    args: argparse.Namespace,
    metrics_out: Path | None,
    save_run_label: str | None,
    skip_baselines: bool,
    promote_best: bool,
) -> list[str]:
    """Rebuild CLI for a manual / post-study confirmation run."""
    lr_schedule = params["lr_schedule"]
    argv = [
        sys.executable,
        "-m",
        "experiments.run_cnn_lstm",
        "--dataset",
        args.dataset,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--split-seed",
        str(args.split_seed),
        "--batch-size",
        str(args.batch_size),
        "--vocab-size",
        str(int(params["vocab_size"])),
        "--max-seq-len",
        str(int(params["max_seq_len"])),
        "--embed-dim",
        str(int(args.embed_dim)),
        "--lr",
        str(float(params["lr"])),
        "--dropout",
        str(float(params["dropout"])),
        "--weight-decay",
        str(float(params["weight_decay"])),
        "--label-smoothing",
        str(float(params["label_smoothing"])),
        "--lr-schedule",
        str(lr_schedule),
    ]
    if lr_schedule == "cosine_restarts":
        argv.extend(["--cosine-t0", str(int(params["cosine_t0"]))])

    if metrics_out is not None:
        argv.extend(["--metrics-out", str(metrics_out.resolve())])
    if save_run_label:
        argv.extend(["--save-run", save_run_label])
    if skip_baselines:
        argv.append("--skip-baselines")
    if promote_best:
        pass  # default promotes unless --no-promote-best
    else:
        argv.append("--no-promote-best")

    if args.fetch_dataset:
        argv.append("--fetch-dataset")
    if args.fasttext_vec:
        argv.extend(["--fasttext-vec", args.fasttext_vec])
        argv.extend(["--fasttext-limit", str(args.fasttext_limit)])
        if args.freeze_pretrained:
            argv.append("--freeze-pretrained")
    if args.no_amp:
        argv.append("--no-amp")
    if args.no_compile or not args.allow_compile:
        argv.append("--no-compile")
    if args.no_live_plot:
        argv.append("--no-live-plot")
    return argv


def _write_experiment_metadata(
    path: Path,
    *,
    study_name: str,
    trial_number: int,
    split_seed: int,
    train_seed: int,
    study_out_dir: Path,
    objective_description: str,
) -> None:
    payload = {
        "optuna": {
            "study_name": study_name,
            "trial_number": trial_number,
            "direction": "maximize",
            "objective_metric": objective_description,
            "study_output_dir": str(study_out_dir.resolve()),
        },
        "seeds_echo": {
            "split_seed": split_seed,
            "cnn_train_seed": train_seed,
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna tuning for CNN-LSTM (subprocess run_cnn_lstm).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--study-name", type=str, default="cnn_lstm_tune")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument(
        "--optuna-seed",
        type=int,
        default=42,
        help="Sampler seed for reproducible trial suggestion order.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max training epochs per trial. Default matches run_cnn_lstm (full run).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=24,
        help="Early-stopping patience per trial. Default matches run_cnn_lstm (full run).",
    )
    p.add_argument("--seed", type=int, default=42, help="CNN training seed (single seed only).")
    p.add_argument("--split-seed", type=int, default=42, dest="split_seed")
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Passed through; 0 = same auto batch sizing as run_cnn_lstm.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset path (default: same as run_cnn_lstm DEFAULT_CHANCHAL_200_CSV).",
    )
    p.add_argument("--fetch-dataset", action="store_true")
    p.add_argument(
        "--embed-dim",
        type=int,
        default=256,
        dest="embed_dim",
        help=(
            "CNN embedding size (passed to run_cnn_lstm). Use 300 with standard FastText/cc.en.300.vec; "
            "default 256 matches run_cnn_lstm without pretrained vectors."
        ),
    )
    p.add_argument("--fasttext-vec", type=str, default=None, dest="fasttext_vec")
    p.add_argument("--fasttext-limit", type=int, default=800_000, dest="fasttext_limit")
    p.add_argument("--freeze-pretrained", action="store_true", dest="freeze_pretrained")
    p.add_argument("--no-amp", action="store_true", dest="no_amp")
    p.add_argument("--no-compile", action="store_true", dest="no_compile")
    p.add_argument(
        "--allow-compile",
        action="store_true",
        dest="allow_compile",
        help=(
            "Allow torch.compile in trial subprocesses. Default OFF for tuning to avoid "
            "triton dependency on Windows. Single-seed trials normally compile by default in "
            "run_cnn_lstm; this flag lets you re-enable it when triton is available."
        ),
    )
    p.add_argument(
        "--no-live-plot",
        action="store_true",
        dest="no_live_plot",
        help=(
            "Disable run_cnn_lstm's live matplotlib window for each trial (loss / val macro-F1 / LR). "
            "Default: live plot is ON (matches run_cnn_lstm). Use this for headless / CI runs."
        ),
    )
    p.add_argument(
        "--no-seed-known-best",
        action="store_true",
        dest="no_seed_known_best",
        help=(
            "Do NOT enqueue the project's known-good config as trial 0. By default the first trial "
            "uses (vocab=10000, seq=384, lr=8e-4, dropout=0.30, wd=1e-4, label_smoothing=0.02, plateau) "
            "so subsequent TPE trials explore around a strong baseline."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo = _repo_root()
    os.chdir(repo)

    if args.dataset is None:
        from src.dataset import DEFAULT_CHANCHAL_200_CSV

        args.dataset = DEFAULT_CHANCHAL_200_CSV

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    study_tag = _safe_filename_tag(args.study_name)
    study_out_dir = repo / "artifacts" / "tuning" / f"{study_tag}_{ts}"
    study_out_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{(study_out_dir / 'optuna.db').resolve().as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        load_if_exists=False,
    )

    if not args.no_seed_known_best:
        study.enqueue_trial(
            {
                "vocab_size": 10_000,
                "max_seq_len": 384,
                "lr": 8e-4,
                "dropout": 0.30,
                "weight_decay": 1e-4,
                "label_smoothing": 0.02,
                "lr_schedule": "plateau",
            }
        )
        logger.info(
            "Seeded trial 0 with known-good config (vocab=10000, seq=384, lr=8e-4, dropout=0.30, "
            "wd=1e-4, label_smoothing=0.02, plateau). Pass --no-seed-known-best to disable."
        )

    objective_metric = "cnn_lstm_splits.validation.f1_macro"

    def objective(trial: optuna.Trial) -> float:
        save_lbl = _trial_run_label(args.study_name, trial.number)
        metrics_out = study_out_dir / f"trial_{trial.number}_metrics.json"
        md_path = study_out_dir / f"trial_{trial.number}_experiment_md.json"

        _write_experiment_metadata(
            md_path,
            study_name=args.study_name,
            trial_number=trial.number,
            split_seed=args.split_seed,
            train_seed=args.seed,
            study_out_dir=study_out_dir,
            objective_description=objective_metric,
        )

        argv = _build_subprocess_argv(
            repo=repo,
            trial=trial,
            args=args,
            study_out_dir=study_out_dir,
            experiment_md_path=md_path,
            save_run_label=save_lbl,
            metrics_out=metrics_out,
        )

        trial.set_user_attr("subprocess_argv", argv)
        trial.set_user_attr("metrics_out", str(metrics_out))

        logger.info("Trial %d starting: %s", trial.number, " ".join(argv[:12]) + " ...")
        proc = subprocess.run(
            argv,
            cwd=str(repo),
            env=os.environ.copy(),
            check=False,
        )

        if not metrics_out.is_file():
            logger.warning(
                "Trial %d produced no metrics file (subprocess exit %s); scoring as -inf.",
                trial.number,
                proc.returncode,
            )
            return float("-inf")

        try:
            score = _read_validation_f1(metrics_out)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Trial %d wrote metrics but could not parse validation F1 (%s); scoring as -inf.",
                trial.number,
                exc,
            )
            return float("-inf")

        if proc.returncode != 0:
            logger.warning(
                "Trial %d subprocess exit code %s (likely a benign Windows shutdown crash after training); "
                "metrics file is intact so trial counts with val_f1_macro=%.6f.",
                trial.number,
                proc.returncode,
                score,
            )
            trial.set_user_attr("subprocess_exit_code", int(proc.returncode))

        run_dir = _latest_matching_run_dir(repo, save_lbl)
        if run_dir is not None:
            trial.set_user_attr("cnn_lstm_run_dir", str(run_dir))
            tj = run_dir / "training.json"
            if tj.is_file():
                trial.set_user_attr("training_json", str(tj))

        logger.info("Trial %d validation macro-F1=%.6f", trial.number, score)
        return score

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    trials_ok = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and isinstance(t.value, (int, float))
        and math.isfinite(float(t.value))
    ]
    if trials_ok:
        best_trial = max(trials_ok, key=lambda t: float(t.value))
        best_value = float(best_trial.value)
    else:
        best_value = None
        best_trial = None

    reproduce_metrics = study_out_dir / "best_params_confirm_metrics.json"
    reproduce_save_label = f"optuna_{study_tag}_best_confirm_{ts}"
    reproduce_argv = []
    if best_trial is not None:
        reproduce_argv = _argv_for_best_params_cli(
            params=best_trial.params,
            args=args,
            metrics_out=reproduce_metrics,
            save_run_label=reproduce_save_label,
            skip_baselines=False,
            promote_best=True,
        )

    summary = {
        "schema_version": 1,
        "study_name": args.study_name,
        "study_storage": storage_url,
        "study_output_dir": str(study_out_dir.resolve()),
        "optuna_sampler_seed": args.optuna_seed,
        "split_seed": args.split_seed,
        "cnn_train_seed": args.seed,
        "epochs_fixed": args.epochs,
        "patience_fixed": args.patience,
        "dataset": args.dataset,
        "batch_size_cli": args.batch_size,
        "embed_dim_cli": args.embed_dim,
        "objective_metric": objective_metric,
        "best_validation_f1_macro": best_value,
        "best_trial_number": best_trial.number if best_trial else None,
        "best_params": best_trial.params if best_trial else None,
        "best_training_json": best_trial.user_attrs.get("training_json") if best_trial else None,
        "reproduce_full_run_argv": reproduce_argv,
        "note": (
            "Trials used --skip-baselines --no-promote-best. "
            "Run reproduce_full_run_argv for baselines + promotion."
        ),
    }
    summary["n_trials_completed_finite_objective"] = len(trials_ok)
    summary_path = study_out_dir / "study_best_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote study summary %s", summary_path)

    if best_trial is not None:
        tj_str = best_trial.user_attrs.get("training_json")
        if tj_str:
            _patch_best_training_json(
                training_json=Path(tj_str),
                study=study,
                best_trial=best_trial,
                study_out_dir=study_out_dir,
                reproduce_argv=reproduce_argv,
            )
        else:
            rd = best_trial.user_attrs.get("cnn_lstm_run_dir")
            if rd:
                fallback = Path(rd) / "training.json"
                if fallback.is_file():
                    _patch_best_training_json(
                        training_json=fallback,
                        study=study,
                        best_trial=best_trial,
                        study_out_dir=study_out_dir,
                        reproduce_argv=reproduce_argv,
                    )

    print("\n--- Optuna tuning finished ---")
    print(f"Study directory: {study_out_dir}")
    print(f"Best validation macro-F1: {best_value}")
    if reproduce_argv:
        print("\nRe-run with baselines + promote (copy-paste):\n")
        print(subprocess.list2cmdline(reproduce_argv))


if __name__ == "__main__":
    main()
