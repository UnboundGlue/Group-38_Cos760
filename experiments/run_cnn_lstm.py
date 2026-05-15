"""End-to-end CNN-LSTM authorship attribution pipeline.

Wires together:
    DatasetLoader → Preprocessor → SubwordTokeniser → CNNLSTMModel → Trainer → Evaluator

Requirements covered:
    9.1 – Reproducible results with a fixed random seed.
    9.2 – Save evaluation metrics to results/metrics.json.
    9.3 – Save trained tokeniser to artifacts/tokeniser.json.
    9.4 – Save best checkpoints under a timestamped artifacts/runs/<label>_<UTC>/
           directory (trainer overwrites model.pt inside that run when val F1 improves).
    9.6 – Produce both CNN-LSTM and baseline evaluation results.

Usage examples::

    python -m experiments.run_cnn_lstm --fetch-dataset
    python -m experiments.run_cnn_lstm --fetch-dataset --no-live-plot
    python -m experiments.run_cnn_lstm --lr-schedule onecycle --lr 6e-4 --label-smoothing 0
    python -m experiments.run_cnn_lstm --seed 42 --ensemble-train-seeds "42,43,44"
    python -m experiments.run_cnn_lstm --no-promote-best
    python -m experiments.load_cnn_checkpoint --artifact artifacts/best_model_bundle --eval
    python -m experiments.run_cnn_lstm --split-seed 42 --ensemble-train-seeds "10,11,12" \\
        --checkpoint artifacts/manual/ens.pt
    python -m experiments.run_cnn_lstm --fasttext-vec path/to/cc.en.300.vec \\
        --fasttext-limit 800000 --embed-dim 300

Chanchal 50-author / fine-tuning (same ``--split-seed``): try **one** knob group per run.
If ``--batch-size 800`` stalls, prefer **more steps per epoch** (smaller batch + slightly higher ``--lr``) or ``--lr-schedule onecycle``::

    # More gradients per epoch + mild LR bump (baseline follow-up after batch 800)
    python -m experiments.run_cnn_lstm --fetch-dataset --epochs 90 --patience 24 --split-seed 42 \\
        --ensemble-train-seeds "42,43,44" --batch-size 384 --lr 9e-4 --vocab-size 12288

    # One-cycle on mid-size batches (keep label smoothing modest)
    python -m experiments.run_cnn_lstm --fetch-dataset --epochs 90 --patience 28 --split-seed 42 \\
        --ensemble-train-seeds "42,43,44" --batch-size 384 --lr-schedule onecycle \\
        --lr 7e-4 --weight-decay 1.5e-4

    # Extra temporal capacity without huge batch-memory jump
    python -m experiments.run_cnn_lstm --fetch-dataset --epochs 90 --patience 24 --split-seed 42 \\
        --ensemble-train-seeds "42,43,44" --batch-size 384 --lr 9e-4 \\
        --lstm-hidden 448 --dropout 0.28 --weight-decay 1.5e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure the Repository/src package is importable when the script is run
# directly from the Repository directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import DEFAULT_CHANCHAL_200_CSV, DatasetLoader
from src.evaluate import (
    ensemble_evaluate_majority_vote,
    evaluate,
    metrics_dict_to_jsonable,
)
from src.model import CNNLSTMModel
from src.models import ModelConfig, TrainingHistory
from src.preprocessing import Preprocessor
from src.sparse_baselines import nested_sparse_baseline_results
from src.tokeniser import SubwordTokeniser
from src import training_hardware
from src.run_artifact import (
    build_training_record,
    history_best_val_f1_macro,
    promote_bundle_if_improved,
    save_run_bundle,
)
from src.trainer import Trainer
from src.training_live_plot import LiveTrainingPlot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Fix all relevant random seeds for reproducibility (Req 9.1)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism defaults; on CUDA, run() may enable cuDNN benchmark for throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _configure_cudnn_for_device(device: torch.device) -> None:
    """On GPU, prefer throughput over strict cudnn bit-for-bit repeatability."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Speeds matmuls on Ampere and newer — does not change optimizer hyperparameters.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _maybe_torch_compile(
    module: CNNLSTMModel,
    device: torch.device,
    *,
    enable: bool,
    compile_mode: str = "default",
) -> torch.nn.Module:
    """Graph compilation (PyTorch 2+) for extra CUDA throughput — same maths, warmup on early batches.

    *compile_mode*: ``default`` → ``torch.compile(..., mode="default")``; ``reduce-overhead`` may help short
    steps; ``max-autotune`` spends more time optimizing graphs (much slower warmup, often fastest steady GPU).

    Not used for multi-seed ensembles: compiling **per member** would pay warmup **N** times."""
    if not enable or device.type != "cuda":
        return module
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return module
    try:
        out = compile_fn(module, mode=compile_mode, fullgraph=False)  # type: ignore[misc]
        logger.info(
            "torch.compile enabled mode=%s (first epochs include graph warmup).",
            compile_mode,
        )
        return out  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001 — optional optimisation
        logger.warning("torch.compile skipped (%s): %s", type(exc).__name__, exc)
        return module


def _dataloader_worker_seeded_init(base_seed: int, worker_id: int) -> None:
    """Per-worker PRNG (must be top-level: Windows spawn must pickle *worker_init_fn*)."""
    s = int(base_seed) + 1000 + int(worker_id)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _make_loader(
    token_ids: np.ndarray,
    labels: list[int],
    batch_size: int,
    shuffle: bool = False,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    dataloader_seed: int | None = None,
) -> DataLoader:
    """Wrap encoded token IDs and labels in a DataLoader.

    When *dataloader_seed* is set and *shuffle* is True, the batch order is tied to
    that seed. With ``num_workers > 0``, workers get derived numpy/torch seeds.
    """
    x = torch.tensor(token_ids, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(x, y)
    opts: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if shuffle and dataloader_seed is not None:
        g = torch.Generator()
        g.manual_seed(dataloader_seed)
        opts["generator"] = g
    if dataloader_seed is not None and num_workers > 0:
        # partial(top_level_fn, int) is picklable; nested def is not (Windows spawn / DataLoader)
        opts["worker_init_fn"] = partial(
            _dataloader_worker_seeded_init, int(dataloader_seed)
        )
    if num_workers > 0:
        opts["persistent_workers"] = True
        opts["prefetch_factor"] = 4
    return DataLoader(dataset, **opts)


def _balanced_class_weights_tensor(labels: list[int], num_classes: int) -> torch.Tensor:
    """sklearn balanced weights for present classes; absent classes stay 1.0."""
    from sklearn.utils.class_weight import compute_class_weight

    y = np.asarray(labels, dtype=np.int64)
    present = np.unique(y)
    cw = compute_class_weight("balanced", classes=present, y=y)
    w = np.ones(num_classes, dtype=np.float32)
    for c, wt in zip(present, cw):
        w[int(c)] = float(wt)
    return torch.from_numpy(w)


def _parse_comma_separated_ints(s: str | None) -> list[int] | None:
    if not s or not str(s).strip():
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and evaluate the CNN-LSTM authorship attribution model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default=DEFAULT_CHANCHAL_200_CSV,
        help=(
            "Path to dataset file (CSV or JSON). Default: Chanchal 200-tweets slice "
            "(more data per author; see DEFAULT_CHANCHAL_200_CSV in src/dataset.py)."
        ),
    )
    p.add_argument(
        "--fetch-dataset",
        action="store_true",
        help="If the file is missing, clone chanchalIITP/AuthorIdentification into data/ (needs git).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (also default split seed).")
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        dest="split_seed",
        help="Stratified train/val/test seed; default = --seed. Use with --ensemble-train-seeds.",
    )
    p.add_argument(
        "--ensemble-train-seeds",
        default=None,
        dest="ensemble_train_seeds",
        help=(
            "Comma-separated training seeds (e.g. 42,43,44). Same data split; retrains one model per "
            "seed and aggregates predictions by majority vote (tie-break: mean logit), plus each member row."
        ),
    )
    p.add_argument("--vocab-size", type=int, default=10_000, dest="vocab_size",
                   help="Tokeniser vocabulary size.")
    p.add_argument("--embed-dim", type=int, default=256, dest="embed_dim",
                   help="Embedding dimension.")
    p.add_argument("--num-filters", type=int, default=256, dest="num_filters",
                   help="Number of CNN filters per kernel size.")
    p.add_argument("--lstm-hidden", type=int, default=384, dest="lstm_hidden",
                   help="LSTM hidden state size.")
    p.add_argument("--lstm-layers", type=int, default=2, dest="lstm_layers",
                   help="Number of stacked LSTM layers.")
    p.add_argument("--dropout", type=float, default=0.30, help="Dropout rate.")
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=384,
        dest="max_seq_len",
        help="Maximum token sequence length (longer captures more of each tweet).",
    )
    p.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        dest="batch_size",
        help="Training batch size; 0 = set from GPU memory or CPU (default).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        dest="num_workers",
        help="DataLoader worker processes; -1 = from CPU count (default). Use 0 to disable workers.",
    )
    p.add_argument("--lr", type=float, default=8e-4, help="Learning rate.")
    p.add_argument("--patience", type=int, default=24, help="Early-stopping patience.")
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        dest="weight_decay",
        help="Adam L2 weight decay (regularisation).",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.02,
        dest="label_smoothing",
        help="Cross-entropy label smoothing (0 = off; try 0.02 for tight multiclass margins).",
    )
    p.add_argument(
        "--lr-schedule",
        choices=("none", "plateau", "cosine_restarts", "onecycle"),
        default="plateau",
        dest="lr_schedule",
        help=(
            "LR schedule: plateau (default), cosine_restarts (per-epoch), "
            "onecycle (OneCycleLR per batch over all training steps)."
        ),
    )
    p.add_argument(
        "--cosine-t0",
        type=int,
        default=8,
        dest="cosine_t0",
        help="Epochs per cosine cycle before first warm restart (cosine_restarts only).",
    )
    p.add_argument(
        "--onecycle-max-lr",
        type=float,
        default=None,
        dest="onecycle_max_lr",
        help="Peak LR for OneCycle; default = --lr.",
    )
    p.add_argument(
        "--onecycle-div-factor",
        type=float,
        default=25.0,
        dest="onecycle_div_factor",
    )
    p.add_argument(
        "--onecycle-pct-start",
        type=float,
        default=0.1,
        dest="onecycle_pct_start",
        help="Fraction of one-cycle steps spent increasing LR.",
    )
    p.add_argument(
        "--onecycle-final-div-factor",
        type=float,
        default=10000.0,
        dest="onecycle_final_div_factor",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        dest="checkpoint",
        help=(
            "Trainer best-weights path. Empty (default): under this run's "
            "artifacts/runs/<label>_<UTC>/ — model.pt (single seed) or "
            "model_train<seed>.pt (ensemble). Pass a path to store checkpoints elsewhere."
        ),
    )
    p.add_argument(
        "--run-label",
        type=str,
        default="run",
        dest="run_label",
        help=(
            "Run directory prefix artifacts/runs/<run-label>_<UTC>/. Ignored when --save-run is used "
            "(that flag sets the label)."
        ),
    )
    p.add_argument(
        "--save-run",
        nargs="?",
        const="cnn",
        default=None,
        dest="save_run",
        help=(
            "Optional alternate run label prefix: artifacts/runs/<label>_<UTC>/ (default label 'cnn' "
            "if the flag is given with no argument). Each training run always snapshots "
            "model.pt + tokeniser.json + training.json into that folder."
        ),
    )
    p.add_argument(
        "--promote-best-dir",
        type=str,
        default="artifacts/best_model_bundle",
        dest="promote_best_dir",
        help=(
            "Canonical bundle directory: after training, updated only when this run's strongest "
            "validation macro-F1 **strictly** beats training.json stored there (or dir is empty)."
        ),
    )
    p.add_argument(
        "--no-promote-best",
        action="store_true",
        dest="no_promote_best",
        help=(
            "Do not refresh the canonical best bundle (--promote-best-dir). Per-run dirs are unchanged."
        ),
    )
    p.add_argument(
        "--metrics-out",
        type=str,
        default="results/metrics.json",
        dest="metrics_out",
        help="Aggregated metrics JSON destination.",
    )
    p.add_argument(
        "--fasttext-vec",
        type=str,
        default=None,
        dest="fasttext_vec",
        help=(
            "Optional FastText/word2vec-format .vec file (wiki-news-300d, cc.en.300, …) "
            "for embedding initialisation; pieces matched case-insensitively."
        ),
    )
    p.add_argument(
        "--fasttext-limit",
        type=int,
        default=800_000,
        dest="fasttext_limit",
        help="Max embedding rows to read from .vec (caps RAM on huge files). Use 0 for whole file.",
    )
    p.add_argument(
        "--freeze-pretrained",
        action="store_true",
        dest="freeze_pretrained",
        help="Freeze embedding table after loading --fasttext-vec.",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        dest="no_compile",
        help=(
            "Disable torch.compile kernel fusion on CUDA when training a single ensemble member "
            "(default: enabled for exactly one CNN training run only)."
        ),
    )
    p.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
        dest="compile_mode",
        help=(
            "torch.compile optimization level (CUDA, single seed only unless --no-compile). "
            "max-autotune = much slower first epochs then often highest throughput."
        ),
    )
    p.add_argument(
        "--no-fused-adam",
        action="store_true",
        dest="no_fused_adam",
        help="Disable CUDA fused Adam (reference kernels). Default: try fused Adam on CUDA when supported.",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        dest="no_amp",
        help="Disable CUDA automatic mixed precision (default: on for GPU; keep for debugging).",
    )
    p.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default="balanced",
        dest="class_weight",
        help="balanced = inverse-frequency loss weights (helps rare authors).",
    )
    p.add_argument(
        "--no-live-plot",
        action="store_true",
        dest="no_live_plot",
        help=(
            "Disable the matplotlib window with live train loss, val macro-F1, and LR "
            "(enabled by default). Use for headless/CI or if the GUI slows training."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    """Execute the full pipeline and return a results dict.

    With ``--ensemble-train-seeds``, the same stratified split (``--split-seed`` /
    ``--seed``) is reused; each training seed retrains CNN-LSTM (new init + loaders).
    Metrics JSON ``cnn_lstm`` uses ensemble **majority vote** when N>1 (ties: mean logit); see
    ``cnn_lstm_members`` for per-seed checkpoints and single-model test metrics.
    """
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    train_seeds = _parse_comma_separated_ints(args.ensemble_train_seeds)
    if train_seeds is None:
        train_seeds = [args.seed]

    logger.info(
        "Split stratify seed=%d | CNN training seeds=%s%s",
        split_seed,
        train_seeds,
        " (+ ensemble headline if |seeds|>1)" if len(train_seeds) > 1 else "",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_cudnn_for_device(device)

    if device.type == "cuda":
        mem_gib = training_hardware.gpu_total_memory_gib(0)
        logger.info(
            "Using device: %s  (%s, ~%.1f GiB total)",
            device,
            torch.cuda.get_device_name(0),
            mem_gib or 0.0,
        )
    else:
        logger.info("Using device: %s", device)
        hint = training_hardware.cuda_build_hint()
        if hint is not None:
            logger.warning("%s", hint)

    pin_memory = device.type == "cuda"
    if args.batch_size <= 0:
        mem = training_hardware.gpu_total_memory_gib(0) if device.type == "cuda" else None
        args.batch_size = training_hardware.suggest_batch_size(
            use_cuda=device.type == "cuda",
            gpu_mem_gib=mem,
            device_index=0,
        )
    if args.num_workers < 0:
        args.num_workers = training_hardware.suggest_num_workers()
    logger.info(
        "Training IO: batch_size=%d, num_workers=%d, pin_memory=%s",
        args.batch_size, args.num_workers, pin_memory,
    )
    # Eval/split metrics loaders: keep ``num_workers=0``. On Windows, ``spawn`` workers
    # re-import this script (and SciPy/OpenBLAS DLLs) per process; after long CUDA runs
    # that can trigger ``paging file is too small`` / worker exit during ensemble eval.
    eval_num_workers = 0
    if len(train_seeds) > 1:
        n_ms = len(train_seeds)
        logger.info(
            "Ensemble note: %d full CNN trainings run **one after another** on this GPU — "
            "expect ~%d× the wall time of a single-seed run (AMP helps each run, it does not parallelize).",
            n_ms,
            n_ms,
        )
        if device.type == "cuda" and not args.no_compile:
            logger.info(
                "torch.compile is disabled for multi-seed runs (avoids %d× graph warmup). "
                "Single-seed runs compile by default on CUDA (use --no-compile to disable).",
                n_ms,
            )

    logger.info("Loading dataset from %s …", args.dataset)
    data_loader_svc = DatasetLoader()
    texts, labels = data_loader_svc.load(args.dataset, fetch_if_missing=args.fetch_dataset)
    num_classes = data_loader_svc.num_authors
    logger.info("Loaded %d samples across %d authors.", len(texts), num_classes)

    preprocessor = Preprocessor()
    texts = preprocessor.batch_clean(texts)
    paired = [(t, l) for t, l in zip(texts, labels) if t]
    if not paired:
        raise ValueError("All texts became empty after preprocessing.")
    texts, labels = zip(*paired)
    texts = list(texts)
    labels = list(labels)

    train_split, val_split, test_split = data_loader_svc.split(texts, labels, seed=split_seed)
    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_split.texts), len(val_split.texts), len(test_split.texts),
    )

    logger.info("Training SubwordTokeniser (vocab_size=%d) …", args.vocab_size)
    tokeniser = SubwordTokeniser()
    tokeniser.train(train_split.texts, vocab_size=args.vocab_size)

    tokeniser_path = "artifacts/tokeniser.json"
    os.makedirs(os.path.dirname(tokeniser_path), exist_ok=True)
    tokeniser.save(tokeniser_path)
    logger.info("Tokeniser saved to %s", tokeniser_path)

    logger.info("Encoding text splits …")
    train_ids = tokeniser.batch_encode(train_split.texts, max_length=args.max_seq_len)
    val_ids = tokeniser.batch_encode(val_split.texts, max_length=args.max_seq_len)
    test_ids = tokeniser.batch_encode(test_split.texts, max_length=args.max_seq_len)

    train_eval_loader = _make_loader(
        train_ids, train_split.labels, args.batch_size,
        shuffle=False, num_workers=eval_num_workers, pin_memory=pin_memory,
    )
    val_eval_loader = _make_loader(
        val_ids, val_split.labels, args.batch_size,
        shuffle=False, num_workers=eval_num_workers, pin_memory=pin_memory,
    )
    test_eval_loader = _make_loader(
        test_ids, test_split.labels, args.batch_size,
        shuffle=False, num_workers=eval_num_workers, pin_memory=pin_memory,
    )

    val_loader = _make_loader(
        val_ids, val_split.labels, args.batch_size,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    model_config = ModelConfig(
        vocab_size=tokeniser.vocab_size(),
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        num_classes=num_classes,
    )
    logger.info("CNN-LSTM architecture (members share this config): %s", model_config)

    if args.class_weight == "balanced":
        class_weights = _balanced_class_weights_tensor(train_split.labels, num_classes)
        logger.info("Using balanced per-class loss weights (train distribution).")
    else:
        class_weights = None
        logger.info("Class-weighted loss disabled (--class-weight none).")

    ft_limit = None if args.fasttext_limit == 0 else int(args.fasttext_limit)

    label = (
        str(args.save_run).strip() or "cnn"
        if args.save_run is not None
        else str(args.run_label).strip() or "run"
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = Path("artifacts/runs") / f"{label}_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory (checkpoints + bundle): %s", run_root.resolve())

    ckpt_explicit = str(args.checkpoint or "").strip()

    member_checkpoints: list[str] = []
    member_test_json: list[dict] = []
    cnn_training_wall_total = 0.0
    dataloader_base = 1_000_000 + split_seed
    ev_amp_kw: bool | None = False if args.no_amp else None
    cuda_amp_infer = not args.no_amp

    promotion_candidates: list[tuple[int, TrainingHistory, float, str]] = []

    prev_live_plot: LiveTrainingPlot | None = None
    for member_idx, train_seed in enumerate(train_seeds):
        _set_seed(train_seed)
        logger.info(
            "=== CNN member %d/%d  train_seed=%d ===",
            member_idx + 1, len(train_seeds), train_seed,
        )

        train_loader = _make_loader(
            train_ids,
            train_split.labels,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            dataloader_seed=dataloader_base + train_seed,
        )

        model = CNNLSTMModel(model_config).to(device)
        if args.fasttext_vec:
            from src.pretrained_embeddings import apply_fasttext_vec_file

            apply_fasttext_vec_file(
                model.embedding,
                tokeniser,
                args.fasttext_vec,
                embed_dim=args.embed_dim,
                limit_vectors=ft_limit,
                freeze_pretrained=args.freeze_pretrained,
            )

        model = _maybe_torch_compile(
            model,
            device,
            enable=device.type == "cuda" and len(train_seeds) == 1 and not args.no_compile,
            compile_mode=args.compile_mode,
        )

        if len(train_seeds) == 1:
            ckpt_path = ckpt_explicit if ckpt_explicit else str(run_root / "model.pt")
        else:
            if ckpt_explicit:
                root, ext = os.path.splitext(ckpt_explicit)
                ckpt_path = f"{root}_train{train_seed}{ext or '.pt'}"
            else:
                ckpt_path = str(run_root / f"model_train{train_seed}.pt")
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

        trainer = Trainer()
        logger.info("Starting training (max_epochs=%d, patience=%d) …", args.epochs, args.patience)
        live_plot: LiveTrainingPlot | None = None
        epoch_hook = None
        if not args.no_live_plot:
            if prev_live_plot is not None:
                prev_live_plot.close()
            plot_title = (
                f"CNN-LSTM  member {member_idx + 1}/{len(train_seeds)}  train_seed={train_seed}"
                if len(train_seeds) > 1
                else f"CNN-LSTM  train_seed={train_seed}"
            )
            live_plot = LiveTrainingPlot(title=plot_title)
            prev_live_plot = live_plot
            epoch_hook = lambda ep, max_ep, loss, f1, lr, lp=live_plot: lp.update(ep, loss, f1, lr)
        t_train0 = time.perf_counter()
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            checkpoint_path=ckpt_path,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            reduce_lr_on_plateau=False,
            class_weights=class_weights,
            lr_schedule=args.lr_schedule,
            cosine_t0=args.cosine_t0,
            onecycle_max_lr=args.onecycle_max_lr,
            onecycle_div_factor=args.onecycle_div_factor,
            onecycle_pct_start=args.onecycle_pct_start,
            onecycle_final_div_factor=args.onecycle_final_div_factor,
            use_amp=False if args.no_amp else None,
            adam_fused=not args.no_fused_adam,
            on_epoch_end=epoch_hook,
        )
        member_wall_s = time.perf_counter() - t_train0
        cnn_training_wall_total += member_wall_s

        if os.path.isfile(ckpt_path):
            dest = (
                run_root / f"member_train{train_seed}"
                if len(train_seeds) > 1
                else run_root
            )
            record = build_training_record(
                history=history,
                training_wall_seconds=member_wall_s,
                lr_schedule=args.lr_schedule,
                split_seed=split_seed,
                train_seed=train_seed,
                cli_namespace=args,
                model_config=model_config,
                checkpoint_path_written=ckpt_path,
                primary_lr_arg=args.lr,
            )
            save_run_bundle(
                dest,
                checkpoint_src=ckpt_path,
                tokeniser_src=tokeniser_path,
                record=record,
            )
            logger.info("Saved run bundle to %s", dest)
        else:
            logger.warning(
                "Checkpoint missing at %s; skipping run bundle for train_seed=%d.",
                ckpt_path,
                train_seed,
            )

        if os.path.isfile(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            logger.info("Loaded best checkpoint from %s", ckpt_path)
        member_checkpoints.append(ckpt_path)

        m_test_one = evaluate(model, test_eval_loader, use_amp=ev_amp_kw)
        member_test_json.append(
            {"train_seed": train_seed, "checkpoint": ckpt_path, "test": metrics_dict_to_jsonable(m_test_one)}
        )
        promotion_candidates.append((train_seed, history, member_wall_s, ckpt_path))

    if not args.no_promote_best:
        promote_dir = Path(str(args.promote_best_dir).strip() or "artifacts/best_model_bundle")
        ranked: list[tuple[float, tuple[int, TrainingHistory, float, str]]] = []
        for train_seed, history, wall_s, ckpt_path in promotion_candidates:
            score = history_best_val_f1_macro(history)
            if score is None:
                logger.warning(
                    "Global best bundle: train_seed=%d has no validation history; skipping for promotion.",
                    train_seed,
                )
                continue
            ranked.append((float(score), (train_seed, history, wall_s, ckpt_path)))
        if not ranked:
            logger.warning("Global best bundle: no usable candidates; skipping promotion.")
        else:
            best_score, (win_seed, win_history, win_wall_s, win_ckpt) = max(ranked, key=lambda x: x[0])
            if not os.path.isfile(win_ckpt):
                logger.warning(
                    "Global best bundle: winning checkpoint missing at %s; skip promotion.",
                    win_ckpt,
                )
            else:
                record = build_training_record(
                    history=win_history,
                    training_wall_seconds=win_wall_s,
                    lr_schedule=args.lr_schedule,
                    split_seed=split_seed,
                    train_seed=win_seed,
                    cli_namespace=args,
                    model_config=model_config,
                    checkpoint_path_written=win_ckpt,
                    primary_lr_arg=args.lr,
                )
                status, prev = promote_bundle_if_improved(
                    promote_dir,
                    candidate_best_val_f1=float(record["best_val_f1_macro"]),
                    checkpoint_src=win_ckpt,
                    tokeniser_src=tokeniser_path,
                    record=record,
                )
                if status == "promoted":
                    if prev is None:
                        logger.info(
                            "Global best bundle: wrote %s (best_val_f1_macro=%.6f).",
                            promote_dir,
                            best_score,
                        )
                    else:
                        logger.info(
                            "Global best bundle: updated %s — new best_val_f1_macro %.6f (was %.6f).",
                            promote_dir,
                            best_score,
                            prev,
                        )
                else:
                    logger.info(
                        "Global best bundle: keeping incumbent at %s (best_val_f1_macro %.6f >= "
                        "this run %.6f, train_seed=%d).",
                        promote_dir,
                        prev or 0.0,
                        best_score,
                        win_seed,
                    )

    if len(train_seeds) > 1:
        ens_index = {
            "schema_version": 1,
            "members": [
                {"train_seed": ts, "relative_dir": f"member_train{ts}"} for ts in train_seeds
            ],
        }
        (run_root / "ensemble_index.json").write_text(
            json.dumps(ens_index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # --- Eval: single model vs ensemble (majority vote) ---
    template = CNNLSTMModel(model_config).to(device)

    if len(train_seeds) == 1:
        if os.path.isfile(member_checkpoints[0]):
            template.load_state_dict(torch.load(member_checkpoints[0], map_location=device))
        t_ev0 = time.perf_counter()
        m_train = evaluate(template, train_eval_loader, use_amp=ev_amp_kw)
        cnn_eval_train_s = time.perf_counter() - t_ev0
        t_ev0 = time.perf_counter()
        m_val = evaluate(template, val_eval_loader, use_amp=ev_amp_kw)
        cnn_eval_val_s = time.perf_counter() - t_ev0
        t_ev0 = time.perf_counter()
        cnn_lstm_metrics = evaluate(template, test_eval_loader, use_amp=ev_amp_kw)
        cnn_eval_test_s = time.perf_counter() - t_ev0
    else:
        logger.info(
            "Ensemble evaluation: majority vote across %d checkpoints (tie-break: mean logit).",
            len(member_checkpoints),
        )
        t_ev0 = time.perf_counter()
        m_train = ensemble_evaluate_majority_vote(
            template, member_checkpoints, train_eval_loader, device, cuda_amp=cuda_amp_infer
        )
        cnn_eval_train_s = time.perf_counter() - t_ev0
        t_ev0 = time.perf_counter()
        m_val = ensemble_evaluate_majority_vote(
            template, member_checkpoints, val_eval_loader, device, cuda_amp=cuda_amp_infer
        )
        cnn_eval_val_s = time.perf_counter() - t_ev0
        t_ev0 = time.perf_counter()
        cnn_lstm_metrics = ensemble_evaluate_majority_vote(
            template, member_checkpoints, test_eval_loader, device, cuda_amp=cuda_amp_infer
        )
        cnn_eval_test_s = time.perf_counter() - t_ev0

    logger.info(
        "CNN-LSTM split metrics — train acc=%.4f f1=%.4f | val acc=%.4f f1=%.4f | test acc=%.4f f1=%.4f",
        m_train.accuracy,
        m_train.f1_macro,
        m_val.accuracy,
        m_val.f1_macro,
        cnn_lstm_metrics.accuracy,
        cnn_lstm_metrics.f1_macro,
    )
    logger.info("CNN-LSTM training wall (all members): %.2fs", cnn_training_wall_total)

    t_base0 = time.perf_counter()
    baseline_results, baseline_timing = nested_sparse_baseline_results(
        train_split, val_split, test_split, seed=split_seed
    )
    baselines_total_wall_s = time.perf_counter() - t_base0

    results: dict = {
        "schema_version": 2,
        "cnn_lstm_run_dir": str(run_root.resolve()),
        "cnn_lstm": metrics_dict_to_jsonable(cnn_lstm_metrics),
        "cnn_lstm_splits": {
            "train": metrics_dict_to_jsonable(m_train),
            "validation": metrics_dict_to_jsonable(m_val),
            "test": metrics_dict_to_jsonable(cnn_lstm_metrics),
        },
        "cnn_lstm_training_seeds": train_seeds,
        "cnn_lstm_split_seed": split_seed,
        "baselines": baseline_results,
        "timings_seconds": {
            "cnn_training_wall": cnn_training_wall_total,
            "cnn_eval_train_wall": cnn_eval_train_s,
            "cnn_eval_validation_wall": cnn_eval_val_s,
            "cnn_eval_test_wall": cnn_eval_test_s,
            "baselines_cpu_wall": baselines_total_wall_s,
            "baselines_detail": baseline_timing,
        },
    }

    if len(train_seeds) > 1:
        results["cnn_lstm_members"] = member_test_json

    results_path = args.metrics_out
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Metrics saved to %s", results_path)

    return results


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
