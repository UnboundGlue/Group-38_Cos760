"""CNN-LSTM model for neural authorship attribution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import training_hardware
from .models import ModelConfig


class CNNLSTMModel(nn.Module):
    """CNN-LSTM model for authorship attribution.

    Subword embeddings are scanned by parallel Conv1d n-gram detectors. Each
    branch yields a sequence of CNN features (**no** pooling before recurrence).
    Multi-scale sequences are concatenated along the channel dimension, aligned to
    the input length *T*, then passed through a **stacked LSTM along time**.
    Finally, max-over-time pooling on LSTM outputs compresses each sequence into
    a single document vector (for classification or downstream reuse).

    This matches an honest CNN→LSTM stack for comparing learnt dense vectors
    to sparse TF-IDF / n-gram baselines.

    Architecture:
        1. Embedding: [B, T] → [B, T, D] with dropout.
        2. Parallel Conv1d + ReLU per kernel *k*: [B, D, T] → [B, F, T-k+1];
           pad each branch on the **right** to length *T*, concat → [B, F·K, T].
        3. Permute → [B, T, F·K], dropout on CNN channels.
        4. LSTM along time → [B, T, lstm_hidden].
        5. Max-over-time on LSTM outputs → document vector [B, lstm_hidden].
        6. Dropout + linear head → logits [B, num_classes].

    Args:
        token_ids must satisfy ``T >= max(kernel_sizes)`` so every conv map has
        at least one position.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=0,
        )
        self.embed_dropout = nn.Dropout(p=config.dropout)

        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.embed_dim,
                out_channels=config.num_filters,
                kernel_size=k,
            )
            for k in config.kernel_sizes
        ])

        cnn_output_dim = config.num_filters * len(config.kernel_sizes)
        self.cnn_dropout = nn.Dropout(p=config.dropout)

        lstm_dropout = config.dropout if config.lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.head_dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(config.lstm_hidden, config.num_classes)

    def encode(self, token_ids: Tensor) -> Tensor:
        """Return the document embedding [B, lstm_hidden] (before softmax head).

        Use this representation for retrieval, probing, or other classification
        heads while keeping subword embeddings and CNN–LSTM feature learning
        shared with :meth:`forward`.
        """
        x = self.embedding(token_ids)          # [B, T, D]
        x = self.embed_dropout(x)
        x_t = x.permute(0, 2, 1)                # [B, D, T]
        _b, _d, seq_len = x_t.shape

        branch_feats: list[Tensor] = []
        for conv in self.conv_branches:
            activated = torch.relu(conv(x_t))  # [B, F, T-k+1]
            l = activated.size(2)
            if l < seq_len:
                activated = F.pad(activated, (0, seq_len - l))
            branch_feats.append(activated)

        cnn_stack = torch.cat(branch_feats, dim=1)  # [B, F*K, T]
        cnn_stack = self.cnn_dropout(cnn_stack)
        seq = cnn_stack.permute(0, 2, 1)            # [B, T, F*K]

        lstm_out, _ = self.lstm(seq)               # [B, T, lstm_hidden]
        doc_vec, _ = lstm_out.max(dim=1)
        return doc_vec

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return classification logits [B, num_classes]."""
        doc_vec = self.encode(token_ids)
        doc_vec = self.head_dropout(doc_vec)
        return self.classifier(doc_vec)

    # --- Training/device defaults for this architecture (delegates to training_hardware) ---

    @staticmethod
    def preferred_training_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def configure_cudnn_for_device(device: torch.device) -> None:
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def gpu_total_memory_gib(device_index: int = 0) -> float | None:
        return training_hardware.gpu_total_memory_gib(device_index)

    @staticmethod
    def cuda_build_hint() -> str | None:
        return training_hardware.cuda_build_hint()

    @staticmethod
    def suggest_batch_size(
        *,
        use_cuda: bool | None = None,
        gpu_mem_gib: float | None = None,
        device_index: int = 0,
        respect_free_vram: bool = True,
    ) -> int:
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        if use_cuda and gpu_mem_gib is None:
            gpu_mem_gib = training_hardware.gpu_total_memory_gib(device_index)
        elif not use_cuda:
            gpu_mem_gib = None
        return training_hardware.suggest_batch_size(
            use_cuda=use_cuda,
            gpu_mem_gib=gpu_mem_gib,
            device_index=device_index,
            respect_free_vram=respect_free_vram,
        )

    @staticmethod
    def suggest_num_workers() -> int:
        return training_hardware.suggest_num_workers()
