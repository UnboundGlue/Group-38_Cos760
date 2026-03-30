"""CNN-LSTM model for neural authorship attribution."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .models import ModelConfig


class CNNLSTMModel(nn.Module):
    """CNN-LSTM model for authorship attribution.

    Architecture:
        1. Embedding lookup: [B, T] → [B, T, D] with dropout
        2. Parallel Conv1d branches per kernel_size with ReLU + global max-over-time pooling
        3. Concatenate multi-scale features → [B, num_filters * len(kernel_sizes)], apply dropout
        4. Reshape for LSTM: [B, 1, num_filters * len(kernel_sizes)]
        5. Stacked LSTM; take last-layer hidden state h_n[-1] → [B, lstm_hidden]
        6. Dropout + Linear classification head → logits [B, num_classes]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        # Step 1: Embedding layer with dropout
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=0,
        )
        self.embed_dropout = nn.Dropout(p=config.dropout)

        # Step 2: Parallel CNN branches — one per kernel size
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.embed_dim,
                out_channels=config.num_filters,
                kernel_size=k,
            )
            for k in config.kernel_sizes
        ])

        # Step 3: Dropout after CNN concatenation
        cnn_output_dim = config.num_filters * len(config.kernel_sizes)
        self.cnn_dropout = nn.Dropout(p=config.dropout)

        # Steps 4–5: Stacked LSTM
        # lstm_dropout only applied between layers (requires lstm_layers > 1)
        lstm_dropout = config.dropout if config.lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Step 6: Dropout + classification head
        self.head_dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(config.lstm_hidden, config.num_classes)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Forward pass.

        Args:
            token_ids: Integer tensor of shape [B, T].

        Returns:
            logits: Float tensor of shape [B, num_classes].
        """
        # Step 1: Embedding lookup + dropout → [B, T, D]
        x = self.embedding(token_ids)          # [B, T, D]
        x = self.embed_dropout(x)

        # Step 2: Parallel CNN branches
        # Conv1d expects [B, C_in, L], so permute to [B, D, T]
        x_t = x.permute(0, 2, 1)              # [B, D, T]

        pooled_features: list[Tensor] = []
        for conv in self.conv_branches:
            conv_out = conv(x_t)               # [B, num_filters, T-k+1]
            activated = torch.relu(conv_out)   # [B, num_filters, T-k+1]
            # Global max-over-time pooling
            pooled, _ = activated.max(dim=2)   # [B, num_filters]
            pooled_features.append(pooled)

        # Step 3: Concatenate + dropout → [B, num_filters * len(kernel_sizes)]
        cnn_concat = torch.cat(pooled_features, dim=1)  # [B, num_filters * len(kernel_sizes)]
        cnn_concat = self.cnn_dropout(cnn_concat)

        # Step 4: Reshape for LSTM → [B, 1, num_filters * len(kernel_sizes)]
        lstm_input = cnn_concat.unsqueeze(1)   # [B, 1, cnn_output_dim]

        # Step 5: LSTM → take last-layer hidden state
        _, (h_n, _) = self.lstm(lstm_input)    # h_n: [lstm_layers, B, lstm_hidden]
        final_hidden = h_n[-1]                 # [B, lstm_hidden]

        # Step 6: Dropout + linear head → logits [B, num_classes]
        final_hidden = self.head_dropout(final_hidden)
        logits = self.classifier(final_hidden) # [B, num_classes]

        return logits
