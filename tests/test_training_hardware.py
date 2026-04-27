"""Unit tests for src.training_hardware heuristics."""

from __future__ import annotations

from src import training_hardware


def test_suggest_batch_size_cuda_tiers() -> None:
    assert training_hardware.suggest_batch_size(use_cuda=True, gpu_mem_gib=4.0) == 32
    assert training_hardware.suggest_batch_size(use_cuda=True, gpu_mem_gib=7.0) == 48
    assert training_hardware.suggest_batch_size(use_cuda=True, gpu_mem_gib=10.0) == 64
    assert training_hardware.suggest_batch_size(use_cuda=True, gpu_mem_gib=14.0) == 96
    assert training_hardware.suggest_batch_size(use_cuda=True, gpu_mem_gib=20.0) == 128


def test_suggest_batch_size_cpu() -> None:
    assert training_hardware.suggest_batch_size(use_cuda=False, gpu_mem_gib=None) == 32


def test_suggest_num_workers_is_non_negative() -> None:
    n = training_hardware.suggest_num_workers()
    assert 0 <= n <= 8
