"""Live matplotlib charts for CNN training (loss, validation macro-F1, learning rate).

Use from the training script with ``Trainer(..., on_epoch_end=...)`` or via
``python -m experiments.run_cnn_lstm`` (live curves on by default; pass ``--no-live-plot`` to disable).

Requires ``matplotlib`` and a display (GUI backend). Headless runs should pass
``--no-live-plot`` or set ``MPLBACKEND`` to an interactive backend when available.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LiveTrainingPlot:
    """Three stacked line plots updated once per epoch."""

    def __init__(self, *, title: str = "CNN-LSTM training") -> None:
        self._title = title
        self._epochs: list[int] = []
        self._losses: list[float] = []
        self._val_f1: list[float] = []
        self._lr: list[float] = []
        self._fig: Any = None
        self._axes: Any = None
        self._line_loss: Any = None
        self._line_f1: Any = None
        self._line_lr: Any = None
        self._plt: Any = None
        self._setup_ok: bool | None = None

    def _try_set_interactive_backend(self) -> None:
        import matplotlib

        if matplotlib.get_backend().lower() != "agg":
            return
        for name in ("TkAgg", "Qt5Agg", "QtAgg"):
            try:
                matplotlib.use(name, force=True)
                logger.info("Live plot using matplotlib backend %s.", name)
                return
            except Exception:
                continue

    def _ensure_figure(self) -> bool:
        if self._setup_ok is False:
            return False
        if self._fig is not None:
            return True
        try:
            self._try_set_interactive_backend()
            import matplotlib.pyplot as plt

            self._plt = plt
            plt.ion()
            self._fig, axes = plt.subplots(
                3, 1, sharex=True, figsize=(9, 7), constrained_layout=True
            )
            ax_l, ax_f, ax_r = axes[0], axes[1], axes[2]
            self._axes = (ax_l, ax_f, ax_r)
            self._fig.suptitle(self._title)

            (self._line_loss,) = ax_l.plot([], [], color="C0", linewidth=1.5, label="train loss")
            ax_l.set_ylabel("loss")
            ax_l.grid(True, alpha=0.3)
            ax_l.legend(loc="upper right")

            (self._line_f1,) = ax_f.plot([], [], color="C2", linewidth=1.5, label="val F1 (macro)")
            ax_f.set_ylabel("val F1")
            ax_f.set_ylim(0.0, 1.0)
            ax_f.grid(True, alpha=0.3)
            ax_f.legend(loc="lower right")

            (self._line_lr,) = ax_r.semilogy([], [], color="C1", linewidth=1.5, label="lr")
            ax_r.set_ylabel("learning rate")
            ax_r.set_xlabel("epoch")
            ax_r.grid(True, alpha=0.3)
            ax_r.legend(loc="upper right")

            self._setup_ok = True
            return True
        except Exception as exc:
            logger.warning("Live training plot unavailable (%s): %s", type(exc).__name__, exc)
            self._setup_ok = False
            return False

    def update(self, epoch: int, train_loss: float, val_f1_macro: float, lr: float) -> None:
        """Append one epoch and refresh the figure."""
        if not self._ensure_figure():
            return
        self._epochs.append(int(epoch))
        self._losses.append(float(train_loss))
        self._val_f1.append(float(val_f1_macro))
        self._lr.append(float(lr))

        ax_l, ax_f, ax_r = self._axes
        self._line_loss.set_data(self._epochs, self._losses)
        self._line_f1.set_data(self._epochs, self._val_f1)
        self._line_lr.set_data(self._epochs, self._lr)

        ax_l.relim()
        ax_l.autoscale_view()
        ax_r.relim()
        ax_r.autoscale_view()
        ax_f.set_xlim(ax_l.get_xlim())
        ax_f.set_ylim(0.0, 1.0)

        try:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
        except Exception as exc:
            logger.warning("Live plot draw failed (%s): %s", type(exc).__name__, exc)
        try:
            self._plt.pause(0.05)
        except Exception:
            pass

    def close(self) -> None:
        """Close the figure and turn interactive mode off."""
        if self._plt is None or self._fig is None:
            return
        try:
            self._plt.close(self._fig)
        finally:
            self._fig = None
            self._axes = None
            self._line_loss = self._line_f1 = self._line_lr = None
