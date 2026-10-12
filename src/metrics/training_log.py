"""
Training run logger for Carepath.

Writes two files to reports/training_runs/ on every training run:
  - <timestamp>.json   machine-readable full metrics
  - <timestamp>.md     human-readable Markdown report

Both files share the same timestamp prefix so they're easy to pair up.
The JSON is append-safe: you can load every *.json in the folder and
build a history table without any database.

Usage (from train_all.py):
    from src.metrics.training_log import TrainingRun

    run = TrainingRun()
    run.set_track_a(...)
    run.set_track_b(...)
    run.save()          # writes both files, prints paths
"""

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── output directory (relative to project root) ───────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = _ROOT / "reports" / "training_runs"


class TrainingRun:
    """Accumulates metrics from one training run and writes them to disk."""

    def __init__(self):
        self.started_at: str = datetime.now(timezone.utc).isoformat()
        self._ts: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.quick_mode: bool = "--quick" in sys.argv
        self.track_a: dict = {}
        self.track_b: dict = {}

    # ── setters called by train_all.py ────────────────────────────────────────

    def set_track_a(
        self,
        *,
        n_train: int,
        n_test: int,
        n_features: int,
        models: dict,          # {model_name: {"auc": float, "f1": float, "per_target": {...}}}
        winner: str,
        winner_auc: float,
        winner_f1: float,
        registry_version: int,
    ):
        """Record Track A (NHANES complication risk) results."""
        self.track_a = {
            "dataset": "NHANES",
            "n_train": n_train,
            "n_test": n_test,
            "n_features": n_features,
            "models": models,
            "winner": winner,
            "winner_auc": round(winner_auc, 4),
            "winner_f1": round(winner_f1, 4),
            "registry_version": registry_version,
        }

    def set_track_b(
        self,
        *,
        n_real_patients: int,
        n_real_windows: int,
        n_total_windows: int,
        gap_max_hours: float,
        window_size: int,
        seeds: int,
        blend_alpha: float,
        mae_results: dict,         # {model_name: {"pooled": float, "per_patient": float}}
        winner: str,
        winner_pooled_mae: float,
        naive_mae: float,
        improvement_mae: float,
        improvement_pct: float,
        registry_version: int,
        pretrain_used: bool,
    ):
        """Record Track B (MIMIC-IV glucose forecast) results."""
        self.track_b = {
            "dataset": "MIMIC-IV Demo",
            "n_real_patients": n_real_patients,
            "n_real_windows": n_real_windows,
            "n_total_windows": n_total_windows,
            "gap_filter_hours": gap_max_hours,
            "window_size": window_size,
            "seeds": seeds,
            "blend_alpha": round(blend_alpha, 3),
            "pretrain_used": pretrain_used,
            "models": mae_results,
            "winner": winner,
            "winner_pooled_mae": round(winner_pooled_mae, 2),
            "naive_pooled_mae": round(naive_mae, 2),
            "improvement_mae": round(improvement_mae, 2),
            "improvement_pct": round(improvement_pct, 1),
            "registry_version": registry_version,
        }

    # ── write ─────────────────────────────────────────────────────────────────

    def save(self) -> tuple[Path, Path]:
        """Write JSON + Markdown reports. Returns (json_path, md_path)."""
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = RUNS_DIR / f"{self._ts}.json"
        md_path   = RUNS_DIR / f"{self._ts}.md"

        payload = self._to_dict()
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(self._to_markdown(payload), encoding="utf-8")

        print(f"\n  [Metrics] JSON  -> {json_path}")
        print(f"  [Metrics] Markdown -> {md_path}")
        return json_path, md_path

    # ── serialisation helpers ─────────────────────────────────────────────────

    def _to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "quick_mode": self.quick_mode,
            "python": sys.version.split()[0],
            "platform": platform.system(),
            "track_a": self.track_a,
            "track_b": self.track_b,
        }

    def _to_markdown(self, d: dict) -> str:
        ts_display = datetime.fromisoformat(d["started_at"]).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"# Training Run — {ts_display}",
            "",
            f"**Quick mode:** {'yes' if d['quick_mode'] else 'no'}  ",
            f"**Python:** {d['python']}  ",
            f"**Platform:** {d['platform']}",
            "",
        ]

        # ── Track A ──────────────────────────────────────────────────────────
        a = d.get("track_a", {})
        if a:
            lines += [
                "## Track A — NHANES Complication Risk",
                "",
                f"| | Value |",
                f"|---|---|",
                f"| Dataset | {a['dataset']} |",
                f"| Train rows | {a['n_train']:,} |",
                f"| Test rows | {a['n_test']:,} |",
                f"| Features | {a['n_features']} |",
                f"| **Winner** | **{a['winner']}** |",
                f"| Winner AUC | {a['winner_auc']} |",
                f"| Winner F1 | {a['winner_f1']} |",
                f"| Registry version | v{a['registry_version']} |",
                "",
                "### Per-model results",
                "",
                "| Model | Mean AUC | Mean F1 |",
                "|---|---|---|",
            ]
            for mname, mvals in a.get("models", {}).items():
                winner_mark = " ✓" if mname == a["winner"] else ""
                lines.append(
                    f"| {mname}{winner_mark} | {mvals['auc']:.3f} | {mvals['f1']:.3f} |"
                )

            # per-target breakdown
            winner_model = a.get("models", {}).get(a.get("winner"), {})
            per_target = winner_model.get("per_target", {})
            if per_target:
                lines += [
                    "",
                    f"### Winner ({a['winner']}) — per-target breakdown",
                    "",
                    "| Target | AUC | F1 |",
                    "|---|---|---|",
                ]
                for tgt, tv in per_target.items():
                    lines.append(f"| {tgt} | {tv['auc']:.3f} | {tv['f1']:.3f} |")
            lines.append("")

        # ── Track B ──────────────────────────────────────────────────────────
        b = d.get("track_b", {})
        if b:
            lines += [
                "## Track B — MIMIC-IV Glucose Forecast",
                "",
                f"| | Value |",
                f"|---|---|",
                f"| Dataset | {b['dataset']} |",
                f"| Real patients | {b['n_real_patients']} |",
                f"| Real windows (gap < {b['gap_filter_hours']}h) | {b['n_real_windows']} |",
                f"| Total windows (incl. augmentation) | {b['n_total_windows']} |",
                f"| Window size | {b['window_size']} |",
                f"| LSTM seeds | {b['seeds']} |",
                f"| Blend alpha (LSTM weight) | {b['blend_alpha']} |",
                f"| Synthetic pretrain | {'yes' if b['pretrain_used'] else 'no'} |",
                f"| **Winner** | **{b['winner']}** |",
                f"| Winner pooled MAE | {b['winner_pooled_mae']} mg/dL |",
                f"| Naive baseline MAE | {b['naive_pooled_mae']} mg/dL |",
                f"| Improvement over naive | {b['improvement_mae']} mg/dL ({b['improvement_pct']}%) |",
                f"| Registry version | v{b['registry_version']} |",
                "",
                "### LOPO-CV results (pooled MAE, mg/dL ↓)",
                "",
                "| Model | Pooled MAE | Per-patient MAE |",
                "|---|---|---|",
            ]
            for mname, mv in b.get("models", {}).items():
                winner_mark = " ✓" if mname == b["winner"] else ""
                lines.append(
                    f"| {mname}{winner_mark} | {mv['pooled']:.1f} | {mv['per_patient']:.1f} |"
                )
            lines.append("")

        lines += [
            "---",
            f"*Generated by Carepath training pipeline · {d['finished_at'][:19]} UTC*",
        ]
        return "\n".join(lines) + "\n"
