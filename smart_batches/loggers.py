"""Run logging: JSONL on disk (source of truth) plus optional Weights & Biases."""

import json
import os
from pathlib import Path


def _wandb_logged_in() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    try:
        return "api.wandb.ai" in netrc.read_text()
    except OSError:
        return False


class RunLogger:
    _offline_notice_shown = False

    def __init__(self, out_dir: Path, config_dict: dict, wandb_cfg, run_name: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
        self._file = open(self.out_dir / "metrics.jsonl", "a", buffering=1)
        self._wandb = None

        if wandb_cfg.enabled and wandb_cfg.mode != "disabled":
            mode = wandb_cfg.mode
            if mode == "auto":
                mode = "online" if _wandb_logged_in() else "offline"
                if mode == "offline" and not RunLogger._offline_notice_shown:
                    print("[wandb] no login found -> logging offline "
                          "(run `wandb login` then `wandb sync` to upload)")
                    RunLogger._offline_notice_shown = True
            try:
                import wandb

                if wandb.run is not None:
                    # inside a `wandb agent` sweep trial (or a caller-initialized
                    # run): adopt it instead of opening a second run
                    self._wandb = wandb.run
                    self._wandb.config.update(config_dict, allow_val_change=True)
                    return
                self._wandb = wandb.init(
                    entity=wandb_cfg.entity,
                    project=wandb_cfg.project,
                    name=run_name,
                    group=wandb_cfg.group or None,
                    config=config_dict,
                    mode=mode,
                    dir=str(self.out_dir),
                )
            except Exception as exc:  # never let tracking kill a run
                print(f"[wandb] init failed ({exc!r}); continuing without wandb")
                self._wandb = None

    def log(self, step: int, metrics: dict) -> None:
        self._file.write(json.dumps({"step": step, **metrics}) + "\n")
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self, summary: dict | None = None) -> None:
        if summary:
            (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if self._wandb is not None:
            if summary:
                for k, v in summary.items():
                    self._wandb.summary[k] = v
            self._wandb.finish()
            self._wandb = None
        if not self._file.closed:
            self._file.close()
