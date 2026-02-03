# main.py
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from transcription.module import D3RM
from transcription.datamodule import MAESTRO_V3_DataModule

# simple demo classes for your convenience
import argparse, os, glob, datetime
from typing import Optional
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from termcolor import colored
import torch
import wandb

torch.backends.cudnn.benchmark = True

class D3RMCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:

        parser.add_argument( "-d", "--debug", type=bool, default=False, help="enable post-mortem debugging",)
        parser.add_argument("--wandb", type=bool, default=False, help="wandb online/offline",)

    def _infer_run_id_from_ckpt_path(self, ckpt_path: Optional[str]) -> str:
        """Infer run id (date string) from checkpoint path or current time."""
        if ckpt_path:
            # Expected: ./checkpoints/<RUN_ID>/<file>.ckpt
            try:
                return os.path.basename(os.path.dirname(ckpt_path))
            except Exception:
                pass
        return datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    def _get_ckpt_path_for_stage(self, stage: str) -> Optional[str]:
        """Best-effort: LightningCLI stores ckpt_path under different namespaces per subcommand."""
        # Prefer stage-specific (test / fit) if present, fall back to common locations.
        for attr_path in (
            (stage, "ckpt_path"),
            ("fit", "ckpt_path"),
            ("ckpt_path",),
        ):
            cfg = self.config
            ok = True
            for key in attr_path:
                if hasattr(cfg, key):
                    cfg = getattr(cfg, key)
                else:
                    ok = False
                    break
            if ok and cfg:
                return str(cfg)
        return None

    def before_fit(self):
        if not self.config.fit.ckpt_path: # if not resuming from checkpoint
            self.now = id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        elif self.config.fit.ckpt_path: # resuming from checkpoint
            ckpt_date = os.path.basename(os.path.dirname(self.config.fit.ckpt_path))
            print(colored("Continue training from checkpoint: ", "red", attrs=['bold']), ckpt_date)
            self.now = id = ckpt_date
        # Logging
        wandb_logger = WandbLogger(save_dir=f"./logs/{self.now}",
                                   name=self.now,
                                   project="D3RM",
                                   offline=(not self.config.fit.wandb),
                                   id=id)

        # Model checkpoint (automatically called after validation)
        model_checkpoint_callback = ModelCheckpoint(
            dirpath=f'./checkpoints/{self.now}',
            monitor='metric_note_with_offsets_f1',
            mode='max',
            save_top_k=7,
            save_last=True,
            verbose=True,
            filename='{step:07}-{metric_note_with_offsets_f1:.4f}') # python recognized '/', '-' as '_'

        self.trainer.logger = wandb_logger
        self.trainer.callbacks.append(model_checkpoint_callback)
        self.config.fit.model.test_save_path = f"./results/{self.now}"
        print(colored("Test results will be saved in: ", "green", attrs=['bold']), self.config.fit.model.test_save_path)

    def before_test(self):
        """
        Ensure test predictions are saved under a run-specific folder.

        - If user explicitly set `model.test_save_path` (e.g. via -c YAML or CLI override),
          we keep it unless it is the generic './results/' default.
        - Otherwise, default to './results/<RUN_ID>' where RUN_ID is derived from ckpt path.
        """
        ckpt_path = self._get_ckpt_path_for_stage("test")
        run_id = self._infer_run_id_from_ckpt_path(ckpt_path)

        # Find the model config namespace for this stage.
        cfg_model = None
        if hasattr(self.config, "test") and hasattr(self.config.test, "model"):
            cfg_model = self.config.test.model
        elif hasattr(self.config, "fit") and hasattr(self.config.fit, "model"):
            cfg_model = self.config.fit.model

        existing = getattr(cfg_model, "test_save_path", None) if cfg_model else None
        existing_str = str(existing) if existing is not None else ""
        is_generic = existing_str.strip().rstrip("/") in ("", ".", "results", "./results")

        save_path = existing_str if (existing_str and not is_generic) else f"./results/{run_id}"

        # Apply to both config and instantiated model (runtime).
        if cfg_model is not None and hasattr(cfg_model, "test_save_path"):
            cfg_model.test_save_path = save_path
        if hasattr(self, "model") and hasattr(self.model, "test_save_path"):
            self.model.test_save_path = save_path

        os.makedirs(save_path, exist_ok=True)
        print(colored("Test results will be saved in: ", "green", attrs=['bold']), save_path)

        
    # def before_instantiate_classes(self) -> None:
    

def cli_main():
    cli = D3RMCLI(D3RM, MAESTRO_V3_DataModule,
                #   save_config_kwargs={"overwrite": True}, #  save_config_callback=None # when using wandb, saving config leads to conflicts.
                  )

if __name__ == "__main__":
    cli_main()