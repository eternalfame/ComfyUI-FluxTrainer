import gc
import os

import torch

import logging

logger = logging.getLogger(__name__)


def _host_empty_cache():
    fn = getattr(torch._C, "_host_emptyCache", None)
    if fn is not None:
        try:
            fn()
        except Exception:
            pass


KEEP_ATTRS = {"loss_recorder", "metadata", "args", "global_step", "current_epoch",
              "num_train_epochs", "epoch_to_start", "initial_step", "save_dtype"}


def release_trainer(trainer, training_loop=None):
    if os.environ.get("FLUXTRAINER_KEEP_TRAINER"):
        return
    try:
        accelerator = getattr(trainer, "accelerator", None)

        for name in ("unet", "network"):
            obj = getattr(trainer, name, None)
            if obj is None:
                continue
            try:
                if accelerator is not None:
                    obj = accelerator.unwrap_model(obj)
                obj.to("cpu")
            except Exception as e:
                logger.warning(f"release_trainer: could not move {name} to cpu: {e}")

        if accelerator is not None:
            try:
                accelerator.free_memory()
            except Exception as e:
                logger.warning(f"release_trainer: accelerator.free_memory failed: {e}")

        opt = getattr(trainer, "optimizer", None)
        if opt is not None:
            try:
                opt.state.clear()
            except Exception:
                pass

        for key in list(vars(trainer).keys()):
            if key not in KEEP_ATTRS:
                setattr(trainer, key, None)
        for cell in (getattr(training_loop, "__closure__", None) or ()):
            try:
                cell.cell_contents = None
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"release_trainer failed: {e}")

    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _host_empty_cache()
