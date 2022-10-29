import time

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from termcolor import cprint
import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EfficiencyCallbackBC(Callback):
    """
    EfficiencyCallback backward-compatible to pytorch-lightning v0.7.1
    https://pytorch-lightning.readthedocs.io/en/0.7.1/callbacks.html
    """
    time_init_start = None
    # time_fit_start = None

    time_train_epoch_start = None
    interval_train_epochs = []
    num_batches_train_epochs = []

    time_valid_epoch_start = None
    interval_valid_epochs = []
    num_batches_valid_epochs = []

    total_epoch_count = 0
    train_batch_count = 0
    valid_batch_count = 0

    stage = None

    def __init__(self, stop_epochs=5):
        super(EfficiencyCallbackBC, self).__init__()
        self.stop_epochs = stop_epochs

    def on_init_start(self, trainer):
        self.time_init_start = time.time()
        cprint("\nInit start", "green")

    def on_epoch_start(self, trainer, pl_module):
        self.stage = "Training"
        self.on_train_epoch_start(trainer, pl_module)

    def on_batch_start(self, trainer, pl_module):
        assert self.stage == "Training"
        self.on_train_batch_start(trainer, pl_module, None, None)

    def on_validation_start(self, trainer: Trainer, pl_module):
        self.stage = "Validation"
        self.on_validation_epoch_start(trainer, pl_module)

        # Replacing on_validation_batch_start
        self.valid_batch_count = trainer.num_val_batches

    def on_validation_end(self, trainer, pl_module):
        # on_validation_epoch_end
        self.on_validation_epoch_end(trainer, pl_module)

        if self.total_epoch_count == self.stop_epochs:
            # Time for throughput & latency
            total_end = time.time()
            dt_init_start = total_end - self.time_init_start
            # dt_fit_start = total_end - self.time_fit_start
            cprint("\n------------------", "green")

            cprint(f"- init_start ~ : {dt_init_start}", "green")
            # cprint(f"- fit_start ~ : {dt_fit_start}", "green")

            cprint("------------------", "yellow")
            total_train_time = sum(self.interval_train_epochs)
            dt_train_epoch = np.mean(self.interval_train_epochs)
            dt_train_batch = sum(self.interval_train_epochs) / sum(self.num_batches_train_epochs)
            total_valid_time = sum(self.interval_valid_epochs)
            dt_valid_epoch = np.mean(self.interval_valid_epochs)
            dt_valid_batch = sum(self.interval_valid_epochs) / sum(self.num_batches_valid_epochs)

            cprint(f"- total_epoch: {self.total_epoch_count}", "yellow")

            cprint(f"- total_train_time: {total_train_time}", "yellow")
            cprint(f"- time / train_epoch: {dt_train_epoch}", "yellow")
            cprint(f"- time / train_batch: {dt_train_batch}", "yellow")

            cprint(f"- total_valid_time: {total_valid_time}", "yellow")
            cprint(f"- time / valid_epoch: {dt_valid_epoch}", "yellow")
            cprint(f"- time / valid_batch: {dt_valid_batch}", "yellow")

            # Memory and parameters
            num_parameters = count_parameters(pl_module)
            # torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved
            max_memory_reserved = torch.cuda.memory_cached()
            max_memory_allocated = torch.cuda.max_memory_allocated()
            cprint(f"\nSummary as Table --- {pl_module.h.subname}", "yellow")
            print("\t".join(str(t) for t in [
                num_parameters, max_memory_reserved, max_memory_allocated,
                dt_init_start, self.total_epoch_count,
                total_train_time, dt_train_epoch, dt_train_batch,
                total_valid_time, dt_valid_epoch, dt_valid_batch,
            ]))
            exit()

    # Belows are not compatible to old pl.

    def on_fit_start(self, trainer, pl_module):
        self.time_fit_start = time.time()
        cprint("\nFit start", "green")

    def on_train_epoch_start(self, trainer, pl_module):
        self.total_epoch_count += 1
        self.train_batch_count = 0
        self.time_train_epoch_start = time.time()
        cprint(f"\nTraining epoch start {self.total_epoch_count}", "green")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused=0):
        self.train_batch_count += 1
        cprint(f"\nTraining batch incremented: {self.train_batch_count}", "green")

    def on_train_all_batches_end(self, trainer, pl_module):
        self.interval_train_epochs.append(time.time() - self.time_train_epoch_start)
        self.num_batches_train_epochs.append(self.train_batch_count)
        cprint("\nTraining all batches end", "green")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.on_train_all_batches_end(trainer, pl_module)
        self.valid_batch_count = 0
        self.time_valid_epoch_start = time.time()
        cprint("\nValidation epoch start", "green")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.valid_batch_count += 1
        cprint(f"\nValidation batch incremented: {self.valid_batch_count}", "green")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.interval_valid_epochs.append(time.time() - self.time_valid_epoch_start)
        self.num_batches_valid_epochs.append(self.valid_batch_count)
        cprint("\nValidation epoch end", "green")
