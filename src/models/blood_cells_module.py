import os
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import matthews_corrcoef as mcc
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassMatthewsCorrCoef

ACCELERATOR = os.environ.get("accelerator", "cpu")


class _Metrics:
    @classmethod
    def function(cls, accelerator, num_classes):
        if accelerator == "cpu":
            return mcc
        elif accelerator in {"mps", "gpu"}:
            return MulticlassMatthewsCorrCoef(num_classes=num_classes)
        else:
            raise NotImplementedError(f"{accelerator} not recognized")


class BloodCellsLitModule(LightningModule):
    """LightningModule for Blood Cell classification."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metrics
        self.metrics_fn = _Metrics.function(ACCELERATOR, num_classes=4)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_mcc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        self.val_mcc_best.reset()

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.train_mcc = self.metrics_fn(preds, targets)
        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/mcc", self.train_mcc, on_step=False, on_epoch=True, prog_bar=True
        )
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.train_loss(loss)
        # update and log metrics
        self.val_loss(loss)
        self.val_mcc = self.metrics_fn(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mcc", self.val_mcc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        mcc = self.val_mcc if ACCELERATOR == "cpu" else self.metrics_fn.compute()
        self.val_mcc_best(mcc)
        self.log("val/mcc_best", self.val_mcc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_mcc = self.metrics_fn(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/mcc", self.test_mcc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, targets = self.step(batch)
        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "blood_cells.yaml")
    _ = hydra.utils.instantiate(cfg)
