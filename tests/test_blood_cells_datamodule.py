import os
from pathlib import Path

import pytest
import torch

from src.datamodules.blood_cells_datamodule import BloodCellsDataModule

TEST_DIR = os.path.dirname(Path(__file__).absolute())


@pytest.mark.parametrize("batch_size", [2, 10])
def test_blood_cells_datamodule(batch_size):
    test_data_dir = os.path.join(TEST_DIR, "fixtures/test_cells")
    train_data_dir = os.path.join(TEST_DIR, "fixtures/train_cells")
    valid_data_dir = os.path.join(TEST_DIR, "fixtures/valid_cells")

    dm = BloodCellsDataModule(
        test_data_dir=test_data_dir,
        train_data_dir=train_data_dir,
        valid_data_dir=valid_data_dir,
        batch_size=batch_size,
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 30

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
