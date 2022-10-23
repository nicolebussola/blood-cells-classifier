import os
from pathlib import Path

import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["logger=[]"]

TEST_DIR = os.path.dirname(Path(__file__).absolute())

train_data_dir = os.path.join(TEST_DIR, "fixtures/train_cells")
test_data_dir = os.path.join(TEST_DIR, "fixtures/test_cells")
valid_data_dir = os.path.join(TEST_DIR, "fixtures/valid_cells")

dir_conf = [
    f"++datamodule.train_data_dir={train_data_dir}",
    f"++datamodule.test_data_dir={test_data_dir}",
    f"++datamodule.valid_data_dir={valid_data_dir}",
]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path):
    """Test running all available experiment configs with fast_dev_run=True."""
    command = (
        [
            startfile,
            "-m",
            "experiment=glob(*)",
            "hydra.sweep.dir=" + str(tmp_path),
            "++trainer.fast_dev_run=true",
        ]
        + overrides
        + dir_conf
    )
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path):
    """Test default hydra sweep."""
    command = (
        [
            startfile,
            "-m",
            "hydra.sweep.dir=" + str(tmp_path),
            "model.optimizer.lr=0.005,0.01",
            "++trainer.fast_dev_run=true",
        ]
        + overrides
        + dir_conf
    )

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path):
    """Test default hydra sweep with ddp sim."""
    command = (
        [
            startfile,
            "-m",
            "hydra.sweep.dir=" + str(tmp_path),
            "trainer=ddp_sim",
            "trainer.max_epochs=3",
            "+trainer.limit_train_batches=1",
            "+trainer.limit_val_batches=1",
            "+trainer.limit_test_batches=1",
            "model.optimizer.lr=0.005,0.01,0.02",
        ]
        + overrides
        + dir_conf
    )
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(tmp_path):
    """Test optuna sweep."""
    command = [
        startfile,
        "-m",
        "hparams_search=blood_cells_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "++trainer.fast_dev_run=true",
        f"++datamodule.train_data_dir={train_data_dir}",
        f"++datamodule.test_data_dir={test_data_dir}",
        f"++datamodule.valid_data_dir={valid_data_dir}",
    ] + overrides
    run_sh_command(command)
