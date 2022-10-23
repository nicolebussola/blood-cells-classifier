<div align="center">

# Blood cell classification

</div>

##  Quickstart

```bash
# clone project
git clone https://github.com/nicolebussola/blood-cells-classifier
cd blood-cells-classifier

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install requirements
pip install -r requirements.txt
```

Installing project as a package:
```bash
pip install -e .
```

##  Workflow

### Data exploration and preparation
Available annotations and dataset statistics can be explored using the notebook  `notebooks/data_exploration.ipynb`.

Run `scripts/data_preparation.py` to crop cells from original images and create the training, validation, and test set stratified by cell type. The script requires the path of the folder containing original images and metadata to be passed as parameter.

```bash
python scripts/data_preparation -p <data_source_path>
```


### Training
Run training with default experiment config `configs/experiment/cell_classification.yaml`:

```bash
python src/train.py
```

To run the model for a fewer number of epochs (default n_epochs=50), add:
```bash
python src/train.py trainer.max_epochs=5
```


If available, model can be trained on gpu/mps by adding:
```bash
python src/train.py trainer.accelerator=mps
```

[optional] log results on [wandb](https://wandb.ai/site) and run hyperparameter optimization:
```bash
python src/train.py logger=wandb hparams_search=blood_cells_optuna
```


### Testing

Run model evaluation on the test set and save predictions:
```bash
python src/eval.py
```

Performance on the test set can be visualized by running the `notebooks/visualize_test_predictions.ipynb` notebook
