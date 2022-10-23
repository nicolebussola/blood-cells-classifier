<div align="center">

# Blood cell classification

##  Quickstart

```bash
# clone project
git clone https://github.com/nicolebussola/blood-cells-classifier
cd blood-cells-classifier

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install requirements
pip install -r requirements.txt
```

##  Workflow

### Prepare data


### Training
Run training with default experiment config "configs/experiment/cell_classification.yaml"

```bash
python src/train.py
```


 and write results in csv file:
```bash
python src/train.py logger=csv hparams_search=blood_cells_optuna
```

If available, model can be trained on gpu/mps by running
```bash
python src/train.py logger=csv hparams_search=blood_cells_optuna trainer.accelerator=mps
```
