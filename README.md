# Trans-perf

Course project for the course LOG6309E - Intelligent DevOps of Large-Scale Software Systems given by professor Heng Li at Polytechnique Montreal. In this project, we investigate the usefulness of transformer models for configurable software performance prediciton and compare the results with state-of-the-art methods.

## Installation 

```
conda create -n transperf python=3.8
conda activate transperf
pip install -r requirements.txt
```

## Reproduction

There are two versions of our model, one in the `main` branch, and one in the `seq` branch. The results are presented in the folder `plots/`.

To run a simple experiment:
```
python src/train_example.py <DATASET_NAME>
```

To run the full experiments that reproduce our results:
```
./run-exps.sh
```
