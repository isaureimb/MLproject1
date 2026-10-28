# Project 1 - Heart disease prediction

Machine Learning course (CS-433), EPFL. We predict whether a person has heart disease (MICHD) from survey data (BRFSS 2015). Only NumPy and matplotlib are used.
This repository contains the code for our first project within the course CS-433 - Machine Learning @ EPFL. 
The goal was to develop a machine learning model that estimates the risk of developing **MICHD** (coronary heart disease or myocardial infarction) from the health-related survey answers of the Behavioral Risk Factor Surveillance System (BRFSS).

All models were implemented from scratch using only NumPy: least squares, ridge regression, and (regularised) logistic regression. 

## Setup

```
conda create -n ml_project1 python=3.11 numpy matplotlib
conda activate ml_project1
```

## Data

Download x_train.csv, y_train.csv and x_test.csv from AIcrowd and put them in a folder called `data/` at the root of the project. Do not rename them. The folder is in .gitignore because the files are big.

## Run

```
python run.ipynb
```

This trains the final model and creates the csv file we submitted to AIcrowd. The random seed is fixed, so you get the same predictions every time.

## Files

- `implementations.py`: the 6 functions asked in the project description
- `helpers.py`: loading the data and writing the submission file (given)
- `preprocessing.py`: data exploration and cleaning (`preprocess`)
- `metrics.py`: accuracy, precision, recall and F1
- `cross_validation.py`: k-fold split and cross-validation
- `model_training.py`: baselines, threshold, class weights, grid search
- `experiments.ipynb` : test of different models accuracy
- `run.ipynb`: final predictions

## Authors

Isaure Imberton, Lila Reinhardt, Laetitia Espinosa