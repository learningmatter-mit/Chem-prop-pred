import argparse
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, AllChem, Descriptors
from rdkit import DataStructs
from rdkit.Chem import Descriptors

import chemprop
from chemprop.args import TrainArgs, PredictArgs
from chemprop.train import cross_validate, run_training, make_predictions

from chemarr.make_screen_data import make_screening_data


PATH_CHEM = os.getcwd()
DATADIR = f"{PATH_CHEM}/data/polyinfo_data"
TYPE = "arr"
MODELDIR = f"{PATH_CHEM}/models"

def train_and_predict(data_path, model_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    TRAIN = f"{data_path}/s_full.csv"
    TRAINFEATS = f"{data_path}/f_full.csv"

    TEST = f"{data_path}/s_screen.csv"
    TESTFEATS = f"{data_path}/f_screen.csv"

    PREDS=  f"{data_path}/preds_screen.csv"
    SAVEDIR = model_path

    #train chemprop model
    argument = [
    "--data_path", f"{TRAIN}",
    "--features_path", f"{TRAINFEATS}",
    "--save_dir", f"{SAVEDIR}",
    "--dataset_type", "regression",
    "--split_size", "0.89", "0.1", "0.01",
    "--metric", "mae",
    "--arr_vtf","arr",
    "--quiet",
    "--depth", "2",
    "--dropout", "0",
    "--ffn_num_layers", "3",
    "--hidden_size", "2400",
    "--gpu", "0",
    "--epochs", "25",
    "--pytorch_seed","5",
    ]

    train_args = TrainArgs().parse_args(argument)

    cross_validate(args=train_args, train_func=run_training)

    pred_args = [
        "--test_path", f"{TEST}",
        "--features_path", f"{TESTFEATS}",
        "--checkpoint_dir", f"{SAVEDIR}",
        "--arr_vtf", "arr", 
        "--preds_path", f"{PREDS}",
    ]

    make_predictions(args=PredictArgs().parse_args(pred_args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing input parameters for cross validation training')
    parser.add_argument('--make_data', choices=['true', 'false'], default='false', 
                        help='Determines whether the data should be generated or not')
    parser.add_argument('--train_predict', choices=['true', 'false'], default='false',
                        help='Should the models be trained or not (takes couple of hours)')
    parser.parse_args()
    
    if parser.make_data == "true":
        print("Creating the cross validation data files for training!")
        make_screening_data(DATADIR, f'{PATH_CHEM}/data/clean_train_data.csv')
    if parser.train_predict == "true":
        print("Training loop begins!")
        train_and_predict(DATADIR, MODELDIR)
        