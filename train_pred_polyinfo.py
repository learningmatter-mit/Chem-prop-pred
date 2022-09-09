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

from chemproppred.make_screen_data import make_screening_data
from chemproppred.screen_polys import screen_poly


PATH_CHEM = os.getcwd()
DATADIR = f"{PATH_CHEM}/data/cross_val_data"
TYPE = "arr"
MODELDIR = f"{PATH_CHEM}/models/screen"
SAVEPATH = f"{PATH_CHEM}/data/polyinfo"
PREDS_PATH = f"{DATADIR}/preds_screen"

def train_and_predict(data_path, model_path, preds_path, gpu=False):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    TRAIN = f"{data_path}/s_full.csv"
    TRAINFEATS = f"{data_path}/f_full.csv"

    TEST = f"{data_path}/s_screen.csv"
    TESTFEATS = f"{data_path}/f_screen.csv"

    PREDS=  f"{preds_path}/preds_screen.csv"
    SAVEDIR = model_path

    #train chemprop model
    argument = [
    "--data_path", f"{TRAIN}",
    "--features_path", f"{TRAINFEATS}",
    "--save_dir", f"{SAVEDIR}",
    "--dataset_type", "regression",
    "--split_size", "0.95", "0.04", "0.01",
    "--metric", "mae",
    "--arr_vtf","arr",
    "--quiet",
    "--depth", "2",
    "--dropout", "0",
    "--ffn_num_layers", "3",
    "--hidden_size", "2400",
    "--epochs", "25",
    "--pytorch_seed","5",
    ]
    
    if gpu:
        argument.append("--gpu")
        argument.append("0")
    else:
        argument.append("--no_cuda")

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
                        help='Train the model on all the data and predict on polyinfo data')
    parser.add_argument('--polyinfo_datafiles', choices=['true', 'false'], default='false',
                        help='Generate easily viewable files for the polyinfo data from the predicitons')
    parser.add_argument('--gpu', choices=['true', 'false'], default='false',
                        help='The model is trained on cuda enabled GPU, default false - training on CPU')
    args = parser.parse_args()
    
    if args.make_data == "true":
        print("Creating the cross validation data files for training!")
        make_screening_data(DATADIR, f'{PATH_CHEM}/data/polyinfo_5salts_4conc.csv')
    if args.train_predict == "true":
        print("Training loop begins!")
        train_and_predict(DATADIR, MODELDIR, PREDS_PATH, args.gpu) 
    if args.polyinfo_datafiles == "true":
        screen_poly(DATADIR, SAVEPATH, PREDS_PATH)
        