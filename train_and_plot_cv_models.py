
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from chemprop.args import TrainArgs, PredictArgs
from chemprop.train import cross_validate, run_training, make_predictions
from chemarr.utils import plot_hexbin
from chemarr.make_balanced_train import make_balanced_data


PATH_CHEM = os.getcwd()
DATADIR = f"{PATH_CHEM}/data/cross_val_data"
TYPE = "arr"
MODELDIR = f"{PATH_CHEM}/models"

def make_training_predictions(data_path, model_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for ROUNDNUM in range(1,5): 
        for TESTNUM in range(2,12):  

            TRAIN=f"{data_path}/s_train_{TESTNUM}.csv"
            TRAINFEATS=f"{data_path}/f_train_{TESTNUM}.csv"

            VAL=f"{data_path}/s_cv_{TESTNUM}.csv"
            VALFEATS=f"{data_path}/f_cv_{TESTNUM}.csv"

            TEST=f"{data_path}/s_test_{TESTNUM}.csv"
            TESTFEATS=f"{data_path}/f_test_{TESTNUM}.csv"

            PREDS=f"{data_path}/{TYPE}/preds_{TESTNUM}_{ROUNDNUM}.csv"
            SAVEDIR=f"{model_path}/checkpoints/check{ROUNDNUM}_{TESTNUM}"

            argument = [
                "--data_path",f"{TRAIN}",
                "--features_path", f"{TRAINFEATS}",
                "--separate_val_path", f"{VAL}",
                "--separate_val_features_path", f"{VALFEATS}",
                "--separate_test_path", f"{TEST}",
                "--separate_test_features_path", f"{TESTFEATS}",
                "--save_dir", f"{SAVEDIR}",
                "--dataset_type", "regression",
                "--metric", "mae",
                "--arr_vtf", "arr",
                "--quiet",
                "--depth", "3",
                "--dropout", "0.15",
                "--ffn_num_layers", "3",
                "--hidden_size", "2300",
                "--batch_size", "100",
                "--gpu", "0",
                "--pytorch_seed", "3",
                "--epochs", "12",
            ]

            train_args = TrainArgs().parse_args(argument)
            
            # TRAIN THE MODEL
            cross_validate(args=train_args, train_func=run_training)
            
            TRAIN_FULL=f"{data_path}/s_full.csv"
            TRAINFEATS_FULL=f"{data_path}/f_full.csv"
            PREDS=f"{data_path}/preds/preds_screen_{ROUNDNUM}_{TESTNUM}.csv"

            pred_args = [
                "--test_path", f"{TRAIN_FULL}",
                "--features_path", f"{TRAINFEATS_FULL}",
                "--checkpoint_dir", f"{SAVEDIR}",
                "--arr_vtf", "arr", 
                "--preds_path", f"{PREDS}",
            ]

            make_predictions(args=PredictArgs().parse_args(pred_args))

def plot_parity(data_path):
    preds_path = f'{data_path}/preds'
    paths_df = os.listdir(preds_path)

    df_ref = pd.read_csv(f"{preds_path}/{paths_df[0]}")
    conductivities = [df_ref.conductivity.values]
    smiles = df_ref.smiles.values

    for i in paths_df[1:]:
        df = pd.read_csv(f"{preds_path}/{i}")
        if (df.smiles.values == smiles).all():
            conductivities.append(df.conductivity.values)
        else:
            raise ValueError(f"The smiles of {paths_df[0]} doesn't line up with {paths_df[1]}")
    pred_cond = np.array(conductivities).mean(axis=0)

    TRAIN=f"{data_path}/s_full.csv"
    df_true = pd.read_csv(TRAIN)

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax, hb = plot_hexbin(df_true.conductivity.values,pred_cond, ax, 'linear')
    ax.set_xlabel('Target Ionic Conductivity (S/cm)', fontdict={'size':20})
    ax.set_ylabel('Predicted Ionic Conductivity (S/cm)', fontdict={'size':20})
    ax.set_title('ChemArr conductivity parity plot',fontdict={'size':26})
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Number of points',fontdict={'size':18})
    plt.tick_params(axis='both', which='major', labelsize=16)
    cb.ax.tick_params(labelsize=16)
    plt.savefig(f'{data_path}/conductivity_parity_plot.png')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing input parameters for cross validation training')
    parser.add_argument('--make_data', choices=['true', 'false'], default='false', 
                        help='Determines whether the data should be generated or not')
    parser.add_argument('--train_predict', choices=['true', 'false'], default='false',
                        help='Should the models be trained or not (takes couple of hours)')
    parser.add_argument('--plot_parity', choices=['true', 'false'], default='false',
                        help='Should the data be plotted, works only when data is made and predicted')
    parser.parse_args()
    
    if parser.make_data == "true":
        print("Creating the cross validation data files for training!")
        make_balanced_data(DATADIR, f'{PATH_CHEM}/data/clean_train_data.csv')
    if parser.train_predict == "true":
        print("Training loop begins!")
        make_training_predictions(DATADIR,MODELDIR)
    if parser.plot_parity == "true":
        print("Plotting results")
        plot_parity(DATADIR)
        
    
    