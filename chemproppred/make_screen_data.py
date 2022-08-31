import os
import pandas as pd
from chemproppred.utils import *

def make_screening_data(data_folder, data_path):
    #read in screening data
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    df = pd.read_csv(data_path)

    df['monomer'] = df['smiles']
    smiles = df['smiles'].value_counts().index

    for smile in smiles:
        if isinstance(smile,str):
            idx = df.index[df['smiles']==smile].tolist()
            long_smile = create_long_smiles(smile,req_length=30)
            df.loc[idx,'smiles'] = long_smile

    #remove any smiles that can't be made long
    df = df[df['smiles']!='None']
    df.reset_index(inplace=True,drop=True)

    #get morgan fingerprints for polymers and salts
    n_bits = 128
    morgan_df = add_morgan_cols(df,n_bits=n_bits)
    morgan_df = get_morgan_fps(morgan_df,n_bits=n_bits)
    print('Morgan FPs added')

    for row in range(len(morgan_df)):
        poly = morgan_df['smiles'][row]
        salt = morgan_df['salt smiles'][row]
        formulation = poly + '.' + salt
        morgan_df['smiles'][row] = formulation

    morgan_df.reset_index(drop=True, inplace=True)

    #get lists of numeric columns and salt columns
    real_val_cols=['mw','molality']
    salt_cols = [col for col in morgan_df.columns if 'SaltMorgan' in col]
    poly_cols = [col for col in morgan_df.columns if 'PolyMorgan' in col]

    xgb_cols = real_val_cols+['temperature']+poly_cols+salt_cols+['monomer','salt smiles']

    xgb_full = morgan_df[xgb_cols]
    chem_full_s = morgan_df[['smiles','temperature']]
    chem_full_f = morgan_df[real_val_cols]

    xgb_full.to_csv('{}/xgb_screen.csv'.format(data_folder),index=False)
    chem_full_s.to_csv('{}/s_screen.csv'.format(data_folder),index=False)
    chem_full_f.to_csv('{}/f_screen.csv'.format(data_folder),index=False)
