import os
from chemarr.utils import *

#function to fill in needed data
def process_df(df,n_bits,req_length):
    #create long smiles
    df['monomer'] = df['smiles']
    smiles = df['smiles'].value_counts().index

    for smile in smiles:
        if isinstance(smile,str):
            idx = df.index[df['smiles']==smile].tolist()
            long_smile = create_long_smiles(smile,req_length)
            df.loc[idx,'smiles'] = long_smile    

    #add Li+ if no salt present for polyions
    df['salt smiles'].fillna('[Li+]',inplace=True)

    #get morgan fingerprints for solvents and salts for XGBoost
    morgan_df = add_morgan_cols(df,n_bits=n_bits)
    print('Adding morgan fingerprints...')
    morgan_df = get_morgan_fps(morgan_df,n_bits=n_bits)
    
    #combine salt and polymer for GNN to featurize together
    for row in range(len(morgan_df)):
        poly = morgan_df['smiles'][row]
        salt = morgan_df['salt smiles'][row]
        formulation = poly + '.' + salt
        morgan_df['smiles'][row] = formulation

    #fill empty molecular weight and molality values
    morgan_df['mw'] = morgan_df['mw'].fillna(65000)
    morgan_df['mw'] = np.log10(morgan_df['mw'])
    morgan_df['molality'].fillna(0,inplace=True)
    morgan_df.reset_index(drop=True,inplace=True)
   

    return morgan_df


def make_balanced_data(data_path, dataset):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    #parameters to adjust data balancing
    n_repeats=0
    peo_divide=5

    #read in experimental dataset
    df = pd.read_csv(dataset)

    #process data for ML
    n_bits=128
    morgan_df = process_df(df,n_bits,req_length=30)

    #make list of all polymers ordered by frequency of occurance
    poly_smiles = morgan_df['monomer'].value_counts().index
    num_smiles = len(poly_smiles)

    #get numeric and salt morgan columns
    real_val_cols=['mw','molality']
    salt_cols = [col for col in morgan_df.columns if 'SaltMorgan' in col]

    #####balance data for full training
    repeat_df = morgan_df.copy()

    #make full data df to use for training full models for screen
    xgb_full = repeat_df.drop(columns=['smiles'])
    chem_full_s = repeat_df[['smiles','conductivity','temperature']]
    chem_full_f = repeat_df[real_val_cols]

    #save full data for use
    xgb_full.to_csv('{}/xgb_full.csv'.format(data_path),index=False)
    chem_full_s.to_csv('{}/s_full.csv'.format(data_path),index=False)
    chem_full_f.to_csv('{}/f_full.csv'.format(data_path),index=False)


    #this loop saves 10 cv splits for estimating error of models
    #get smiles string for test polymer
    for test_num in range(2,12):
        
        #get test data by taking 1/10 of polymers for test set
        test_jump = 10
        test_nums = [(test_num%test_jump+num) for num in range(0,(num_smiles+30),test_jump) \
                    if (test_num%test_jump+num)<num_smiles and (test_num%test_jump+num)>1]
        test_smiles = poly_smiles[test_nums]
        #this bit remove polyanions from the test data
        test_smiles = [smile for smile in test_smiles if '-' not in smile]
        
        #get cv set by taking 1/5 of polymers
        cv_jump = 5
        cv_nums = [(test_num%cv_jump+num) for num in range(1,(num_smiles+30),cv_jump) \
                if (test_num%cv_jump+num)<num_smiles and (test_num%cv_jump+num)>1]
        cv_smiles = poly_smiles[cv_nums]

        #get row indexes containing the cv and test smiles
        cv_idx = morgan_df[morgan_df['monomer'].isin(cv_smiles)].index
        test_idx = morgan_df[morgan_df['monomer'].isin(test_smiles)].index

        #separate data into train test and cv 
        cv = morgan_df.iloc[cv_idx]
        test = morgan_df.iloc[test_idx]
        train = morgan_df.drop(cv_idx)
        train.drop(test_idx,inplace=True)    

        #make chemprop data
        cv_chem_s = cv[['smiles','conductivity','temperature']]
        test_chem_s = test[['smiles','conductivity','temperature']]
        train_chem_s = train[['smiles','conductivity','temperature']]
        
        cv_chem_f = cv[real_val_cols]
        test_chem_f = test[real_val_cols]
        train_chem_f = train[real_val_cols]
        
        xgb_cv = cv.drop(columns=['smiles'])
        xgb_test = test.drop(columns=['smiles'])
        xgb_train = train.drop(columns=['smiles'])
        #save data to folder
        train_chem_s.to_csv('{}/s_train_{}.csv'.format(data_path,test_num),index=False)
        cv_chem_s.to_csv('{}/s_cv_{}.csv'.format(data_path,test_num),index=False)
        test_chem_s.to_csv('{}/s_test_{}.csv'.format(data_path,test_num),index=False)

        train_chem_f.to_csv('{}/f_train_{}.csv'.format(data_path,test_num),index=False)
        cv_chem_f.to_csv('{}/f_cv_{}.csv'.format(data_path,test_num),index=False)
        test_chem_f.to_csv('{}/f_test_{}.csv'.format(data_path,test_num),index=False)

        xgb_train.to_csv('{}/xgb_train_{}.csv'.format(data_path,test_num),index=False)
        xgb_cv.to_csv('{}/xgb_cv_{}.csv'.format(data_path,test_num),index=False)
        xgb_test.to_csv('{}/xgb_test_{}.csv'.format(data_path,test_num),index=False)
