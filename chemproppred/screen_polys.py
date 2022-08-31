import os
from chemproppred.utils import *
from rdkit.Chem import PandasTools

def screen_poly(data_path, save_path, preds_path):
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #load xgboost screen set and predictions from chemprop
    xgb_df = pd.read_csv('{}/xgb_screen.csv'.format(data_path))
    chem_preds = pd.read_csv('{}/preds_screen.csv'.format(preds_path))

    #average chemprop and xgboost predictions
    ypreds = chem_preds['conductivity'].values
    Eas = chem_preds['Ea'].values
    logAs = chem_preds['logA'].values

    #sort data for compiling in html file
    sort_idx = list(np.argsort(ypreds))
    sort_idx.reverse()
    sorted_smiles = xgb_df['monomer'][sort_idx]
    sorted_preds = ypreds[sort_idx]
    sorted_temps = xgb_df['temperature'][sort_idx]
    sorted_molals = xgb_df['molality'][sort_idx]
    sorted_salts = xgb_df['salt smiles'][sort_idx]
    sorted_mws = xgb_df['mw'][sort_idx]
    #sorted_complexities = xgb_df['synth_complex'][sort_idx]


    #write screen predictions to html file to show to experimentalists
    df_pred = pd.DataFrame({'Monomer':np.nan,
                            'Salt':sorted_salts,
                            'Molecular Weight' :sorted_mws,
                            'Molality':sorted_molals,
                            'Temperature':sorted_temps,
                            'Predicted Conductivity':sorted_preds,
                            'SMILES':sorted_smiles})

    PandasTools.AddMoleculeColumnToFrame(df_pred, 'SMILES', 'Monomer')
    df_pred.drop(columns=['SMILES'],inplace=True)
    numeric_columns = ['Molecular Weight','Molality','Temperature','Predicted Conductivity',]

    df_pred[numeric_columns] = df_pred[numeric_columns].round(2)
    df_pred.head()

    html = df_pred.to_html()
    with open('{}/screened_polys.html'.format(save_path), 'w') as f:
        f.write(html)

        
    #save predictions for screened data for further analysis
    monomers = xgb_df['monomer'].values
    salts = xgb_df['salt smiles'].values
    temps = xgb_df['temperature'].values
    mws = xgb_df['mw'].values
    molals = xgb_df['molality'].values

    screen_preds = pd.DataFrame(list(zip(monomers,salts,mws,molals,temps, \
                                        ypreds,Eas,logAs)),#complexs)), \
                                columns=['monomer','salt','mw','molality','temperature', \
                                        'predicted_cond','Ea','logA'])
    screen_preds.to_csv('{}/screen_predictions.csv'.format(save_path),index=False)
