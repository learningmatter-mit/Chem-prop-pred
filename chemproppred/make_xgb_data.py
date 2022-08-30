from sklearn.metrics import mean_absolute_error
from chemproppred.utils import *


def make_xgb_data(data_path, file_name="dists_resids"):
    #make list of all polymers ordered by frequency of occurance
    ys,ypreds,ypreds_chem,dists,all_smiles = [],[],[],[],[]
    all_salts, all_molals, all_mws, all_temps = [],[],[],[]

    #get smiles string for test polymer
    for test_num in range(2,12):
        #load train and cv data
        train = pd.read_csv('{}/xgb_train_{}.csv'.format(data_path,test_num))
        test = pd.read_csv('{}/xgb_test_{}.csv'.format(data_path,test_num))

        #get y values
        y_train = train['conductivity'].values
        y_test = test['conductivity'].values

        chem_preds_df = pd.read_csv('{}/arr/preds_{}_5.csv'.format(data_path,test_num))
        chem_preds = chem_preds_df['conductivity'].values
        
        #add real ys and predicted ys to list
        smiles = test['monomer'].values
        salts = test['salt smiles'].values
        molals = test['molality'].values
        mws = test['mw'].values
        temps = test['temperature'].values


        all_smiles.extend(smiles)
        all_salts.extend(salts)
        all_molals.extend(molals)
        all_mws.extend(mws)
        all_temps.extend(temps)

        ys.extend(y_test)
        ypreds_chem.extend(chem_preds)
        avg_dists = calc_chem_dists(train,test)
        dists.extend(avg_dists)
        

    yps = ypreds_chem
    resids = np.array(ys) - np.array(yps)
    mae = mean_absolute_error(yps,ys)
    print("Mean Absolue Error: ",mae)

    #save distance vs residual data for later error estimates
    dists_df = pd.DataFrame(list(zip(all_smiles,all_salts,all_molals,all_mws,all_temps,dists,resids,ys,yps)), \
                            columns=['monomer','salt','molality','mw','temperature', \
                                    'Distance','Residual','experimental','pred'])
    formulations = dists_df['monomer']+dists_df['salt']+dists_df['molality'].astype(str)+dists_df['mw'].astype(str)
    forms = formulations.unique()

    dists, errs = [],[]

    for form in forms:
        idx = formulations[formulations==form].index
        tempdf = dists_df.iloc[idx]
        
        conds = tempdf['experimental'].values
        preds = tempdf['pred'].values

        err = mean_absolute_error(conds, preds)
        dist = tempdf['Distance'].values[0]

        errs.append(err)
        dists.append(dist)

    dists_df = pd.DataFrame(zip(errs,dists), columns={'Residual','Distance'})
    dists_df.to_csv('{}/{}.csv'.format(data_path,file_name),index=False)


