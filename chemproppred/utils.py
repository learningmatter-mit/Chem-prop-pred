import pandas as pd
import numpy as np
import math
import scipy.spatial as sp
import scipy
import sys

from rdkit.Chem import MolFromSmiles, MolToSmiles, AllChem, rdFingerprintGenerator
from rdkit.Chem.Lipinski import HeavyAtomCount


# function that calculates long smiles
def create_long_smiles(smile, req_length):
    # check if smile is a polymer
    if "Cu" in smile:
        # calculate required repeats so smiles > 30 atoms long
        num_heavy = HeavyAtomCount(MolFromSmiles(smile)) - 2
        repeats = math.ceil(req_length / num_heavy) - 1

        # if polymer is less than 30 long, repeat until 30 long
        if repeats > 0:
            try:
                # code to increase length of monomer
                mol = MolFromSmiles(smile)
                new_mol = mol

                # join repeats number of monomers into polymer
                for i in range(repeats):
                    # join two polymers together at Cu and Au sites
                    rxn = AllChem.ReactionFromSmarts("[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]")
                    results = rxn.RunReactants((mol, new_mol))
                    assert len(results) == 1 and len(results[0]) == 1, smile
                    new_mol = results[0][0]

                new_smile = MolToSmiles(new_mol)

            except:
                # make smile none if reaction fails
                return "None"

        # if monomer already long enough use 1 monomer unit
        else:
            new_smile = smile

        # caps ends of polymers with carbons
        new_smile = (
            new_smile.replace("[Cu]", "C").replace("[Au]", "C").replace("[Ca]", "C")
        )

    else:
        new_smile = smile

    # make sure new smile in cannonical
    long_smile = MolToSmiles(MolFromSmiles(new_smile))
    return long_smile


# function that turns monomers into long polymer ring
def make_ring(smile, req_length):
    try:
        smile = create_long_smiles(smiles, req_length)
        # connect ends of monomers to form ring structure
        mol = MolFromSmiles(smile)
        rxn = AllChem.ReactionFromSmarts("([Cu][*:1].[*:2][Au])>>[*:1]-[*:2]")
        results = rxn.RunReactants([mol])
        mol = results[0][0]
        SanitizeMol(mol)
        new_smile = MolToSmiles(mol)
    except:
        return "None"

    return new_smile


# add morgan fingerprint columns to dataframe
def add_morgan_cols(df, n_bits=32, salts=True):
    morgan_df = df.copy()

    poly_cols = []
    salt_cols = []
    for i in range(n_bits):
        poly_col = "PolyMorgan" + str(i)
        poly_cols.append(poly_col)
        if salts:
            salt_col = "SaltMorgan" + str(i)
            salt_cols.append(salt_col)

    morgan_df.loc[:, poly_cols + salt_cols] = 0

    return morgan_df


# fill morgan fingerprint columns with values
def get_morgan_fps(df, n_bits=32, salts=True):
    # get polymer smiles and columns
    poly_cols = [col for col in df.columns if "PolyMorgan" in col]
    smiles = df["smiles"].value_counts().index

    # get morgan fps for each polymer smile and add to df
    for smile in smiles:
        if isinstance(smile, str):
            idx = df.index[df["smiles"] == smile].tolist()

            m = MolFromSmiles(smile)
            # fp = list(AllChem.GetMorganFingerprintAsBitVect(m,3,nBits=n_bits))
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
            fp = list(mfpgen.GetFingerprint(m))
            df.loc[idx, poly_cols] = fp

    # repeat for salts
    if salts:
        salt_cols = [col for col in df.columns if "SaltMorgan" in col]
        salt_smiles = df["salt smiles"].value_counts().index

        for smile in salt_smiles:
            if isinstance(smile, str):
                idx = df.index[df["salt smiles"] == smile].tolist()

                m = MolFromSmiles(smile)
                # fp = list(AllChem.GetMorganFingerprintAsBitVect(m,3,nBits=n_bits))
                mfpgen = rdFingerprintGenerator.GetMorganGenerator(
                    radius=3, fpSize=n_bits
                )
                fp = list(mfpgen.GetFingerprint(m))
                df.loc[idx, salt_cols] = fp

    return df


# calculates the weighted avg chemical tanimoto distance between test and training chemicals
def calc_chem_dists(df_train, df_test, a=3, per_poly=0.7):
    # calculate avg distance for each data point
    avg_dists = []

    poly_cols = [col for col in df_train.columns if "PolyMorgan" in col]
    salt_cols = [col for col in df_train.columns if "SaltMorgan" in col]

    # convert data to numpy arrays
    poly_train_array = df_train[poly_cols].to_numpy()
    poly_test_array = df_test[poly_cols].to_numpy()
    salt_train_array = df_train[salt_cols].to_numpy()
    salt_test_array = df_test[salt_cols].to_numpy()

    # calculate distances for all polymers and salts
    poly_dists = sp.distance.cdist(poly_test_array, poly_train_array, "rogerstanimoto")
    salt_dists = sp.distance.cdist(salt_test_array, salt_train_array, "rogerstanimoto")

    # get sdc values
    sdc_poly = np.exp(-a * poly_dists / (1 - poly_dists))
    sdc_salt = np.exp(-a * salt_dists / (1 - salt_dists))

    # get average sdc for polys and salts
    poly_avg = 1 - sdc_poly.mean(axis=1)
    salt_avg = 1 - sdc_salt.mean(axis=1)

    # calc weighted average of distances for poly and salts
    avg_dists = per_poly * poly_avg + (1 - per_poly) * salt_avg

    return avg_dists


# estimate prediction error based on chemical distance
def get_error_from_dist(ypreds, all_dists, dists_df, dist_range=0.04, n_points=200):
    stds, cis, probs = [], [], []

    for num in range(len(all_dists)):
        # get distance and prediction
        dist_test = all_dists[num]
        pred_test = ypreds[num]

        # get all data within range of distance
        dist_min = dist_test - dist_range
        dist_max = dist_test + dist_range
        df_dist_test = dists_df[
            dists_df["Distance"].between(dist_min, dist_max, inclusive=False)
        ]

        # if less than 200 points in range, increase range until 200 points
        while len(df_dist_test) < n_points:
            dist_range += 0.0005
            dist_min = dist_test - dist_range
            dist_max = dist_test + dist_range
            df_dist_test = dists_df[
                dists_df["Distance"].between(dist_min, dist_max, inclusive=False)
            ]

        # get residuals in distance range
        test_resids = list(df_dist_test["Residual"])

        # calculate std dev of residuals
        n = len(test_resids)
        std = np.sqrt(sum(np.square(test_resids)) / (n - 2))

        # make confidence intervals for each prediction
        ci = 1.66 * std
        cntr = pred_test

        cis.append(ci)
        stds.append(std)

        probz = (-3 - cntr) / std
        prob = scipy.stats.norm.sf(probz)
        probs.append(prob)

    return stds, cis, probs


# function that repeats smiles a number of times
def repeat_monomer(smile, repeats):
    # check if smile is a polymer
    mol = MolFromSmiles(smile)
    new_mol = mol

    try:
        # join repeats number of monomers into polymer
        for i in range(0, repeats):
            # join two polymers together at Cu and Au sites
            rxn = AllChem.ReactionFromSmarts("[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]")
            results = rxn.RunReactants((mol, new_mol))
            assert len(results) == 1 and len(results[0]) == 1, smile
            new_mol = results[0][0]

        new_smile = MolToSmiles(new_mol)
    except:
        return "None"

    # caps ends of polymers with carbons
    new_smile = new_smile.replace("[Cu]", "C").replace("[Au]", "C").replace("[Ca]", "C")

    long_smile = MolToSmiles(MolFromSmiles(new_smile))

    return long_smile


def plot_hexbin(pred, targ, ax, scale="log", plot_helper_lines=False):
    if scale == "log":
        pred = np.abs(pred) + 1e-8
        targ = np.abs(targ) + 1e-8

    hb = ax.hexbin(
        pred,
        targ,
        cmap="viridis",
        gridsize=80,
        bins="log",
        mincnt=1,
        edgecolors=None,
        linewidths=(0.1,),
        xscale=scale,
        yscale=scale,
        extent=(
            min(np.min(pred), np.min(targ)) * 1.1,
            max(np.max(pred), np.max(targ)) * 1.1,
            min(np.min(pred), np.min(targ)) * 1.1,
            max(np.max(pred), np.max(targ)) * 1.1,
        ),
    )

    lim_min = min(np.min(pred), np.min(targ)) * 1.1
    lim_max = max(np.max(pred), np.max(targ)) * 1.1

    ax.set_aspect("equal")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    # ax.set_aspect(aspect=1)

    ax.plot(
        (lim_min, lim_max),
        (lim_min, lim_max),
        color="#000000",
        zorder=-1,
        linewidth=0.5,
    )

    return ax, hb
