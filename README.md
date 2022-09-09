# ChemPropPred

Repository for predicting conductivities through Arrhenious parameters for polymer electrolytes using a modified version of the original chemprop model described in [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237) and available at https://github.com/chemprop/chemprop. The original model was modified to incorporate Arrhenius behavior in its predictions to improve its performance on electrolyte prediction tasks, although it can be applied to any system that follows Arrhenius behavior. All arguments necessary to operate the model with Arrhenius behavior are described under [Running Arrhenius Chemprop](#running-arrhenius-chemprop). Portions of the original chemprop readme are kept intact below for reference during implementation.

#### Installation Steps

1. `git submodule init`
2. `git submodule update`
3. `cd chemarr`
4. `conda env create -f environment.yml`
5. `conda activate chemprop` (Name of the environment variable in the yml file, can be changed)
6. `pip install -e .`
7. `cd ../`
8. `pip install -e .` (Installs the files necessary to launch the training)

#### Scripts and paper reproduction

`train_and_plot_cv_models.py` is a python script that will create a 10fold cross validation split for training the model and reproducing the parity plots from the paper.

You can run `python train_and_plot_cv_models.py --help` to see the possible commands of training. In general the default arguments are set to false, so running the script without any inputs will make nothing happen. To reproduce the parity plot created in the paper one can run `python train_and_plot_cv_models.py --make_data true --train_predict true --plot_parity true`
Though all of these commands can be run separately to test and analyze the separate steps of creating the cross validated data sets, the training and the prediction.

`train_pred_polyinfo.py` is a script that will train the model on all of the training data and use them to predict on the polymers that were scrapped from the PolyInfo database.
You can run `python train_pred_polyinfo.py --help` to see the possible commands of predicting on the PolyInfo dataset. To create the datasets, train the model and predict the conductivities of the PolyInfo dataset run `python train_pred_polyinfo.py --make_data true --train_predict true --polyinfo_datafiles true`

For both training procedures the usage of GPU's has been by default turned off, can be enabled by having a `--gpu` argument flag.

To see information on how to use your own data with these models, how to train the models more customly check it at the [ChemArr](https://github.mit.edu/gbrad/ChemArr) repo.
