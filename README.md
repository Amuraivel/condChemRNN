# condChemRNN
A repository for generative chemistry using a conditional recurrent neural network.
It is a super model of 6 different models found in the literature, each can be configured using the flags in the configuration file.

- **poster.html** : explains the rationale and the background
- **conf.py** : sets up the parameters for the various RNN models
- **model_definitions.py** : Defines the RNN model.
- **data_preprocessor.py** : Sets up the data and parses variables.
- **RNN.py** : Main script that coordinates the model and generates SMILES.
- **evaluate_model.py** : Looks in a directory and gets .csv files with generated files.
- **enumerate_SMILES.py** : Lists out various versions of canonical SMILES.


# How to use
- Add a local SQLITE.db with your choice of SMILES
- Point script at that db path in the *config.py*
- Setup your choice of parameters in the *config.py*
- Run RNN.py
- Put .csv files into results directory
- Run *evaluate_model.py*

# Requirements
 - Tensforflow
 - Nvidia GPU
   
   
