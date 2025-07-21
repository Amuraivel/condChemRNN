# condChemRNN
A repository for generative chemistry using a conditional recurrent neural network.
It is a super model of about 6 different models found in the literature, each can be configured using the flags in the configuration file

- poster.html : explains the rationale and the background
- conf.py : sets up the parameters for the various RNN models
- **model_definitions.py**
- data_preprocessor.py : sets up the data and parses variables.
- **RNN.py** : main script that generates smiles
- **Revaluate_model.py** : Looks in a directory and gets .csv files with generated files.

  # How to use

  - Add a local SQLITE.db with your choice of SMILES
  - Point script at that db path in the *config.py*
  - Setup your choice of parameters in the *config.py*
  - 

# Requirements
 - Tensforflow
 - Nvidia GPU
   
   
