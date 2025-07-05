
CONFIG = {
    # TF_GPU_ALLOCATOR=cuda_malloc_async use as an environmental variable
    # --- Data Paths and Column Names ---
    'data_path': 'training_v4.5.csv', #
    'db_data_path': '/home/mark/sambashare/training_data.db',
    'test_data_path': 'training_v1.0.csv', # Frame pre-split
    'output_csv_path': '/home/mark/sambashare/RNN23_SeglerEnumerated.csv',
    # --- Model  Parameters ---
    'min_len': 32,  # Max SMILES length for padding/truncation
    'max_len': 96,  # Max SMILES length for padding/truncation
    'embedding_dim': 1,  #Bjerrum=Token embedding size nil, 1 used for matrix compat; Polykovskiy=64 (i.e. size of vocabulary),  128 maybe overfitting
    'hidden_dim': 256,  # 256 baseline=Bjerrum_2017_Molecular-Gener;   # segler, 2018= 1024, # 992
    # Conditional Model variables
    'smiles_column': 'SMILE',                      #Column
    "cnnscore_column_name": "CNNscore",
    'cnnscore_embedding_dim': 64,         # Gnina Binding score: 8=default
    'CNNminAffinity_column_name': 'minimizedAffinity',
    'CNNminAffinity_embedding_dim': 0,
    "bindingAffinity_column_name": "CNNaffinity",
    "CNNaffinity_embedding_dim": 0,  # Gnina Affinity metric: 8=default
    "SMILElength_column_name": "SMILE_length",
    'SMILElength_embedding_dim': 0, # Length, 1=default
    'sa_score_column': 'sa_score',  # Synthetic accessibility
    'sa_embedding_dim': 0,  # Synthesis: 16
    'slogp_column_name': 'SlogP',  # Name of the Solubility
    'slogp_embedding_dim': 0,  # SlogP embedding size (e.g., 16, or 0 to disable)
    'numaromaticrings_column_name': 'NumAromaticRings',
    'numaromaticrings_embedding_dim': 0, # Aramatic rings
    'numhbd_column_name': 'NumHBD',
    'numhbd_embedding_dim': 0,           # H donors
    'numhba_column_name': 'NumHBA',
    'numhba_embedding_dim': 0,             # H acceptors
    'exactmw_column_name': 'ExactMW',
    'exactmw_embedding_dim': 0,
    'numrotatablebonds_column_name': 'NumRotatableBonds',
    'numrotatablebonds_embedding_dim': 0,
    # Optional tie ins for later
    'calculate_rotbonds_if_missing': False,
    "calculate_cnnscore_if_missing": False,
    # --- Training & Controls Hyperparameters ---
    'shuffle_seed': 42,
    'batch_size': 512, # 2,4,8,16,32,64,128, 256,512==too big with the 3layer model, 2048=TOO big for GPU
    'epochs': 128, # 2,4,8,16,32,64, 128=TOO dimishing returns
    'learning_rate': 0.001,# Default=0.001 works well Bjerrum=0.007
    'dropout_rate': 0.1, # Segler 20218" a dropout ratio of 0.2, to regularize the neural network",  # Gupta base model: 0.2 do,0.3
    # --- Vocabulary and Tokenization ---
    # which should work, but gave intermittent issues depending on model version.
    'char_padding_token': '%', # Not legal SMILE
    'start_token': '!',       # Bjerrum_2017_Molecular-Gener used ! and I wonder if this was better than the <START>, <END> I was using due to the <PAD> errors I got
    'char_end_token': '?',   # End tok
    'vocabulary_size': 64,  # This is a fall back is determined dynamically in the script
    # --- Execution Control & Output ---
    'model_save_path': 'trained_smiles_rnn_generator.keras', #
    'generated_SMILES': 1000,
    'sample_size': None,                    # <= YOUR EXISTING VALUE (e.g., 10000 or None for all data)
    # You might have other custom settings here, keep them!
    'calculate_mw_if_missing': False,
    'calculate_numaromaticrings_if_missing': False,
    'calculate_numhbd_if_missing': False,
    'calculate_numhba_if_missing': False,
    'overfitting_check_epochs': 2,
    'validation_fraction': 0.1,
    'clipnorm': 50, # 5=Segler 2018, pg. 12
    'Bjerrum': True, # Berrum first model in lit; Gupta drops the FF layers
    'Gupta': False,
    'attention': False, # Experiment
    'tune_me':False
}

DIGRAPH_ELEMENT_MAP = {
    """
    "As": "🜺", # Alchemical Arsenic
    "Ag": "☽",  # Alchemical symbol for Silver (Moon)
    "Au": "☉",  # Alchemical symbol for Gold (Sun)
    "Bi": "♆", # Alchemical Bisthmuth
    "Be": "铍", # Chinese character for Beryllium
    "Br": "Β", # Beta for Bromine
    "Co": "🜶", # Cobalt
    "Cl": "Χ", # Chi for Chlorine
    "Ca": "κ", # Kappa for Calcium
    "Cu": "♀", # Alchemical symbol for Copper (Venus)
    "Hg": "☿", # Alchemical symbol for Mercury (Mercury)
    "Fe": "♂",  # Alchemical symbol for Iron (Mars)
    "Li": "Λ", # Lambda for Lithium
    "Mg": "Μ", # Capital Mu for Magnesium
    "Mn": "μ", # Lowercase mu for Manganese
    "Na": "钠", # Nu for Sodium (careful if 'N' and 'a' have other meanings) whose Mandarin name is "nà"
    "Pb": "♄", # Alchemical symbol for lead (Saturn)
    "Sb": "♁", # Antimony
    "Si": "Σ",  # Sigma for Silicon
    "Sn": "♃", # Alchemical symbol for Tin (Jupiter)
    # Add more common two-letter elements if desired, e.g.:
    # Leaving these aside for the moment due to meaning of radicals
    # "[nH]": "ν", # Lowercase nu for [nH] (example for a common bracketed atom)
    # "[N+]": "Π", # Capital Pi for [N+]
    # "[O-]": "Ω", # Omega for [O-]
    """
    "Hg": "☿", # Alchemical symbol for Mercury (Mercury)
}


CHAR_MAP = {0: '%', 1: '!', 2: '?', 3: '#', 4: '(', 5: ')', 6: '-', 7: '.', 8: '/', 9: '0', 10: '1', 11: '2', 12: '3', 13: '4', 14: '5', 15: '6', 16: '7', 17: '8', 18: '9', 19: '@', 20: 'B', 21: 'C', 22: 'F', 23: 'H', 24: 'I', 25: 'N', 26: 'O', 27: 'P', 28: 'S', 29: '[', 30: '\\', 31: ']', 32: 'a', 33: 'c', 34: 'e', 35: 'i', 36: 'l', 37: 'n', 38: 'o', 39: 'r', 40: 's'}

# Add default configuration values for missing keys
def ensure_config_defaults():
    """Ensure all required config keys exist with default values"""
    defaults = {
        'cnnscore_column_name': 'CNNscore',
        'numhba_column_name': 'NumHBA',
    }

    for key, default_value in defaults.items():
        if key not in CONFIG:
            CONFIG[key] = default_value
            print(f"Added missing config key '{key}' with default value: {default_value}")


