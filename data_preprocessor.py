# PANDAS SPLIT FIX - Replace the entire SmilesDataPreprocessor class
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import os

class SmilesDataPreprocessor:
    def __init__(self, config, digraph_element_map, use_testing_data):
        self.config = config
        self.digraph_element_map = digraph_element_map
        self.reverse_element_map = {v: k for k, v in self.digraph_element_map.items()}
        self.char2idx = None
        self.idx2char = None
        self.vocab_size = 0
        self.use_testing_data = use_testing_data

        # Store processed data for both train and validation
        self.train_data = None
        self.val_data = None

        self.data_features_configuration = [
            ('sa', 'sa_embedding_dim', self.config['sa_score_column'], None, None),
            ('slogp', 'slogp_embedding_dim', self.config['slogp_column_name'], None, None),
            ('mw', 'exactmw_embedding_dim', self.config['exactmw_column_name'], Descriptors.ExactMolWt,
             'calculate_mw_if_missing'),
            ('rotbonds', 'numrotatablebonds_embedding_dim', self.config['numrotatablebonds_column_name'],
             Descriptors.NumRotatableBonds, 'calculate_rotbonds_if_missing'),
            ('cnn', 'cnnscore_embedding_dim', self.config['cnnscore_column_name'], None, None),
            ('cnnAff', 'CNNaffinity_embedding_dim', self.config['bindingAffinity_column_name'], None, None),
            ('smilen', 'SMILElength_embedding_dim', self.config['SMILElength_column_name'], None, None),
            ('CNNminAff', 'CNNminAffinity_embedding_dim', self.config['CNNminAffinity_column_name'], None, None),
            ('arorings', 'numaromaticrings_embedding_dim', self.config['numaromaticrings_column_name'],
             Lipinski.NumAromaticRings, 'calculate_numaromaticrings_if_missing'),
            ('hbd', 'numhbd_embedding_dim', self.config['numhbd_column_name'], Lipinski.NumHDonors,
             'calculate_numhbd_if_missing'),
            ('hba', 'numhba_embedding_dim', self.config['numhba_column_name'], Lipinski.NumHAcceptors,
             'calculate_numhba_if_missing')
        ]

    def _preprocess_smiles_for_elements(self, smiles_string: str) -> str:
        processed_smiles = smiles_string
        for element, single_char in sorted(self.digraph_element_map.items(), key=lambda item: len(item[0]),
                                           reverse=True):
            processed_smiles = processed_smiles.replace(element, single_char)
        return processed_smiles

    def load_and_process_csv(self, validation_fraction=0.2, shuffle_seed=42):
        """Load CSV and split at pandas level, then process each split identically"""
        print(f"Loading data from: {self.config['data_path']}")

        # Create a SQL connection to our SQLite database

        try:
            SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

            if self.use_testing_data:
                DATA_FILE_PATH = os.path.join(SCRIPT_DIR, self.config['test_data_path'])
            else:
                DATA_FILE_PATH = os.path.join(SCRIPT_DIR, self.config['data_path'])

            print(f"Loading data from absolute path: {DATA_FILE_PATH}")

            #df = pd.read_csv(DATA_FILE_PATH, nrows=100000)
            con = sqlite3.connect(self.config['db_data_path'])
            # table_names = pd.read_sql_query("SELECT name FROM datasets.sqlite_master  L;", con)
            # print(table_names)
            # Bjerrum fill 100k - 80kidx_to_char_map
            # Gupta: fill=541555-80000
            # Grisoni:
            # Segler = 1.4m
            #df = pd.read_sql_query("SELECT * FROM (SELECT * FROM SCORED_DAT UNION SELECT * FROM (SELECT * FROM FILLER_DAT  ORDER BY random() LIMIT 132000))", con)
            df = pd.read_sql_query("SELECT * FROM SCORED_DAT ORDER BY random() LIMIT 2000000",con)
            # USe just the tunning
            #if CONFIG['tune_me']==True:
            #    df = pd.read_sql_query("SELECT * FROM SCORED_DAT", con)


            if self.config['smiles_column'] not in df.columns:
                raise KeyError(f"SMILES column '{self.config['smiles_column']}' not found.")

            print(f"Loaded {len(df)} rows.")

            # PANDAS-LEVEL SPLIT - This is the key innovation
            df_shuffled = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

            # Split into train and validation
            num_val_samples = int(len(df_shuffled) * validation_fraction)
            num_train_samples = len(df_shuffled) - num_val_samples

            df_train = df_shuffled[:num_train_samples].copy()
            df_val = df_shuffled[num_train_samples:].copy() if num_val_samples > 0 else None

            print(
                f"Split data: {len(df_train)} train samples, {len(df_val) if df_val is not None else 0} validation samples")

            # Process training data
            print("Processing training data...")
            self.train_data = self._process_dataframe(df_train, "train")

            # Process validation data (if exists)
            if df_val is not None and len(df_val) > 0:
                print("Processing validation data...")
                self.val_data = self._process_dataframe(df_val, "validation")
            else:
                print("No validation data to process")
                self.val_data = None

            # Build vocabulary from training data only
            if self.train_data and 'processed_smiles' in self.train_data:
                self._build_vocab(self.train_data['processed_smiles'])
                print(f"Built vocabulary with {self.vocab_size} tokens")
            else:
                raise ValueError("No training data available to build vocabulary")

            return True

        except Exception as e:
            print(f"Error during data loading/processing: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _process_dataframe(self, df, split_name):
        """Process a single dataframe (train or validation) identically"""
        processed_smiles = []
        feature_data = {name: [] for name, _, _, _, _ in self.data_features_configuration}

        needs_mol_object = any(self.config.get(calc_flag_key, False)
                               for _, _, _, _, calc_flag_key in self.data_features_configuration if calc_flag_key)

        valid_count = 0
        for index, row in df.iterrows():
            smi_raw = str(row[self.config['smiles_column']])
            if pd.isna(smi_raw) or not smi_raw:
                continue

            mol = None
            if needs_mol_object:
                mol = Chem.MolFromSmiles(smi_raw)

            # Process features for this row
            current_feature_values = {}
            all_features_valid = True

            for name, dim_key, col_key, calc_func, calc_flag_key in self.data_features_configuration:
                if self.config[dim_key] > 0:
                    val = np.nan
                    obtained = False
                    try:
                        if col_key in row and not pd.isna(row[col_key]):
                            val = float(row[col_key])
                            obtained = not pd.isna(val)
                        elif calc_func and self.config.get(calc_flag_key, False):
                            if not mol and needs_mol_object:
                                all_features_valid = False
                                break
                            if mol:
                                val = float(calc_func(mol))
                                obtained = not pd.isna(val)
                            else:
                                obtained = False

                        if not obtained:
                            all_features_valid = False
                            break
                    except (ValueError, TypeError):
                        all_features_valid = False
                        break
                    current_feature_values[name] = val

            if not all_features_valid:
                continue

            # If we get here, this row is valid
            processed_smiles.append(self._preprocess_smiles_for_elements(smi_raw))
            for name, dim_key, _, _, _ in self.data_features_configuration:
                if self.config[dim_key] > 0:
                    feature_data[name].append(current_feature_values[name])

            valid_count += 1

        print(f"Processed {valid_count} valid samples for {split_name} split")

        # Convert feature lists to numpy arrays
        feature_arrays = {}
        for name, dim_key, _, _, _ in self.data_features_configuration:
            if self.config[dim_key] > 0 and feature_data[name]:
                arr = np.array(feature_data[name], dtype=np.float32).reshape(-1, 1)
                feature_arrays[name] = arr
                print(f"  {split_name}: {len(arr)} {name.upper()} values")

        return {
            'processed_smiles': processed_smiles,
            'feature_arrays': feature_arrays,
            'count': valid_count
        }

    def _build_vocab(self, processed_smiles_list):
        """Build vocabulary from processed SMILES strings"""
        if not processed_smiles_list:
            raise ValueError("Cannot build vocabulary: No processed SMILES strings available.")

        charset_from_data = sorted(list(set("".join(processed_smiles_list))))

        PAD_TOKEN = self.config['char_padding_token']
        START_TOKEN_STR = self.config['start_token']
        END_TOKEN_STR = self.config['char_end_token']


        # Pad character is assigned 0
        self.char2idx = {PAD_TOKEN: 0}

        idx_counter = 1
        for token in [START_TOKEN_STR, END_TOKEN_STR]:
            # add it into the list
            if token not in self.char2idx:
                self.char2idx[token] = idx_counter
                idx_counter += 1
        # Add in dataset characters
        for char_token in charset_from_data:
            if char_token not in self.char2idx:
                self.char2idx[char_token] = idx_counter
                idx_counter += 1

        print(self.char2idx)
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        self.config['vocabulary_size'] = self.vocab_size

    def _tokenize_and_pad_sequences(self, processed_smiles_list):
        """Tokenize and pad SMILES sequences"""
        if not processed_smiles_list:
            raise ValueError("No SMILES strings available for tokenization.")
        if not self.char2idx:
            raise ValueError("Vocabulary not built.")

        pad_id = self.char2idx[self.config['char_padding_token']]
        start_id = self.char2idx[self.config['start_token']]
        end_id = self.char2idx[self.config['char_end_token']]
        max_len = self.config['max_len']

        input_sequences = []
        target_sequences = []

        for smiles_str in processed_smiles_list:
            content_ids = [self.char2idx.get(c, pad_id) for c in smiles_str]

            input_seq = [start_id] + content_ids
            input_seq = input_seq[:max_len]
            input_seq_padded = input_seq + [pad_id] * (max_len - len(input_seq))
            input_sequences.append(input_seq_padded)

            target_seq = content_ids + [end_id]
            target_seq = target_seq[:max_len]
            target_seq_padded = target_seq + [pad_id] * (max_len - len(target_seq))
            target_sequences.append(target_seq_padded)

        return np.array(input_sequences, dtype=np.int32), np.array(target_sequences, dtype=np.int32)

    def prepare_tf_datasets(self, validation_fraction=0.2, shuffle_seed=None):
        """Create TensorFlow datasets from pre-processed pandas splits"""
        if self.train_data is None:
            raise ValueError("No training data available. Call load_and_process_csv() first.")

        print("Creating TensorFlow datasets from pandas-split data...")

        # Tokenize training data
        train_inputs, train_targets = self._tokenize_and_pad_sequences(self.train_data['processed_smiles'])
        print(f"Tokenized {len(train_inputs)} training sequences")

        # Tokenize validation data (if exists)
        val_inputs, val_targets = None, None
        if self.val_data is not None:
            val_inputs, val_targets = self._tokenize_and_pad_sequences(self.val_data['processed_smiles'])
            print(f"Tokenized {len(val_inputs)} validation sequences")

        # Get active features
        active_features = []
        for name, dim_key, _, _, _ in self.data_features_configuration:
            if self.config[dim_key] > 0:
                if name in self.train_data['feature_arrays']:
                    active_features.append(name)

        print(f"Active auxiliary features: {active_features}")

        # Create datasets based on whether we have auxiliary features
        if active_features:
            # WITH auxiliary features - use dictionary structure
            print("Creating datasets with auxiliary features (dictionary structure)")

            # Training dataset
            train_input_dict = {'tokens': train_inputs}
            for feature_name in sorted(active_features):  # Sort for consistency
                train_input_dict[feature_name] = self.train_data['feature_arrays'][feature_name].squeeze()

            train_dataset = tf.data.Dataset.from_tensor_slices((train_input_dict, train_targets))

            # Validation dataset (if exists)
            val_dataset = None
            if val_inputs is not None and self.val_data is not None:
                val_input_dict = {'tokens': val_inputs}
                for feature_name in sorted(active_features):
                    if feature_name in self.val_data['feature_arrays']:
                        val_input_dict[feature_name] = self.val_data['feature_arrays'][feature_name].squeeze()

                val_dataset = tf.data.Dataset.from_tensor_slices((val_input_dict, val_targets))
        else:
            # WITHOUT auxiliary features - use simple tensor structure
            print("Creating datasets with tokens only (simple tensor structure)")

            train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (val_inputs, val_targets)) if val_inputs is not None else None

        # Apply one-hot encoding if needed
        if self.config.get('embedding_dim', 1) == 0:
            def one_hot_map_fn(inputs, targets):
                if isinstance(inputs, dict):
                    # Handle dictionary input
                    processed_inputs = inputs.copy()
                    processed_inputs['tokens'] = tf.one_hot(tf.cast(inputs['tokens'], tf.int32),
                                                            depth=self.vocab_size, dtype=tf.float32)
                    for key in processed_inputs:
                        if key != 'tokens':
                            processed_inputs[key] = tf.cast(processed_inputs[key], tf.float32)
                else:
                    # Handle tensor input
                    processed_inputs = tf.one_hot(tf.cast(inputs, tf.int32),
                                                  depth=self.vocab_size, dtype=tf.float32)
                return processed_inputs, tf.cast(targets, tf.int32)

            train_dataset = train_dataset.map(one_hot_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
            if val_dataset is not None:
                val_dataset = val_dataset.map(one_hot_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and optimize
        train_dataset = train_dataset.batch(self.config['batch_size'], drop_remainder=True)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        if val_dataset is not None:
            val_dataset = val_dataset.batch(self.config['batch_size'], drop_remainder=False)
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        # Debug output
        print(f"Train dataset element spec: {train_dataset.element_spec}")
        if val_dataset is not None:
            print(f"Validation dataset element spec: {val_dataset.element_spec}")
        else:
            print("No validation dataset created")

        return train_dataset, val_dataset