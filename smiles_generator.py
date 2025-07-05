# smiles_generator.py
import tensorflow as tf
import numpy as np
import traceback # ADD THIS LINE
from config import CONFIG, DIGRAPH_ELEMENT_MAP


REVERSE_ELEMENT_MAP = {v: k for k, v in DIGRAPH_ELEMENT_MAP.items()}

def postprocess_smiles_for_elements(smiles_string: str) -> str:
    """Convert single characters back to multi-character elements"""
    processed_smiles = smiles_string
    for single_char, element in sorted(REVERSE_ELEMENT_MAP.items(), key=lambda item: len(item[0]),
                                       reverse=True):
        processed_smiles = processed_smiles.replace(single_char, element)
    return processed_smiles


class SmilesGenerator:
    def __init__(self, model, char_to_idx_map, idx_to_char_map, config):
        """
        Initializes the SMILES generator.

        Args:
            model: The trained Keras model.
            char_to_idx_map (dict): Mapping from characters to integer indices.
            idx_to_char_map (dict): Mapping from integer indices to characters.
            config (dict): Configuration dictionary containing necessary parameters like
                           'embedding_dim', 'vocabulary_size', 'start_token',
                           'char_padding_token', 'char_end_token', 'digraph_element_map'.
        """
        self.model = model
        self.char_to_idx_map = char_to_idx_map
        self.idx_to_char_map = idx_to_char_map
        self.config = config

        if 'vocabulary_size' not in self.config:
            raise ValueError("Missing 'vocabulary_size' in generator config.")
        if 'embedding_dim' not in self.config:
            print("Warning: 'embedding_dim' not in generator config, defaulting to 1 (uses embedding layer).")

        self.reverse_element_map = {v: k for k, v in self.config.get('digraph_element_map', {}).items()}

        self.start_token_id = self.char_to_idx_map.get(self.config.get('start_token', '<START>'))
        if self.start_token_id is None:
            raise ValueError(f"Start token '{self.config.get('start_token', '<START>')}' not in vocab.")

        self.pad_token_id = self.char_to_idx_map.get(self.config.get('char_padding_token', '<PAD>'))
        self.end_token_id = self.char_to_idx_map.get(self.config.get('char_end_token', '<END>'))


    def _postprocess_smiles_for_elements(self, smiles_string: str) -> str:
        """Converts single characters in a SMILES string back to their multi-character element representations."""
        processed_smiles = smiles_string
        for single_char, element in sorted(self.reverse_element_map.items(), key=lambda item: len(item[0]),
                                           reverse=True):
            processed_smiles = processed_smiles.replace(single_char, element)
        return processed_smiles


    def generate_smiles(self, min_len, max_len, temperature=1.0, auxiliary_features=None):
        """
        Generates a SMILES sequence with a length between min_len and max_len,
        incorporating censored sampling and RNN state management.

        Args:
            min_len (int): The minimum length of the sequence to generate.
            max_len (int): The maximum length of the sequence to generate.
            temperature (float): Controls randomness in sampling. Higher values increase randomness.
            auxiliary_features (dict, optional): Dictionary of auxiliary features for conditional generation.
                                                 Keys should match expected input names in the model's call method
                                                 (e.g., 'cnn', 'cnnAff', 'smilen', 'CNNminAff').
                                                 Values should be raw numbers, which will be converted to tensors.

        Returns:
            str: The generated and post-processed SMILES string.
        """
        if min_len > max_len:
            raise ValueError("min_len cannot be greater than max_len.")

        # Initialize input sequence with the start token
        current_input_id = tf.constant([self.start_token_id], dtype=tf.int32)[:, tf.newaxis] # Shape: (1, 1)
        text_generated_chars = []
        states = None # Initialize RNN states to None

        # Prepare auxiliary features for model input
        model_expects_aux_features = any([
            self.model.cnnscore_embedding_dim > 0,
            self.model.CNNaffinity_embedding_dim > 0,
            self.model.smilelength_embedding_dim > 0,
            self.model.CNNminAffinity_embedding_dim > 0
        ])

        processed_aux_features = {}
        if model_expects_aux_features and auxiliary_features is not None:
            for key, value in auxiliary_features.items():
                if not isinstance(value, tf.Tensor):
                    processed_aux_features[key] = tf.constant([value], dtype=tf.float32)
                else:
                    processed_aux_features[key] = value

        for i in range(max_len):
            try:
                # Prepare model input for the current step
                if model_expects_aux_features:
                    model_input = {
                        'tokens': current_input_id,
                        **processed_aux_features
                    }
                else:
                    model_input = current_input_id

                # Get predictions and updated states from the model
                # The model's call method now returns (logits, states) during inference
                returned_values = self.model(model_input, states=states, training=False)

                # Ensure returned_values is a tuple of (logits, states)
                if not isinstance(returned_values, tuple) or len(returned_values) != 2:
                    raise ValueError("Model did not return tuple of (logits, states) during generation. "
                                     "Ensure ConditionalRNNGenerator.call returns states when training=False.")

                logits_full_seq, new_states = returned_values
                states = new_states # Update states for the next step

                logits_last_step = logits_full_seq[:, -1, :]
                logits_for_prediction = logits_last_step / temperature

                # Mask out the START token to prevent it from being sampled again
                mask = tf.one_hot(self.start_token_id, depth=logits_for_prediction.shape[-1])
                logits_for_prediction = logits_for_prediction - mask * 1e9

                # Sample the next token
                predicted_id_tensor = tf.random.categorical(logits_for_prediction, num_samples=1)
                predicted_id = tf.squeeze(predicted_id_tensor, axis=-1).numpy()[0]

                # --- Censored Sampling Logic (from generate_sequence_censored) ---
                current_length = len(text_generated_chars)

                # If current length is less than min_len, re-sample if PAD/END is predicted
                if current_length < min_len:
                    if predicted_id == self.pad_token_id or \
                       (self.end_token_id is not None and predicted_id == self.end_token_id):
                        # Mask out PAD and END tokens and re-sample
                        temp_logits_for_resampling = tf.identity(logits_for_prediction)
                        if self.pad_token_id is not None:
                            pad_mask_one_hot = tf.one_hot(self.pad_token_id, depth=logits_for_prediction.shape[-1])
                            temp_logits_for_resampling = temp_logits_for_resampling - pad_mask_one_hot * 1e9
                        if self.end_token_id is not None:
                            end_mask_one_hot = tf.one_hot(self.end_token_id, depth=logits_for_prediction.shape[-1])
                            temp_logits_for_resampling = temp_logits_for_resampling - end_mask_one_hot * 1e9

                        predicted_id_tensor = tf.random.categorical(temp_logits_for_resampling, num_samples=1)
                        predicted_id = tf.squeeze(predicted_id_tensor, axis=-1).numpy()[0]

                        # If after re-sampling, it's still a termination token (unlikely but possible), break
                        if predicted_id == self.pad_token_id or \
                           (self.end_token_id is not None and predicted_id == self.end_token_id):
                            break # Give up and terminate

                # Check for termination (applies if min_len is met or after re-sampling)
                """
                if predicted_id == self.pad_token_id or \
                   (self.end_token_id is not None and predicted_id == self.end_token_id):
                    break
                """

                # Ensure START token is not appended to the output string
                if predicted_id == self.start_token_id:
                    continue # Skip appending and continue to next step with same input_eval
                if predicted_id == self.end_token_id:
                    break  # Stop at end token (it's already checked above, but good for safety)
                if predicted_id == self.pad_token_id:
                    continue  # Skip padding token


                # Convert ID to character and append
                predicted_char = self.idx_to_char_map.get(predicted_id, '')
                if predicted_char is not None:
                    if predicted_char:  # Only append if it's a valid character
                        text_generated_chars.append(predicted_char)

                # Update the input for the next step
                current_input_id = tf.constant([predicted_id], dtype=tf.int32)[:, tf.newaxis]

            except Exception as e:
                print(f"Error at generation step {i}: {e}")
                traceback.print_exc()
                break

        generated_smiles_sequence = "".join(text_generated_chars)
        return self._postprocess_smiles_for_elements(generated_smiles_sequence)