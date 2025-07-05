#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluates the trained RNN model on the test set
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Import custom modules
from model_definitions import ConditionalRNNGenerator
from config import CONFIG, DIGRAPH_ELEMENT_MAP, ensure_config_defaults
from DL_models.model4_RNN.old.data_preprocessor22 import SmilesDataPreprocessor

# Configure TensorFlow
tf.config.run_functions_eagerly(False)

# Ensure all config keys exist
ensure_config_defaults()


def evaluate_trained_model():
    """
    Evaluate the trained model on the test set
    """
    print("=== Model Evaluation on Test Set ===")
    print(f"TensorFlow version: {tf.__version__}")

    # Paths
    model_path = CONFIG.get('model_save_path', './trained_rnn_generator.keras')
    test_data_path = './data/training_data/test_v0.8v.csv'  # Ensure this is your correct test data path

    print(f"Model path: {model_path}")
    print(f"Test data path: {test_data_path}")

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None

    if not os.path.exists(test_data_path):
        print(f"ERROR: Test data file not found: {test_data_path}")
        return None

    # Load the trained model
    print("\n--- Loading Trained Model ---")
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'ConditionalRNNGenerator': ConditionalRNNGenerator}
        )
        print("✓ Model loaded successfully")
        try:
            # This might not be available directly for subclassed models until built with concrete shapes
            if model.built:
                print(f"Model input shape (from model.input_shape if available): {model.input_shape}")
                print(f"Model output shape (from model.output_shape if available): {model.output_shape}")
            else:
                print("Model input/output shapes not available until first call.")
        except AttributeError:
            print("Custom model - input/output shapes may not be directly queryable like this.")

        if hasattr(model, 'built') and model.built:
            print("✓ Model is built and ready")
        else:
            print("Model not built yet - will be built on first call during evaluation/debug.")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Prepare test data using the same preprocessor
    print("\n--- Preparing Test Data ---")

    # Create a copy of config for test data
    test_config = CONFIG.copy()
    test_config['data_path'] = test_data_path
    # Ensure sample_size is None for test data to use all of it
    test_config['sample_size'] = None

    # Initialize preprocessor for test data
    test_preprocessor = SmilesDataPreprocessor(
        config=test_config,
        digraph_element_map=DIGRAPH_ELEMENT_MAP
    )

    # Load and process test data
    try:
        success = test_preprocessor.load_and_process_csv()
        if not success or not test_preprocessor.final_smiles_for_tokenization:  # Added check for empty list
            print("ERROR: Failed to load or process test data, or no valid SMILES found.")
            return None
    except Exception as e:
        print(f"ERROR: Exception while loading test data: {e}")
        return None

    print(f"✓ Test data loaded: {len(test_preprocessor.final_smiles_for_tokenization)} samples")

    # Create test dataset (no validation split, so validation_fraction=0.0)
    # prepare_tf_datasets returns (train_ds, val_ds), we only need the first for test_ds
    test_ds, _ = test_preprocessor.prepare_tf_datasets(validation_fraction=0.0,
                                                       shuffle_seed=None)  # No need to shuffle test data

    if test_ds is None:
        print("ERROR: Test dataset could not be created.")
        return None

    print(f"Test dataset element spec: {test_ds.element_spec}")
    print(f"Test dataset cardinality (batches): {test_ds.cardinality()}")

    # Debug: Check model output structure first
    print("\n--- Debugging Model Output Structure ---")
    try:
        sample_batch = next(iter(test_ds))
        sample_x_features_tuple, sample_y_labels = sample_batch

        # To take just one sample from the batch for this debug call:
        # Create a new tuple where each feature tensor is sliced.
        # This assumes sample_x_features_tuple is indeed a tuple from the dataset.
        if isinstance(sample_x_features_tuple, tuple):
            debug_sample_x = tuple(feat_tensor[:1] for feat_tensor in sample_x_features_tuple)
        else:  # Single feature input (e.g. tokens only)
            debug_sample_x = sample_x_features_tuple[:1]

        # The model's call method has slicing for 3D inputs, so warnings might appear here if inputs are still 3D
        # Test with training=True
        output_train = model(debug_sample_x, training=True)
        print(f"Model output (training=True): type={type(output_train)}")
        if isinstance(output_train, tf.RaggedTensor):
            print(f"  Output shape (Ragged): {output_train.shape}")
        elif isinstance(output_train, tuple):
            print(f"  Tuple length: {len(output_train)}")
            for i, elem in enumerate(output_train):
                print(f"    Element {i} shape: {elem.shape if hasattr(elem, 'shape') else type(elem)}")
        else:  # Should be EagerTensor
            print(f"  Output shape: {output_train.shape}")

        # Test with training=False
        output_infer = model(debug_sample_x, training=False)
        print(f"Model output (training=False): type={type(output_infer)}")
        if isinstance(output_infer, tf.RaggedTensor):
            print(f"  Output shape (Ragged): {output_infer.shape}")
        elif isinstance(output_infer, tuple):
            print(f"  Tuple length: {len(output_infer)}")
            for i, elem in enumerate(output_infer):
                print(
                    f"    Element {i} (logits) shape: {elem[0].shape if isinstance(elem, list) and elem else (elem.shape if hasattr(elem, 'shape') else type(elem))}")
                if isinstance(elem, list) and len(elem) > 1:
                    print(
                        f"    Element {i} (states) details: {len(elem[1]) if isinstance(elem[1], list) else type(elem[1])}")

        else:  # Should be EagerTensor
            print(f"  Output shape: {output_infer.shape}")


    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Using Manual Evaluation ---")
    evaluation_results = manual_evaluation(model, test_ds, test_preprocessor)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    if evaluation_results:
        print(f"Model Path: {model_path}")
        print(f"Test Data: {test_data_path}")
        print(f"Evaluation Method: {evaluation_results.get('method', 'N/A')}")
        print(f"Test Loss: {evaluation_results.get('test_loss', float('nan')):.6f}")
        print(f"Test Accuracy: {evaluation_results.get('test_accuracy', float('nan')):.6f}")
        print(f"Test Samples Processed by Eval: {evaluation_results.get('total_samples_processed_in_eval', 0)}")
        print(f"Original Test Samples Loaded: {evaluation_results.get('original_test_samples_loaded', 0)}")
        print(f"Vocabulary Size: {evaluation_results.get('vocab_size', 'Unknown')}")
        print(f"Num Batches Processed in Eval: {evaluation_results.get('num_batches_processed', 0)}")
    else:
        print("Evaluation did not produce results.")
    print("=" * 60)

    # Save results to CSV
    output_path = CONFIG.get('evaluation_results_csv_path', './evaluation_results.csv')  # Use config
    if evaluation_results:
        try:
            results_df = pd.DataFrame([evaluation_results])
            results_df.to_csv(output_path, index=False)
            print(f"✓ Results saved to: {output_path}")
        except Exception as e:
            print(f"Failed to save results: {e}")

    return evaluation_results


def manual_evaluation(model, test_ds, test_preprocessor):
    """
    Manual evaluation loop that handles model output correctly
    """
    print("Performing manual evaluation...")

    total_loss = tf.constant(0.0, dtype=tf.float32)
    total_accuracy = tf.constant(0.0, dtype=tf.float32)
    num_batches = 0
    total_samples_processed_in_eval = tf.constant(0, dtype=tf.int32)

    for batch_idx, batch_data in enumerate(test_ds):
        try:
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                batch_x_features_tuple, batch_y_labels = batch_data
            else:
                print(f"Warning: Unexpected batch structure in batch {batch_idx}: {type(batch_data)}. Skipping batch.")
                continue

            # The dtypes should be correct from data_preprocessor.
            # Token input is batch_x_features_tuple[0] (int32)
            # Conditional inputs are batch_x_features_tuple[1:] (float32)
            # batch_y_labels is int32.

            # No need to cast batch_x_features_tuple as a whole.
            # Cast batch_y_labels to int32 if it's not already (though it should be).
            batch_y_labels = tf.cast(batch_y_labels, tf.int32)

            # Get predictions - use training=True to get just logits
            model_output = model(batch_x_features_tuple, training=True)

            # The model's call method should return only logits when training=True
            predictions = model_output  # Assuming model_output is directly the logits tensor

            current_batch_size = tf.shape(batch_y_labels)[0]  # Actual batch size for this batch

            loss_per_sample = tf.keras.losses.sparse_categorical_crossentropy(batch_y_labels, predictions,
                                                                              from_logits=True)

            # Create a mask to ignore padding tokens in loss calculation
            # Assuming pad_token_id = 0 (as per typical vocab setup)
            pad_token_id = test_preprocessor.char2idx.get(test_preprocessor.config['char_padding_token'], 0)
            mask = tf.cast(tf.not_equal(batch_y_labels, pad_token_id), dtype=loss_per_sample.dtype)
            loss_per_sample *= mask  # Apply mask

            # Sum loss over sequence length and then mean over batch
            batch_loss = tf.reduce_sum(loss_per_sample) / tf.reduce_sum(mask)  # Average over non-padded tokens

            if tf.math.is_nan(batch_loss) or tf.math.is_inf(batch_loss):
                print(
                    f"  Warning: Invalid loss (NaN/Inf) in batch {num_batches}, skipping accumulation for this batch.")
            else:
                total_loss += batch_loss * tf.cast(current_batch_size,
                                                   tf.float32)  # Weight by actual samples in batch for sum

            pred_ids = tf.argmax(predictions, axis=-1, output_type=tf.int32)
            correct_predictions = tf.cast(tf.equal(pred_ids, batch_y_labels), dtype=tf.float32)
            correct_predictions *= mask  # Apply mask to accuracy calculation as well

            batch_accuracy = tf.reduce_sum(correct_predictions) / tf.reduce_sum(mask)

            if tf.math.is_nan(batch_accuracy) or tf.math.is_inf(batch_accuracy):
                print(
                    f"  Warning: Invalid accuracy (NaN/Inf) in batch {num_batches}, skipping accumulation for this batch.")
            else:
                total_accuracy += batch_accuracy * tf.cast(current_batch_size, tf.float32)  # Weight by actual samples

            total_samples_processed_in_eval += current_batch_size
            num_batches += 1

            if num_batches % 50 == 0 or num_batches == 1:
                current_avg_loss = (total_loss / tf.cast(total_samples_processed_in_eval,
                                                         tf.float32)) if total_samples_processed_in_eval > 0 else 0.0
                current_avg_acc = (total_accuracy / tf.cast(total_samples_processed_in_eval,
                                                            tf.float32)) if total_samples_processed_in_eval > 0 else 0.0
                print(
                    f"  Processed batch {num_batches}... Current Avg Loss: {current_avg_loss:.4f}, Current Avg Acc: {current_avg_acc:.4f}")

        except tf.errors.OutOfRangeError:  # Should not happen with finite datasets in loop
            print("End of test data reached unexpectedly in loop.")
            break
        except Exception as e:
            print(f"Error processing batch {num_batches} in manual_evaluation: {e}")
            import traceback
            traceback.print_exc()
            # Optionally continue to next batch or break
            # continue

    final_loss_val = 0.0
    final_accuracy_val = 0.0
    if total_samples_processed_in_eval > 0:
        final_loss_val = float(total_loss / tf.cast(total_samples_processed_in_eval, tf.float32))
        final_accuracy_val = float(total_accuracy / tf.cast(total_samples_processed_in_eval, tf.float32))
        if np.isnan(final_loss_val): final_loss_val = 0.0  # Handle potential NaN if all batches had issues
        if np.isnan(final_accuracy_val): final_accuracy_val = 0.0

    print(
        f"\nManual evaluation: Final metrics calculated from {num_batches} batches and {int(total_samples_processed_in_eval)} samples processed.")
    print(f"Final loss: {final_loss_val:.6f}")
    print(f"Final accuracy: {final_accuracy_val:.6f}")

    return {
        'method': 'manual_evaluation_v2',
        'test_loss': final_loss_val,
        'test_accuracy': final_accuracy_val,
        'original_test_samples_loaded': len(test_preprocessor.final_smiles_for_tokenization),
        'total_samples_processed_in_eval': int(total_samples_processed_in_eval.numpy()),
        'vocab_size': test_preprocessor.config.get('vocabulary_size', 'Unknown'),
        'num_batches_processed': num_batches
    }


if __name__ == "__main__":
    print("Starting model evaluation...")
    results = evaluate_trained_model()

    if results:
        print(f"\n✓ Evaluation completed successfully!")
        if 'test_accuracy' in results and 'test_loss' in results:
            print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"Final Test Loss: {results['test_loss']:.4f}")
        else:
            print("Results dictionary is missing expected keys 'test_accuracy' or 'test_loss'.")
            print(f"Full results: {results}")
    else:
        print("\n✗ Evaluation failed or did not produce results!")