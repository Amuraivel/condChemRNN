import tensorflow as tf
import numpy as np
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import traceback
import pandas as pd
# Import custom modules
from model_definitions import ConditionalRNNGenerator, Attention
from config import CONFIG, DIGRAPH_ELEMENT_MAP, ensure_config_defaults
from data_preprocessor import SmilesDataPreprocessor
from smiles_generator import SmilesGenerator
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping
import random

# Configure TensorFlow - DISABLE MIXED PRECISION FOR NOW
# Comment out mixed precision to avoid compatibility issues
# mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
# Restrict TensorFlow to only use the first GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if gpus:
    try:
        for gpu in gpus:
            #tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Ensure all config keys exist
ensure_config_defaults()


def debug_dataset_structure(dataset, name="dataset"):
    """Enhanced debug function to inspect dataset structure - FIXED for dictionaries"""
    print(f"\n=== Debugging {name} ===")
    try:
        sample_batch = next(iter(dataset))

        print(f"Sample batch type: {type(sample_batch)}")
        if isinstance(sample_batch, tuple):
            print(f"Dataset yields tuple with {len(sample_batch)} elements:")
            for i, element in enumerate(sample_batch):
                if isinstance(element, dict):
                    print(f"  Element {i}: dictionary with keys: {list(element.keys())}")
                    for key, value in element.items():
                        if hasattr(value, 'shape'):
                            print(f"    '{key}': shape={value.shape}, dtype={value.dtype}")
                        else:
                            print(f"    '{key}': type={type(value)}")
                elif isinstance(element, tuple):
                    print(f"  Element {i}: tuple with {len(element)} sub-elements")
                    for j, sub_elem in enumerate(element):
                        if hasattr(sub_elem, 'shape'):
                            print(f"    Sub-element {j}: shape={sub_elem.shape}, dtype={sub_elem.dtype}")
                        else:
                            print(f"    Sub-element {j}: type={type(sub_elem)}")
                else:
                    if hasattr(element, 'shape'):
                        print(f"  Element {i}: shape={element.shape}, dtype={element.dtype}")
                    else:
                        print(f"  Element {i}: type={type(element)}")
        else:
            if hasattr(sample_batch, 'shape'):
                print(f"Dataset yields single element: shape={sample_batch.shape}, dtype={sample_batch.dtype}")
            else:
                print(f"Dataset yields single element: type={type(sample_batch)}")

        print(f"Dataset element spec: {dataset.element_spec}")

        # Test iteration consistency
        sample_count = 0
        for batch in dataset.take(2):
            sample_count += 1
            print(f"Batch {sample_count} structure: {type(batch)}")
            if isinstance(batch, tuple):
                print(f"  Batch has {len(batch)} elements")
                for i, elem in enumerate(batch):
                    if isinstance(elem, dict):
                        print(f"    Element {i}: dictionary with {len(elem)} keys")
                    elif hasattr(elem, 'shape'):
                        print(f"    Element {i}: shape={elem.shape}")
                    else:
                        print(f"    Element {i}: type={type(elem)}")

    except Exception as e:
        print(f"Error inspecting {name}: {e}")
        import traceback
        traceback.print_exc()
    print("=" * 50)


def validate_dataset_compatibility(train_ds, val_ds):
    """Validate that train and validation datasets have compatible structures - FIXED for dictionaries"""
    print("\n=== Validating Dataset Compatibility ===")

    try:
        train_sample = next(iter(train_ds))
        val_sample = next(iter(val_ds)) if val_ds else None

        if val_sample is None:
            print("No validation dataset to compare")
            return True

        # Check structure compatibility
        def get_structure_signature(sample):
            if isinstance(sample, tuple):
                if len(sample) == 2:  # (inputs, targets)
                    inputs, targets = sample
                    if isinstance(inputs, dict):
                        keys = sorted(inputs.keys())
                        return f"dict_inputs_keys_{keys}"
                    elif isinstance(inputs, tuple):
                        return f"tuple_inputs_{len(inputs)}_elements"
                    else:
                        return "single_input"
                else:
                    return f"unexpected_tuple_{len(sample)}"
            else:
                return "single_element"

        train_sig = get_structure_signature(train_sample)
        val_sig = get_structure_signature(val_sample)

        print(f"Train dataset signature: {train_sig}")
        print(f"Validation dataset signature: {val_sig}")

        compatible = train_sig == val_sig
        print(f"Datasets compatible: {compatible}")

        if not compatible:
            print("ERROR: Train and validation datasets have incompatible structures!")
            return False

        return True

    except Exception as e:
        print(f"Error validating dataset compatibility: {e}")
        return False


def print_input_structure(sample_inputs, name="sample_inputs"):
    """Helper function to print input structure safely"""
    print(f"{name} type: {type(sample_inputs)}")

    if isinstance(sample_inputs, dict):
        print(f"{name} structure: dictionary with keys: {list(sample_inputs.keys())}")
        for key, value in sample_inputs.items():
            if hasattr(value, 'shape'):
                print(f"  '{key}': shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  '{key}': type={type(value)}")
    elif isinstance(sample_inputs, tuple):
        print(f"{name} structure: tuple with {len(sample_inputs)} elements")
        for i, inp in enumerate(sample_inputs):
            if hasattr(inp, 'shape'):
                print(f"  Input {i}: shape={inp.shape}, dtype={inp.dtype}")
            else:
                print(f"  Input {i}: type={type(inp)}")
    else:
        if hasattr(sample_inputs, 'shape'):
            print(f"{name} shape: {sample_inputs.shape}, dtype: {sample_inputs.dtype}")
        else:
            print(f"{name} type: {type(sample_inputs)}")



def train_model_fixed(model, train_ds, val_ds, epochs=10, verbose=1):
    """Fixed training function with proper error handling"""
    print(f"\n=== Training Model for {epochs} epochs ===")

    # Validate dataset compatibility first
    if not validate_dataset_compatibility(train_ds, val_ds):
        print("Dataset compatibility check failed. Training without validation.")
        val_ds = None

    # Setup callbacks
    callbacks = []
    if val_ds is not None:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

    try:
        print("Starting training with model.fit()...")

        # First try with validation
        if val_ds is not None:
            history = model.fit(
                train_ds,
                epochs=epochs,
                validation_data=val_ds,
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            history = model.fit(
                train_ds,
                epochs=epochs,
                callbacks=[],  # No early stopping without validation
                verbose=verbose
            )

        print("Training completed successfully!")
        return pd.DataFrame(history.history)

    except Exception as e:
        print(f"Training with model.fit() failed: {e}")
        print("Error details:")
        import traceback
        traceback.print_exc()

        # Try without validation if that was the issue
        if val_ds is not None:
            print("\nRetrying without validation data...")
            try:
                history = model.fit(
                    train_ds,
                    epochs=epochs,
                    callbacks=[],
                    verbose=verbose
                )
                print("Training without validation completed successfully!")
                return pd.DataFrame(history.history)
            except Exception as e2:
                print(f"Training without validation also failed: {e2}")

        print("\nFalling back to manual training loop...")
        return manual_training_loop_robust(model, train_ds, val_ds, epochs, verbose)


def manual_training_loop_robust(model, train_ds, val_ds, epochs, verbose):
    """Robust manual training loop with better error handling"""
    print("Using manual training loop...")
    history = {'loss': [], 'accuracy': []}

    # Only add validation metrics if validation dataset exists
    if val_ds is not None:
        history.update({'val_loss': [], 'val_accuracy': []})

    optimizer = model.optimizer
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # Enforces vocabulary size

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training phase
        train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        try:
            for step, batch_data in enumerate(train_ds):
                # Handle batch structure
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    batch_x, batch_y = batch_data
                else:
                    print(f"Unexpected training batch structure at step {step}: {type(batch_data)}")
                    continue

                # Ensure targets are properly formatted
                if isinstance(batch_y, tuple):
                    batch_y = batch_y[0]  # Take first element if it's a tuple

                batch_y = tf.cast(batch_y, tf.int32)

                # Training step
                with tf.GradientTape() as tape:
                    predictions = model(batch_x, training=True)
                    loss = loss_fn(batch_y, predictions)

                # Apply gradients
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                # Update metrics
                train_loss_metric.update_state(loss)
                train_accuracy_metric.update_state(batch_y, predictions)

        except Exception as e:
            print(f"Error in training phase: {e}")
            traceback.print_exc()
            break

        # Get training metrics
        train_loss = float(train_loss_metric.result())
        train_accuracy = float(train_accuracy_metric.result())

        # Validation phase
        val_loss = 0.0
        val_accuracy = 0.0

        if val_ds is not None:
            val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
            val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

            try:
                for step, batch_data in enumerate(val_ds):
                    # Handle batch structure
                    if isinstance(batch_data, tuple) and len(batch_data) == 2:
                        batch_x, batch_y = batch_data
                    else:
                        print(f"Unexpected validation batch structure at step {step}: {type(batch_data)}")
                        continue

                    # Ensure targets are properly formatted
                    if isinstance(batch_y, tuple):
                        batch_y = batch_y[0]

                    batch_y = tf.cast(batch_y, tf.int32)

                    # Validation step
                    predictions = model(batch_x, training=False)
                    loss = loss_fn(batch_y, predictions)

                    # Update metrics
                    val_loss_metric.update_state(loss)
                    val_accuracy_metric.update_state(batch_y, predictions)

            except Exception as e:
                print(f"Error in validation phase: {e}")
                traceback.print_exc()

            val_loss = float(val_loss_metric.result())
            val_accuracy = float(val_accuracy_metric.result())

        # Store metrics
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)

        if val_ds is not None:
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

        # Print progress
        if verbose:
            print(f"  Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}", end="")
            if val_ds is not None:
                print(f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            else:
                print()

    return pd.DataFrame(history)



# COMPLETE main function for RNN.py - replace your existing main function
def main(train_me=True, test_me=False, tune_me=False, generate_me=True, gen_temperature=1):
    """Complete main function to work with pandas-split approach"""
    print("=== SMILES RNN Generator ===")
    print(f"TensorFlow version: {tf.__version__}")

    # Initialize data preprocessor
    print("\n--- Initializing Data Preprocessor ---")
    preprocessor = SmilesDataPreprocessor(
        config=CONFIG,
        digraph_element_map=DIGRAPH_ELEMENT_MAP,
        use_testing_data=False
    )

    # Load and split training data at pandas level

    print("--- Loading and Splitting Training Data ---")
    data_load_success = preprocessor.load_and_process_csv(
        validation_fraction=CONFIG.get('validation_split', 0.2),
        shuffle_seed=CONFIG.get('shuffle_seed', 42)
    )
    if not data_load_success:
        print("ERROR: Could not load training data. Exiting.")
        return
    print("✓ Training data loaded and split successfully")

    # Prepare training datasets from pandas splits
    print("--- Creating TensorFlow Datasets ---")
    train_ds, val_ds = preprocessor.prepare_tf_datasets()

    if train_me or test_me:

        if train_ds is None:
            print("ERROR: Training dataset could not be created. Exiting.")
            return

        print('Training dataset:')
        debug_dataset_structure(train_ds, "train_ds")

        if val_ds:
            print('Validation dataset:')
            debug_dataset_structure(val_ds, "val_ds")
        else:
            print("No validation dataset to debug.")

        # Validate dataset compatibility
        if val_ds is not None:
            print("\n--- Validating Dataset Compatibility ---")
            if not validate_dataset_compatibility(train_ds, val_ds):
                print("Warning: Dataset compatibility issues detected. Will train without validation.")
                val_ds = None

        # Create model
        print(f"\n--- Creating Model (vocab_size={CONFIG['vocabulary_size']}) ---")
        # --- START OF MODIFICATION ---

        # 1. Define Input Layers explicitly
        max_len = CONFIG['max_len']
        vocab_size = CONFIG['vocabulary_size'] # Ensure this is from preprocessor.vocab_size after building vocab

        input_layers = {}
        if CONFIG['embedding_dim'] > 0:
            input_layers['tokens'] = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='tokens')
        else:
            # If embedding_dim is 0, data preprocessor performs one-hot encoding
            input_layers['tokens'] = tf.keras.Input(shape=(max_len, vocab_size), dtype=tf.float32, name='tokens')

        # Add input layers for active auxiliary features
        if CONFIG['cnnscore_embedding_dim'] > 0:
            input_layers['cnn'] = tf.keras.Input(shape=(), dtype=tf.float32, name='cnn') # Scalar
        if CONFIG['CNNaffinity_embedding_dim'] > 0:
            input_layers['cnnAff'] = tf.keras.Input(shape=(), dtype=tf.float32, name='cnnAff') # Scalar
        if CONFIG['SMILElength_embedding_dim'] > 0:
            input_layers['smilen'] = tf.keras.Input(shape=(), dtype=tf.float32, name='smilen') # Scalar
        if CONFIG['CNNminAffinity_embedding_dim'] > 0:
            input_layers['CNNminAff'] = tf.keras.Input(shape=(), dtype=tf.float32, name='CNNminAff') # Scalar
        # Add other auxiliary inputs here if their embedding_dim > 0 in CONFIG
        # Example:
        if CONFIG['sa_embedding_dim'] > 0:
            input_layers['sa'] = tf.keras.Input(shape=(), dtype=tf.float32, name='sa')
        if CONFIG['slogp_embedding_dim'] > 0:
            input_layers['slogp'] = tf.keras.Input(shape=(), dtype=tf.float32, name='slogp')
        if CONFIG['exactmw_embedding_dim'] > 0:
            input_layers['mw'] = tf.keras.Input(shape=(), dtype=tf.float32, name='mw')
        if CONFIG['numrotatablebonds_embedding_dim'] > 0:
            input_layers['rotbonds'] = tf.keras.Input(shape=(), dtype=tf.float32, name='rotbonds')
        if CONFIG['numaromaticrings_embedding_dim'] > 0:
            input_layers['arorings'] = tf.keras.Input(shape=(), dtype=tf.float32, name='arorings')
        if CONFIG['numhbd_embedding_dim'] > 0:
            input_layers['hbd'] = tf.keras.Input(shape=(), dtype=tf.float32, name='hbd')
        if CONFIG['numhba_embedding_dim'] > 0:
            input_layers['hba'] = tf.keras.Input(shape=(), dtype=tf.float32, name='hba')


        # 2. Instantiate the ConditionalRNNGenerator (your subclassed layer)
        generator_layer = ConditionalRNNGenerator(
            vocab_size=vocab_size, # Use the actual vocab_size from preprocessor
            embedding_dim=CONFIG.get('embedding_dim', 128),
            rnn_units=CONFIG.get('hidden_dim', 256),
            dropout_rate=CONFIG.get('dropout_rate', 0.2),
            sa_embedding_dim=CONFIG.get('sa_embedding_dim', 0),
            slogp_embedding_dim=CONFIG.get('slogp_embedding_dim', 0),
            exactmw_embedding_dim=CONFIG.get('exactmw_embedding_dim', 0),
            numrotatablebonds_embedding_dim=CONFIG.get('numrotatablebonds_embedding_dim', 0),
            cnnscore_embedding_dim=CONFIG['cnnscore_embedding_dim'],
            SMILElength_embedding_dim=CONFIG['SMILElength_embedding_dim'],
            CNNaffinity_embedding_dim=CONFIG['CNNaffinity_embedding_dim'],
            CNNminAffinity_embedding_dim=CONFIG['CNNminAffinity_embedding_dim'],
            numaromaticrings_embedding_dim=CONFIG.get('numaromaticrings_embedding_dim', 0),
            numhbd_embedding_dim=CONFIG.get('numhbd_embedding_dim', 0),
            numhba_embedding_dim=CONFIG.get('numhba_embedding_dim', 0),
            random_seed=CONFIG['shuffle_seed']
        )

        # 3. Call the generator_layer with the input_layers to get the output tensor
        # If input_layers is a dictionary, pass it directly.
        # If it's a single Input tensor (no auxiliary features), pass it directly.
        if len(input_layers) > 1: # Check if it's a dictionary of inputs
            # Pass the dictionary of Input tensors to the generator layer
            outputs = generator_layer(input_layers, training=True) # Assume training=True for graph definition
            # The 'outputs' here will be the tensor produced by your generator_layer's call method.
            # It's the output for the loss calculation.
            model_inputs = input_layers # The model's inputs are the dictionary of Input layers
        else: # Only one input (tokens)
            model_inputs = list(input_layers.values())[0] # Get the single Input tensor
            outputs = generator_layer(model_inputs, training=True) # Pass the single Input tensor

        # 4. Create the final tf.keras.Model
        model = tf.keras.Model(inputs=model_inputs, outputs=outputs)

        # --- END OF MODIFICATION ---

        # Build model properly - This step is now implicitly handled by tf.keras.Model construction
        print("\n--- Building Model ---")
        try:
            # model.build is implicitly called by tf.keras.Model(inputs=..., outputs=...)
            # We can directly print the summary now, as the model is built.
            print("✓ Model built successfully via functional API construction")

        except Exception as e:
            print(f"ERROR: Model building failed: {e}")
            traceback.print_exc()
            return

        # Print model summary
        try:
            model.summary(line_length=120)
        except Exception as e:
            print(f"Could not print model summary: {e}")

        # Compile model
        print("\n--- Compiling Model ---")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=CONFIG.get('learning_rate', 0.001)
            # Default
                #clipnorm = CONFIG.get('clipnorm',5)
            #
            )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        print("✓ Model compiled successfully")

        if train_me:
            # Train model using the fixed training function
            print("\n--- Training Model ---")
            history_df = train_model_fixed(  # Use the fixed training function
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                epochs=CONFIG.get('epochs', 10),
                verbose=1
            )

            print("\n--- Training History ---")
            if history_df is not None and not history_df.empty:
                print(history_df.tail())

                # Plot training history if possible
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(12, 4))

                    plt.subplot(1, 2, 1)
                    plt.plot(history_df['loss'], label='Training Loss')
                    if 'val_loss' in history_df.columns:
                        plt.plot(history_df['val_loss'], label='Validation Loss')
                    plt.title('Model Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()

                    plt.subplot(1, 2, 2)
                    plt.plot(history_df['accuracy'], label='Training Accuracy')
                    if 'val_accuracy' in history_df.columns:
                        plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
                    plt.title('Model Accuracy')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.legend()

                    plt.tight_layout()
                    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
                    #plt.show()
                    print("✓ Training plots saved to training_history.png")

                except ImportError:
                    print("Matplotlib not available for plotting")
                except Exception as e:
                    print(f"Could not create training plots: {e}")
            else:
                print("Training did not complete successfully, no history available.")

            # Save model
            model_save_path = CONFIG['model_save_path']
            print(f"\n--- Saving Model to {model_save_path} ---")
            try:
                model.save(model_save_path)
                print(f"✓ Model saved successfully to {model_save_path}")
            except Exception as e:
                print(f"Model saving failed: {e}")
                traceback.print_exc()

        # Test on out-of-sample data if requested
        if test_me:
            print("\n--- Testing on Out-of-Sample Data ---")
            test_preprocessor = SmilesDataPreprocessor(
                config=CONFIG,
                digraph_element_map=DIGRAPH_ELEMENT_MAP,
                use_testing_data=True
            )

            test_load_success = test_preprocessor.load_and_process_csv(
                validation_fraction=0.0,  # No validation split for test data
                shuffle_seed=CONFIG.get('shuffle_seed', 42)
            )
            if not test_load_success:
                print("ERROR: Could not load test data.")
                return

            # Use the same vocabulary from training
            test_preprocessor.char2idx = preprocessor.char2idx
            test_preprocessor.idx2char = preprocessor.idx2char
            test_preprocessor.vocab_size = preprocessor.vocab_size

            test_ds, _ = test_preprocessor.prepare_tf_datasets()

            if test_ds:
                print("Testing dataset:")
                debug_dataset_structure(test_ds, "test_ds")

                # Evaluate on test set
                print("\n--- Evaluating on Test Set ---")
                try:
                    test_results = model.evaluate(test_ds, verbose=1)
                    print(f"Test Loss: {test_results[0]:.4f}, Test Accuracy: {test_results[1]:.4f}")
                except Exception as e:
                    print(f"Test evaluation failed: {e}")
                    traceback.print_exc()




    if tune_me: # This block is for fine-tuning an already saved model
        model_save_path = CONFIG['model_save_path']
        print(f"\n--- Loading Model for Fine-tuning from {model_save_path} ---")
        try:
            # Load the model directly
            # For loading, you must provide the custom_objects to deserialize your ConditionalRNNGenerator
            model = tf.keras.models.load_model(
                model_save_path,
                custom_objects={'ConditionalRNNGenerator': ConditionalRNNGenerator, 'Attention': Attention} # Add Attention if it's used in your model and needs serialization
            )
            print(f"✓ Model loaded successfully from {model_save_path}")

            # Recompile the model for fine-tuning
            print("\n--- Recompiling Model for Fine-tuning ---")
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
            print("✓ Model recompiled for fine-tuning")

            # Prepare tune_ds (if different from train_ds, otherwise use train_ds)
            # Assuming tune_ds is equivalent to train_ds or specifically prepared for tuning
            tune_ds = train_ds # Or load a specific fine-tuning dataset if you have one

            # Fine-tune the model
            print("\n--- Fine-tuning Model ---")
            fine_tune_epochs = CONFIG.get('fine_tune_epochs', 5) # New config parameter
            history_df_finetune = train_model_fixed(
                model=model,
                train_ds=tune_ds,
                val_ds=val_ds, # Use validation data if available
                epochs=fine_tune_epochs,
                verbose=1
            )

            print("\n--- Fine-tuning History ---")
            if history_df_finetune is not None and not history_df_finetune.empty:
                print(history_df_finetune.tail())

            # Save the fine-tuned model
            fine_tuned_model_save_path = CONFIG.get('fine_tuned_model_save_path', './fine_tuned_model.keras')
            print(f"\n--- Saving Fine-tuned Model to {fine_tuned_model_save_path} ---")
            try:
                model.save(fine_tuned_model_save_path)
                print(f"✓ Fine-tuned Model saved successfully to {fine_tuned_model_save_path}")
            except Exception as e:
                print(f"Fine-tuned model saving failed: {e}")
                traceback.print_exc()

        except Exception as e:
            print(f"ERROR: Could not load or fine-tune model: {e}")
            traceback.print_exc()
            return # Exit if fine-tuning fails




    # Initialize the SMILES Generator
    if generate_me:
        print("\n--- Initializing SMILES Generator ---")

        # --- MOVE THIS BLOCK HERE ---
        # Prepare conditional inputs for generation - FIXED for dictionary structure
        # Ensure these are single-element tensors or have a batch dimension of 1
        generation_conditional_inputs = {}
        if CONFIG['cnnscore_embedding_dim'] > 0:
            generation_conditional_inputs['cnn'] = tf.constant([1.0], dtype=tf.float32)  # Use float for CNN score
        if CONFIG['CNNaffinity_embedding_dim'] > 0:
            generation_conditional_inputs['cnnAff'] = tf.constant([7.5], dtype=tf.float32)
        if CONFIG['SMILElength_embedding_dim'] > 0:
            generation_conditional_inputs['smilen'] = tf.constant([70.0], dtype=tf.float32)  # Use float
        if CONFIG['CNNminAffinity_embedding_dim'] > 0:
            generation_conditional_inputs['CNNminAff'] = tf.constant([-10.0], dtype=tf.float32)  # Use float
        # Add other auxiliary inputs here for generation
        if CONFIG['sa_embedding_dim'] > 0:
            generation_conditional_inputs['sa'] = tf.constant([2.5], dtype=tf.float32)  # Example value
        if CONFIG['slogp_embedding_dim'] > 0:
            generation_conditional_inputs['slogp'] = tf.constant([3.0], dtype=tf.float32)  # Example value
        if CONFIG['exactmw_embedding_dim'] > 0:
            generation_conditional_inputs['mw'] = tf.constant([300.0], dtype=tf.float32)  # Example value
        if CONFIG['numrotatablebonds_embedding_dim'] > 0:
            generation_conditional_inputs['rotbonds'] = tf.constant([5.0], dtype=tf.float32)  # Example value
        if CONFIG['numaromaticrings_embedding_dim'] > 0:
            generation_conditional_inputs['arorings'] = tf.constant([2.0], dtype=tf.float32)  # Example value
        if CONFIG['numhbd_embedding_dim'] > 0:
            generation_conditional_inputs['hbd'] = tf.constant([1.0], dtype=tf.float32)  # Example value
        if CONFIG['numhba_embedding_dim'] > 0:
            generation_conditional_inputs['hba'] = tf.constant([3.0], dtype=tf.float32)  # Example value
        # --- END MOVE ---

        try:
            print('Loading the saved model.....')
            model = tf.keras.models.load_model(
                CONFIG['model_save_path'],
                custom_objects={'ConditionalRNNGenerator': ConditionalRNNGenerator, 'Attention': Attention}
            )
            print('Model loaded.....')

            generator_core_layer = None
            for layer in model.layers:
                if isinstance(layer, ConditionalRNNGenerator):
                    generator_core_layer = layer
                    break

            if generator_core_layer is None:
                raise ValueError("ConditionalRNNGenerator layer not found in the loaded model.")

        except Exception as e:
            print(f"Model not loaded {e}")
            traceback.print_exc()
            return

        generator_config = CONFIG.copy()
        generator_config['digraph_element_map'] = DIGRAPH_ELEMENT_MAP

        try:
            generator = SmilesGenerator(
                model=generator_core_layer,
                char_to_idx_map=preprocessor.char2idx,
                idx_to_char_map=preprocessor.idx2char,
                config=generator_config
            )
            print("\n--- Generating Samples ---")
            smiles_output = []

            print("Generating SMILES sequences...")
            print(CONFIG['generated_SMILES'])

            i = 0
            while i < CONFIG['generated_SMILES']:
                random_len = random.uniform(CONFIG['min_len'], CONFIG['max_len'])
                random_len = int(random_len)

                smile = generator.generate_smiles(
                    min_len=CONFIG['min_len'],
                    max_len=random_len,
                    temperature=gen_temperature,
                    auxiliary_features=generation_conditional_inputs
                )
                # Fixed some odd regression in the code.
                if smile.startswith('None'):
                    smile = smile[4:]
                smiles_output.append(smile)
                i += 1
                print(f'{i}, {smile}')

            # ... (rest of saving results code - no change here) ...




        except Exception as e:
            print(f"SMILES generation failed: {e}")
            traceback.print_exc()






    print("\n=== Script Completed ===")
    if train_me and 'model_save_path' in locals() and os.path.exists(CONFIG['model_save_path']):
        print(f"Model saved to: {CONFIG['model_save_path']}")


if __name__ == "__main__":
    # Set train_me=True for training, test_me=True for out-of-sample testing
    main(train_me=True, test_me=False, tune_me=False, generate_me=True, gen_temperature=0.5)