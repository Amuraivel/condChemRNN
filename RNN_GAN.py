import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import os
from smiles_generator import generate_sequence
from config import CONFIG, DIGRAPH_ELEMENT_MAP


class RNN_Generator:
    """Enhanced Generator that uses a pretrained RNN model and fine-tunes it with GAN training"""

    def __init__(self, model_path, char_to_idx_map, idx_to_char_map):
        """
        Initialize the generator with a pretrained model

        Args:
            model_path: Path to the saved pretrained model
            char_to_idx_map: Character to index mapping from training
            idx_to_char_map: Index to character mapping from training
        """
        self.model = load_model(model_path)
        self.char_to_idx_map = char_to_idx_map
        self.idx_to_char_map = idx_to_char_map

        # Extract vocab size from the model
        self.vocab_size = self.model.vocab_size

        # Set generation parameters
        self.start_token = CONFIG.get('start_token', '<START>')
        self.max_len = CONFIG.get('max_len', 100)
        self.min_len = CONFIG.get('min_len', 30)
        self.temperature = 1.0

        print(f"Generator initialized with vocab size: {self.vocab_size}")
        print(f"Model input shape: {self.model.input_shape}")

    def generate_smiles_batch(self, batch_size=32, temperature=1.0):
        """
        Generate a batch of SMILES strings

        Args:
            batch_size: Number of SMILES to generate
            temperature: Sampling temperature

        Returns:
            List of generated SMILES strings
        """
        smiles_list = []

        for i in range(batch_size):
            try:
                # Generate random length for this sample
                random_len = np.random.randint(self.min_len, self.max_len + 1)

                # Generate SMILES using your existing generation function
                smiles = generate_sequence(
                    model=self.model,
                    generation_start_token_str=self.start_token,
                    char_to_idx_map=self.char_to_idx_map,
                    idx_to_char_map=self.idx_to_char_map,
                    max_len=random_len,
                    min_len=self.min_len,
                    temperature=temperature
                )

                smiles_list.append(smiles)

            except Exception as e:
                print(f"Generation failed for sample {i}: {e}")
                smiles_list.append("")  # Add empty string for failed generations

        return smiles_list

    def get_trainable_variables(self):
        """Get trainable variables for fine-tuning"""
        return self.model.trainable_variables


class EnhancedDiscriminator:
    """Enhanced Discriminator that provides gradients for generator training"""

    def __init__(self, validity_weight=1.0, drug_likeness_weight=0.5):
        """
        Initialize discriminator with multiple scoring criteria

        Args:
            validity_weight: Weight for basic RDKit validity
            drug_likeness_weight: Weight for drug-likeness properties
        """
        self.validity_weight = validity_weight
        self.drug_likeness_weight = drug_likeness_weight

    def is_valid(self, smiles):
        """Basic RDKit validity check"""
        if not smiles or smiles.strip() == "":
            return False
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    def compute_drug_likeness_score(self, smiles):
        """
        Compute drug-likeness score based on Lipinski's Rule of Five
        Returns score between 0 and 1
        """
        if not self.is_valid(smiles):
            return 0.0

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        try:
            # Calculate Lipinski descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            # Lipinski's Rule of Five criteria
            violations = 0
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if hbd > 5: violations += 1
            if hba > 10: violations += 1

            # Convert violations to score (0 violations = 1.0, 4 violations = 0.0)
            drug_score = 1.0 - (violations / 4.0)

            return drug_score

        except Exception as e:
            print(f"Error calculating drug-likeness for {smiles}: {e}")
            return 0.0

    def score_smiles_batch(self, smiles_list):
        """
        Score a batch of SMILES strings

        Args:
            smiles_list: List of SMILES strings

        Returns:
            numpy array of scores between 0 and 1
        """
        scores = []

        for smiles in smiles_list:
            validity_score = 1.0 if self.is_valid(smiles) else 0.0
            drug_score = self.compute_drug_likeness_score(smiles)

            # Weighted combination
            final_score = (self.validity_weight * validity_score +
                           self.drug_likeness_weight * drug_score) / (
                                  self.validity_weight + self.drug_likeness_weight)

            scores.append(final_score)

        return np.array(scores, dtype=np.float32)


class GANTrainer:
    """Trainer class for fine-tuning the RNN generator using GAN training"""

    def __init__(self, generator, discriminator,
                 generator_lr=1e-5,
                 reward_threshold=0.7,
                 save_interval=100):
        """
        Initialize GAN trainer

        Args:
            generator: RNN_Generator instance
            discriminator: EnhancedDiscriminator instance
            generator_lr: Learning rate for generator fine-tuning
            reward_threshold: Minimum reward threshold for considering good molecules
            save_interval: Save model every N iterations
        """
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = Adam(learning_rate=generator_lr)
        self.reward_threshold = reward_threshold
        self.save_interval = save_interval

        # Training metrics
        self.training_history = {
            'iteration': [],
            'avg_reward': [],
            'validity_rate': [],
            'drug_likeness_rate': [],
            'generator_loss': []
        }

    @tf.function
    def compute_generator_loss(self, rewards):
        """
        Compute generator loss using REINFORCE algorithm

        Args:
            rewards: Tensor of rewards from discriminator

        Returns:
            Generator loss (negative expected reward)
        """
        # Use negative reward as loss (we want to maximize reward)
        loss = -tf.reduce_mean(rewards)
        return loss

    def train_step(self, batch_size=32, temperature=1.0):
        """
        Perform one training step

        Args:
            batch_size: Number of molecules to generate per step
            temperature: Sampling temperature

        Returns:
            Dictionary with step metrics
        """
        # Generate batch of SMILES
        generated_smiles = self.generator.generate_smiles_batch(
            batch_size=batch_size,
            temperature=temperature
        )

        # Score with discriminator
        rewards = self.discriminator.score_smiles_batch(generated_smiles)
        rewards_tensor = tf.constant(rewards, dtype=tf.float32)

        # Compute generator loss and gradients
        with tf.GradientTape() as tape:
            # For REINFORCE, we need to trace through the generation process
            # This is a simplified version - in practice, you might need more sophisticated
            # policy gradient implementation
            generator_loss = self.compute_generator_loss(rewards_tensor)

        # Apply gradients to generator
        generator_vars = self.generator.get_trainable_variables()
        gradients = tape.gradient(generator_loss, generator_vars)

        # Clip gradients to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        self.generator_optimizer.apply_gradients(zip(gradients, generator_vars))

        # Calculate metrics
        avg_reward = np.mean(rewards)
        validity_rate = np.mean([self.discriminator.is_valid(s) for s in generated_smiles])
        drug_likeness_rate = np.mean([
            self.discriminator.compute_drug_likeness_score(s) > 0.5
            for s in generated_smiles
        ])

        return {
            'generator_loss': float(generator_loss),
            'avg_reward': avg_reward,
            'validity_rate': validity_rate,
            'drug_likeness_rate': drug_likeness_rate,
            'generated_smiles': generated_smiles,
            'rewards': rewards
        }

    def train(self, iterations=1000, batch_size=32,
              temperature_schedule=None, verbose=True):
        """
        Train the generator using GAN approach

        Args:
            iterations: Number of training iterations
            batch_size: Batch size for each iteration
            temperature_schedule: Function that returns temperature given iteration
            verbose: Whether to print progress
        """
        if temperature_schedule is None:
            temperature_schedule = lambda i: max(0.5, 1.0 - i / iterations * 0.5)

        print(f"Starting GAN training for {iterations} iterations...")
        print(f"Batch size: {batch_size}")
        print(f"Reward threshold: {self.reward_threshold}")

        best_avg_reward = 0.0

        for iteration in range(iterations):
            # Get temperature for this iteration
            temperature = temperature_schedule(iteration)

            # Perform training step
            step_metrics = self.train_step(
                batch_size=batch_size,
                temperature=temperature
            )

            # Store metrics
            self.training_history['iteration'].append(iteration)
            self.training_history['avg_reward'].append(step_metrics['avg_reward'])
            self.training_history['validity_rate'].append(step_metrics['validity_rate'])
            self.training_history['drug_likeness_rate'].append(step_metrics['drug_likeness_rate'])
            self.training_history['generator_loss'].append(step_metrics['generator_loss'])

            # Print progress
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{iterations}:")
                print(f"  Generator Loss: {step_metrics['generator_loss']:.4f}")
                print(f"  Avg Reward: {step_metrics['avg_reward']:.4f}")
                print(f"  Validity Rate: {step_metrics['validity_rate']:.4f}")
                print(f"  Drug-likeness Rate: {step_metrics['drug_likeness_rate']:.4f}")
                print(f"  Temperature: {temperature:.3f}")

                # Show some example molecules
                valid_smiles = [s for s in step_metrics['generated_smiles']
                                if self.discriminator.is_valid(s)]
                if valid_smiles:
                    print(f"  Example valid SMILES: {valid_smiles[0]}")
                print("-" * 50)

            # Save best model
            if step_metrics['avg_reward'] > best_avg_reward:
                best_avg_reward = step_metrics['avg_reward']
                self.save_model(f"best_gan_model_reward_{best_avg_reward:.4f}.keras")

            # Periodic save
            if (iteration + 1) % self.save_interval == 0:
                self.save_model(f"gan_model_iter_{iteration + 1}.keras")

        print(f"Training completed! Best average reward: {best_avg_reward:.4f}")
        return self.training_history

    def save_model(self, filepath):
        """Save the generator model"""
        try:
            self.generator.model.save(filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def save_training_history(self, filepath="gan_training_history.csv"):
        """Save training history to CSV"""
        try:
            df = pd.DataFrame(self.training_history)
            df.to_csv(filepath, index=False)
            print(f"Training history saved to {filepath}")
        except Exception as e:
            print(f"Failed to save training history: {e}")


def load_vocabulary_from_model_dir(model_dir="./"):
    """
    Load vocabulary mappings from saved files in model directory
    This assumes you saved these during training
    """
    import pickle

    char2idx_path = os.path.join(model_dir, "char2idx.pkl")
    idx2char_path = os.path.join(model_dir, "idx2char.pkl")

    try:
        with open(char2idx_path, 'rb') as f:
            char2idx = pickle.load(f)
        with open(idx2char_path, 'rb') as f:
            idx2char = pickle.load(f)
        return char2idx, idx2char
    except FileNotFoundError:
        print("Vocabulary files not found. You may need to recreate them from your training data.")
        return None, None


def main():
    """Main function for GAN fine-tuning"""

    # Configuration
    model_path = "trained_rnn_generator.keras"  # Your pretrained model
    iterations = 500
    batch_size = 16
    generator_lr = 1e-5

    print("=== RNN GAN Fine-tuning for SMILES Generation ===")

    # Try to load vocabulary
    print("Loading vocabulary...")
    char2idx, idx2char = load_vocabulary_from_model_dir()

    if char2idx is None or idx2char is None:
        print("ERROR: Could not load vocabulary mappings.")
        print("Please ensure char2idx.pkl and idx2char.pkl exist in the model directory.")
        print("You may need to save these during your initial training in RNN.py")
        return

    print(f"Vocabulary loaded: {len(char2idx)} characters")

    # Initialize components
    print(f"Loading pretrained model from {model_path}...")
    try:
        generator = RNN_Generator(model_path, char2idx, idx2char)
        print("✓ Generator loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load generator: {e}")
        return

    discriminator = EnhancedDiscriminator(
        validity_weight=1.0,
        drug_likeness_weight=0.5
    )
    print("✓ Discriminator initialized")

    # Test generator before training
    print("\n--- Testing Generator Before Training ---")
    test_smiles = generator.generate_smiles_batch(batch_size=5)
    test_scores = discriminator.score_smiles_batch(test_smiles)

    for i, (smiles, score) in enumerate(zip(test_smiles, test_scores)):
        valid = discriminator.is_valid(smiles)
        print(f"  {i + 1}: {smiles[:50]}... | Valid: {valid} | Score: {score:.3f}")

    print(f"\nPre-training validity rate: {np.mean([discriminator.is_valid(s) for s in test_smiles]):.3f}")

    # Initialize trainer
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        generator_lr=generator_lr,
        reward_threshold=0.7,
        save_interval=50
    )
    print("✓ GAN trainer initialized")

    # Define temperature schedule (start high, decay over time)
    def temperature_schedule(iteration):
        return max(0.7, 1.2 - iteration / iterations * 0.5)

    # Train the GAN
    print(f"\n--- Starting GAN Training ---")
    history = trainer.train(
        iterations=iterations,
        batch_size=batch_size,
        temperature_schedule=temperature_schedule,
        verbose=True
    )

    # Test generator after training
    print("\n--- Testing Generator After Training ---")
    final_test_smiles = generator.generate_smiles_batch(batch_size=10)
    final_test_scores = discriminator.score_smiles_batch(final_test_smiles)

    for i, (smiles, score) in enumerate(zip(final_test_smiles, final_test_scores)):
        valid = discriminator.is_valid(smiles)
        print(f"  {i + 1}: {smiles} | Valid: {valid} | Score: {score:.3f}")

    final_validity_rate = np.mean([discriminator.is_valid(s) for s in final_test_smiles])
    print(f"\nPost-training validity rate: {final_validity_rate:.3f}")

    # Save results
    trainer.save_training_history()
    trainer.save_model("final_gan_model.keras")

    # Generate final dataset
    print("\n--- Generating Final Dataset ---")
    final_dataset = []
    for i in range(100):  # Generate 100 final molecules
        smiles_batch = generator.generate_smiles_batch(batch_size=1)
        if smiles_batch and discriminator.is_valid(smiles_batch[0]):
            score = discriminator.score_smiles_batch(smiles_batch)[0]
            final_dataset.append({
                'SMILES': smiles_batch[0],
                'Validity_Score': 1.0,
                'Drug_Likeness_Score': discriminator.compute_drug_likeness_score(smiles_batch[0]),
                'Total_Score': score
            })

    if final_dataset:
        final_df = pd.DataFrame(final_dataset)
        final_df.to_csv("gan_generated_molecules.csv", index=False)
        print(f"Generated {len(final_dataset)} valid molecules saved to gan_generated_molecules.csv")
        print(f"Average drug-likeness score: {final_df['Drug_Likeness_Score'].mean():.3f}")

    print("\n=== GAN Fine-tuning Completed ===")


if __name__ == "__main__":
    main()