from rdkit import Chem
import numpy as np

def generate_randomized_smiles(input_smiles: str, num_variations: int = 128, isomericSmiles: bool = True) -> set:
    """
    Generates a set of valid randomized SMILES strings for a given molecule.

    Args:
        input_smiles (str): The input SMILES string of the molecule.
        num_variations (int): The number of randomized SMILES to attempt to generate.
        isomericSmiles (bool): Whether to include stereochemical information in the generated SMILES.

    Returns:
        set: A set of unique, valid randomized SMILES strings.
             Returns an empty set if the input SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(input_smiles)

    if mol is None:
        print(f"Warning: Input SMILES '{input_smiles}' is invalid. Returning empty set.")
        return set()

    randomized_smiles_set = set()

    # Add the canonical SMILES as a base (it's always valid and unique)
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles, canonical=True)
    randomized_smiles_set.add(canonical_smiles)

    # Attempt to generate more randomized SMILES
    for _ in range(num_variations):
        # Chem.MolToSmiles with doRandom=True provides a simple way to get random SMILES
        # However, it doesn't guarantee a different SMILES each time, so we loop.
        # For more robust randomization, you might re-implement the atom renumbering
        # as was done in the original SmilesEnumerator.randomize_smiles.
        try:
            # Re-creating mol and renumbering atoms for true randomization (more like original code)
            # This ensures different permutations
            temp_mol = Chem.MolFromSmiles(input_smiles)
            if temp_mol is not None:
                atoms = list(range(temp_mol.GetNumAtoms()))
                np.random.shuffle(atoms)
                renumbered_mol = Chem.RenumberAtoms(temp_mol, atoms)
                random_smi = Chem.MolToSmiles(renumbered_mol, isomericSmiles=isomericSmiles, canonical=False)

                # Validate the generated SMILES by trying to convert it back to a molecule
                if Chem.MolFromSmiles(random_smi) is not None:
                    randomized_smiles_set.add(random_smi)
        except Exception as e:
            # Catch potential RDKit errors during randomization if an edge case causes an issue
            print(f"Error during SMILES randomization: {e}")
            continue

    return randomized_smiles_set

# Example Usage:
smiles1 = "OC1CCN(C2=NC(CN3CCCN(C4=C(OCCCC(O)=O)C(C)=CC5=C4N=CC=C5)CC3)=CS2)CC1"

print(f"Original SMILES: {smiles1}")
random_variations = generate_randomized_smiles(smiles1, num_variations=20)
print(f"Generated randomized SMILES (count: {len(random_variations)}):")
for s in sorted(list(random_variations)): # Sorting for consistent output order
    print(f"- {s}")
print("\n---")
