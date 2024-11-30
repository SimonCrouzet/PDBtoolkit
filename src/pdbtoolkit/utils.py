import logging
from Bio.Data.IUPACData import protein_letters_3to1
import Bio.PDB as PDB
import numpy as np

"""
Basic utils functions
Meant to be refactored and improved
"""


def get_rmsd(atoms1, atoms2):
    """
    Calculate RMSD between two lists of atoms.
    
    :param atoms1: List of atoms
    :param atoms2: List of atoms
    :return: RMSD value
    """
    if len(atoms1) != len(atoms2):
        raise ValueError("Number of atoms in atoms1 and atoms2 must be equal")
    
    if (not isinstance(atoms1, np.ndarray)) and (not isinstance(atoms2, np.ndarray)):
        coords1 = np.array([atom.coord for atom in atoms1])
        coords2 = np.array([atom.coord for atom in atoms2])
    else:
        coords1 = atoms1
        coords2 = atoms2

    distances = []
    for coord1, coord2 in zip(coords1, coords2):
        distance = np.linalg.norm(coord1 - coord2)
        distances.append(distance ** 2)

    return np.sqrt(np.mean(distances))

def three_to_one(residue):
    return protein_letters_3to1.get(residue.capitalize(), 'X')

def get_residue_indices_from_struct(structure):
    chain_indices = {}
    for chain in structure.get_chains():
        chain_indices[chain.id] = [residue.id[1] for residue in chain if PDB.is_aa(residue)]
    return chain_indices

def extract_coords(structure, atom_type):
    """Extract coordinates of specified atoms from Bio.PDB Structure.
    
    Args:
        structure: Bio.PDB Structure object
        atom_type: Atom name to extract (e.g., "CA" for alpha carbon)
    
    Returns:
        np.ndarray: Nx3 array of coordinates
    """
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':  # Skip HETATMs, waters, etc
                    continue
                try:
                    coord = residue[atom_type].get_coord()
                    coords.append(coord)
                except KeyError:
                    continue
        break  # Only use first model
    if len(coords) == 0:
        raise ValueError(f"No {atom_type} atoms found in structure")
    return np.array(coords)

def extract_sequences(pdb_file, by_chain=False):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("temp", pdb_file)
    return extract_sequences_from_struct(structure, by_chain)

def extract_sequences_from_struct(structure, by_chain=False):
    """Extract sequence from structure."""
    if by_chain:
        sequences = {}
        for chain in structure.get_chains():
            seq = "".join([three_to_one(residue.resname) for residue in chain if PDB.is_aa(residue)])
            sequences[chain.id] = seq
        return sequences
    else:
        return "".join([three_to_one(residue.resname) for residue in structure.get_residues() if PDB.is_aa(residue)])
