import logging
from Bio.Data.IUPACData import protein_letters_3to1
import Bio.PDB as PDB
from Bio import pairwise2
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

def get_sequence_index_mapping(seq1, seq2, indices1=None, indices2=None):
    """
    Create a mapping of indices from seq1 to seq2 using sequence alignment.
    Can use custom index lists for each sequence.
    
    Args:
        seq1: First sequence string
        seq2: Second sequence string
        indices1: Optional list of custom indices for seq1. If None, uses position indices
        indices2: Optional list of custom indices for seq2. If None, uses position indices
    
    Returns:
        dict: Mapping of positions from seq1 indices to seq2 indices
    """
    # Verify indices lengths if provided
    if indices1 is not None and len(indices1) != len(seq1):
        raise ValueError(f"indices1 length ({len(indices1)}) doesn't match seq1 length ({len(seq1)})")
    if indices2 is not None and len(indices2) != len(seq2):
        raise ValueError(f"indices2 length ({len(indices2)}) doesn't match seq2 length ({len(seq2)})")
    
    # If indices not provided, use position indices
    if indices1 is None:
        indices1 = list(range(len(seq1)))
    if indices2 is None:
        indices2 = list(range(len(seq2)))
    
    # Get the best global alignment
    alignments = pairwise2.align.globalxx(seq1, seq2)
    best_alignment = alignments[0]
    
    # Get aligned sequences
    aligned_seq1, aligned_seq2 = best_alignment.seqA, best_alignment.seqB
    
    # Create the mapping
    mapping = {}
    pos1 = 0  # Position in original seq1
    pos2 = 0  # Position in original seq2
    
    for i in range(len(aligned_seq1)):
        if aligned_seq1[i] != '-' and aligned_seq2[i] != '-':
            # Both positions have residues (not gaps)
            mapping[indices1[pos1]] = indices2[pos2]
            pos1 += 1
            pos2 += 1
        elif aligned_seq1[i] == '-':
            # Gap in seq1
            pos2 += 1
        else:  # aligned_seq2[i] == '-'
            # Gap in seq2
            pos1 += 1
            
    return mapping

def extract_residue_indices_with_insertion(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("temp", pdb_file)
    return extract_residue_indices_with_insertion_from_struct(structure)

def extract_residue_indices_with_insertion_from_struct(structure):
    """
    Extract residue indices with insertion codes for each chain in the structure.
    Returns them in the format "100", "100A", "101" etc.
    
    Args:
        structure: Bio.PDB Structure object
    
    Returns:
        dict: Dictionary with chain IDs as keys and lists of residue identifiers as values
              e.g., {'A': ['1', '2A', '2B', '3'], 'B': ['1', '2']}
    """
    chain_indices = {}
    for chain in structure.get_chains():
        residue_list = []
        for residue in chain:
            if PDB.is_aa(residue):
                # residue.id is a tuple: (hetero_flag, sequence_number, insertion_code)
                resid = str(residue.id[1])  # sequence number
                insertion = residue.id[2]    # insertion code
                if insertion.strip():        # if insertion code exists and isn't just whitespace
                    resid = f"{resid}{insertion}"
                residue_list.append(resid)
        if residue_list:  # Only add chains that have amino acids
            chain_indices[chain.id] = residue_list
    return chain_indices

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
