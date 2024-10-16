import copy
from Bio.PDB import Superimposer
from Bio.PDB.StructureAlignment import StructureAlignment
from Bio.PDB import PDBParser
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment, PairwiseAligner
from src.cealign import CEAligner
from Bio.Data.PDBData import protein_letters_3to1_extended
import numpy as np
from src.selection import StructureSelector
from src.align import *

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

def get_aligned_coords(coords1, coords2, alignment):
    """
    Get aligned coordinates based on alignment.
    
    :param coords1: Coordinates of the first structure
    :param coords2: Coordinates of the second structure
    :param alignment: List of tuples containing indices of aligned atoms
    :return: Aligned coordinates for both structures
    """
    aligned1 = np.array([coords1[i] for (i, _) in alignment])
    aligned2 = np.array([coords2[j] for (_, j) in alignment])
    return aligned1, aligned2

def rms_cur(mobile_structure, target_structure, alignment_match=None):
    """
    Calculate RMSD between two structures without altering their positions.
    
    :param mobile_structure: First structure (from StructureSelector.select())
    :param target_structure: Second structure (from StructureSelector.select())
    :return: dict with keys 'rmsd', 'n_atoms'
    """
    if alignment_match is not None:
        try:
            mobile_atoms = list(StructureSelector(mobile_structure).select([a[0] for a in alignment_match]).get_atoms())
            target_atoms = list(StructureSelector(target_structure).select([a[1] for a in alignment_match]).get_atoms())
        except Exception as e:
            raise ValueError(f"Exception {e} occured when using alignment_match.\nAlignment match must be a list of tuples with atom indices - first index is for mobile structure and second index is for target structure")
    else:
        mobile_atoms = list(mobile_structure.get_atoms())
        target_atoms = list(target_structure.get_atoms())

    return {
        'rmsd': get_rmsd(mobile_atoms, target_atoms),
        'n_atoms': len(mobile_atoms)
    }