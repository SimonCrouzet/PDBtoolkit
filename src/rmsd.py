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

    if len(mobile_atoms) != len(target_atoms):
        raise ValueError("Number of atoms in mobile and target structures must be equal")

    distances = []
    for m_atom, t_atom in zip(mobile_atoms, target_atoms):
        distance = np.linalg.norm(m_atom.coord - t_atom.coord)
        distances.append(distance ** 2)

    rmsd = np.sqrt(np.mean(distances))
    return {
        'rmsd': rmsd,
        'n_atoms': len(mobile_atoms)
    }