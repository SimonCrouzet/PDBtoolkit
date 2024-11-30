import copy
import logging
from Bio.PDB import Superimposer
from Bio.Align import PairwiseAligner
from .cealign import CEAligner
from Bio.Data.PDBData import protein_letters_3to1_extended
import numpy as np
from .selection import StructureSelector
from .tmalign import tmalign
from itertools import combinations

def align(mobile_structure, target_structure, method='superimposer', **kwargs):
    """
    Perform a flexible alignment of two structures using various methods.

    :param mobile_structure: Structure to be aligned (from StructureSelector.select())
    :param target_structure: Reference structure (from StructureSelector.select())
    :param method: Alignment method to use ('superimposer', 'ce', 'sequence', 'tm')
    :param transform: If True, apply the transformation to mobile_structure
    :param kwargs: Additional parameters for specific alignment methods
    :return: dict with alignment results
    """
    if method == 'superimposer':
        return superimposer_align(mobile_structure, target_structure)
    elif method == 'cealign':
        return cealign(mobile_structure, target_structure, **kwargs)
    elif method == 'sequence':
        return sequence_align(mobile_structure, target_structure, **kwargs)
    elif method == 'tmalign':
        return tmalign(mobile_structure, target_structure, **kwargs)
    else:
        raise ValueError(f"Unknown alignment method: {method}")

def superimposer_align(mobile_structure, target_structure, transform=False):
    """
    Perform alignment using Bio.PDB.Superimposer
    """
    mobile_atoms = list(mobile_structure.get_atoms())
    target_atoms = list(target_structure.get_atoms())
    if len(mobile_atoms) != len(target_atoms):
        raise ValueError("Number of atoms in mobile and target structures must be equal")
    sup = Superimposer()
    sup.set_atoms(target_atoms, mobile_atoms)
   
    if transform:
        sup.apply(mobile_structure.get_atoms())
    return {
        'rmsd': sup.rms,
        'n_aligned': len(mobile_atoms),
        'rotation': sup.rotran[0],
        'translation': sup.rotran[1]
    }

def sequence_align(mobile_structure, target_structure, **kwargs):
    """
    Perform alignment based on sequence using Bio.PDB.StructureAlignment
    """
    # Extract sequences and residues from structures
    mobile_seq = extract_sequence(mobile_structure)
    target_seq = extract_sequence(target_structure)
    mobile_residues = list(mobile_structure.get_residues())
    target_residues = list(target_structure.get_residues())
    logging.debug(f"Sequence lengths: mobile={len(mobile_seq)}, target={len(target_seq)}")

    # Perform sequence alignment using PairwiseAligner
    aligner = PairwiseAligner()
    # Set alignment parameters (you can customize these or pass them via kwargs)
    # aligner.match_score = kwargs.get('match_score', 2)
    # aligner.mismatch_score = kwargs.get('mismatch_score', -1)
    # aligner.open_gap_score = kwargs.get('open_gap_score', -0.5)
    # aligner.extend_gap_score = kwargs.get('extend_gap_score', -0.1)
    
    alignments = aligner.align(mobile_seq, target_seq)
    best_alignment = alignments[0]  # Get the best alignment

    # Create a mapping between mobile and target residues based on the alignment
    mobile_to_target = {}
    mobile_index = 0
    target_index = 0

    logging.debug(f"Best alignment is {best_alignment}")

    for align_mobile, align_target in zip(best_alignment[0], best_alignment[1]):
        if align_mobile != '-' and align_target != '-':
            logging.debug(f"Mobile index is {mobile_index} and target index is {target_index}")
            mobile_to_target[mobile_residues[mobile_index]] = target_residues[target_index]
            mobile_index += 1
            target_index += 1
        elif align_mobile != '-':
            mobile_index += 1
        elif align_target != '-':
            target_index += 1

    # Perform superposition based on aligned residues
    mobile_atoms = [r['CA'] for r in mobile_to_target.keys() if 'CA' in r]
    target_atoms = [r['CA'] for r in mobile_to_target.values() if 'CA' in r]

    sup = Superimposer()
    sup.set_atoms(target_atoms, mobile_atoms)

    return {
        'rmsd': sup.rms,
        'n_aligned': len(mobile_atoms),
        'rotation': sup.rotran[0],
        'translation': sup.rotran[1],
        'alignment': best_alignment,
        'alignment_score': best_alignment.score
    }

def extract_sequence(structure):
    """
    Extract amino acid sequence from a structure
    """
    return ''.join([protein_letters_3to1_extended.get(residue.get_resname(), "X") for residue in structure.get_residues() if residue.id[0] == ' '])

def cealign(mobile_structure, target_structure, transform=False, window_size=8, max_gap=30):
    """
    Perform a CE (Combinatorial Extension) alignment of two structures.
    
    :param mobile_structure: Structure to be aligned (from StructureSelector.select())
    :param target_structure: Reference structure (from StructureSelector.select())
    :param transform: If True, apply the transformation to mobile_structure
    :param window_size: CE algorithm parameter (default: 8)
    :param max_gap: CE algorithm parameter (default: 30)
    :return: aligned structure
    """
    aligner = CEAligner(window_size=window_size, max_gap=max_gap)
    aligner.set_reference(target_structure)
    return aligner.align(mobile_structure, transform=transform)


class NaiveAligner:
    def __init__(self, max_atoms=25):
        self.max_atoms = max_atoms

    def align(self, coords1, coords2):
        """
        Perform a naive alignment between two coordinate arrays.
        
        :param coords1: numpy array of shape (n, 3) representing coordinates of the first structure
        :param coords2: numpy array of shape (m, 3) representing coordinates of the second structure
        :return: dictionary containing the best alignment and total distance
        :raises ValueError: if the number of atoms in the shorter array exceeds max_atoms
        """
        if len(coords1) <= len(coords2):
            shorter, longer = coords1, coords2
            swap = False
        else:
            shorter, longer = coords2, coords1
            swap = True

        if len(longer) > self.max_atoms:
            raise ValueError(f"Number of atoms in the longer array ({len(longer)}) exceeds the maximum allowed ({self.max_atoms})")

        best_alignment = None
        best_distance = float('inf')

        # Generate all possible alignments
        for alignment in self._generate_alignments(len(shorter), len(longer)):
            distance = self._calculate_distance(shorter, longer, alignment)
            if distance < best_distance:
                best_distance = distance
                best_alignment = alignment

        # If we swapped the arrays, we need to swap back the alignment
        if swap:
            best_alignment = [(j, i) for i, j in best_alignment]

        return {
            'alignment': best_alignment,
            'total_distance': best_distance
        }

    def _generate_alignments(self, n, m):
        """
        Generate all possible alignments between arrays of length n and m,
        where n <= m, using all atoms from the shorter array (length n)
        and allowing gaps only in the longer array.
        """
        for indices in combinations(range(m), n):
            yield list(enumerate(indices))

    def _calculate_distance(self, coords1, coords2, alignment):
        """Calculate total distance for a given alignment."""
        return sum(np.linalg.norm(coords1[i] - coords2[j]) for i, j in alignment)
