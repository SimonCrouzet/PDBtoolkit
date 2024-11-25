import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Data.PDBData import protein_letters_3to1_extended

def tmalign(mobile_structure, target_structure, mol_type=0):
    """
    Perform TM-align between mobile and target structures.
    
    :param mobile_structure: Structure to be aligned (from StructureSelector.select())
    :param target_structure: Reference structure (from StructureSelector.select())
    :param mol_type: 0 for protein, 1 for RNA
    :return: Dict containing alignment results
    """
    # Extract sequences and coordinates
    mobile_seq = extract_sequence(mobile_structure)
    target_seq = extract_sequence(target_structure)
    mobile_coords = np.array([res['CA'].coord for res in mobile_structure.get_residues() if 'CA' in res])
    target_coords = np.array([res['CA'].coord for res in target_structure.get_residues() if 'CA' in res])
    
    # Initial alignment using sequence
    aligner = PairwiseAligner()
    initial_alignment = aligner.align(mobile_seq, target_seq)[0]
    alignment = convert_alignment_to_pairs(initial_alignment, len(mobile_seq), len(target_seq))
    
    # Calculate initial d0
    L_target = len(target_coords)
    d0 = calculate_d0(L_target, mol_type)
    
    # Iterative refinement
    max_iterations = 20
    prev_score = -np.inf
    
    for _ in range(max_iterations):
        # Perform superposition based on aligned residues
        mobile_aligned, target_aligned = get_aligned_coords(mobile_coords, target_coords, alignment)
        
        # Superimpose using Kabsch algorithm
        rotation, translation = kabsch(mobile_aligned, target_aligned)
        
        # Calculate TM-score
        tm_score = tmscore(mobile_coords, target_coords, rotation, translation, L_target, d0)
        
        if tm_score - prev_score < 1e-4:
            break
        
        prev_score = tm_score
        
        # Update alignment
        transformed_mobile = np.dot(mobile_coords, rotation.T) + translation
        alignment = update_alignment(transformed_mobile, target_coords)

    final_tm_score = tmscore(mobile_coords, target_coords, rotation, translation, L_target, d0)
    
    return {
        'rmsd': np.sqrt(np.mean(np.sum((mobile_aligned - target_aligned)**2, axis=1))),
        'n_aligned': len(alignment),
        'rotation': rotation,
        'translation': translation,
        'alignment': alignment,
        'tm_score': final_tm_score
    }

def extract_sequence(structure):
    """Extract sequence from structure."""
    return ''.join([protein_letters_3to1_extended.get(residue.get_resname(), "X") for residue in structure.get_residues() if residue.id[0] == ' '])

def convert_alignment_to_pairs(alignment, len_mobile, len_target):
    """Convert string alignment to list of index pairs."""
    pairs = []
    i, j = 0, 0
    for a, b in zip(*alignment):
        if a != '-' and b != '-':
            pairs.append((i, j))
        if a != '-':
            i += 1
        if b != '-':
            j += 1
    return pairs

def calculate_d0(L, mol_type):
    """Calculate d0 based on sequence length and molecule type."""
    if mol_type == 1:  # RNA
        if L <= 11:
            return 0.3
        elif L <= 15:
            return 0.4
        elif L <= 19:
            return 0.5
        elif L <= 23:
            return 0.6
        elif L < 30:
            return 0.7
        else:
            return 0.6 * (L - 0.5)**(1/2) - 2.5
    else:  # Protein
        return 1.24 * (L - 15)**(1/3) - 1.8 if L > 21 else 0.5

def get_aligned_coords(coords1, coords2, alignment):
    """Get aligned coordinates based on the current alignment."""
    # Create a mapping between mobile and target residues based on the alignment
    coords1_to_coords2 = {}
    index1 = 0
    index2 = 0

    for align_mobile, align_target in zip(alignment[0], alignment[1]):
        if align_mobile != '-' and align_target != '-':
            coords1_to_coords2[index1] = index2
            index1 += 1
            index2 += 1
        elif align_mobile != '-':
            index1 += 1
        elif align_target != '-':
            index2 += 1
    coords1_aligned = np.array([coords1[i] for i in coords1_to_coords2.keys()])
    coords2_aligned = np.array([coords2[i] for i in coords1_to_coords2.values()])
    return coords1_aligned, coords2_aligned
def kabsch(P, Q):
    """
    Kabsch algorithm for finding the optimal rotation matrix.
    """
    C = np.dot(P.T, Q)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    rotation = np.dot(V, W)
    translation = np.mean(Q, axis=0) - np.dot(np.mean(P, axis=0), rotation)
    return rotation, translation

def tmscore(mobile_coords, target_coords, rotation, translation, L_target, d0):
    """Calculate TM-score."""
    transformed_mobile = np.dot(mobile_coords, rotation.T) + translation
    distances = np.linalg.norm(transformed_mobile[:, np.newaxis] - target_coords, axis=2)
    min_distances = np.min(distances, axis=0)
    return np.sum(1 / (1 + (min_distances / d0)**2)) / L_target

def update_alignment(transformed_mobile, target_coords, cutoff=8.0):
    """Update alignment based on spatial proximity."""
    distances = np.linalg.norm(transformed_mobile[:, np.newaxis] - target_coords, axis=2)
    new_alignment = []
    for i in range(len(transformed_mobile)):
        j = np.argmin(distances[i])
        if distances[i, j] < cutoff:
            new_alignment.append((i, j))
    return new_alignment