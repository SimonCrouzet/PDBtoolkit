import copy
from Bio.PDB import Superimposer
from Bio.PDB.StructureAlignment import StructureAlignment
from Bio.PDB import PDBParser
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment, PairwiseAligner
from .cealign import CEAligner
from Bio.Data.PDBData import protein_letters_3to1_extended
import numpy as np
from .selection import StructureSelector
from .align import *

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

def sanity_check(nb_atoms_expected, structure):
    """
    Check if the number of atoms in the structure matches the expected number.
    
    :param nb_atoms_expected: Expected number of atoms
    :param structure: Structure to check
    :return: True if the number of atoms matches, False otherwise
    """
    nb_atoms = len(list(structure.get_atoms()))
    if nb_atoms != nb_atoms_expected:
        return False, nb_atoms
    return True, nb_atoms

def calculate_antibody_rmsd(mobile_structure, reference_structure, mobile_metadata, reference_metadata, all_atoms=False, align_method='cealign', calculate_entire_fab=True):
    """
    Calculate RMSD of antibodies, including antigen and entire Fab.

    :param mobile_structure: Mobile structure (from StructureSelector.select())
    :param reference_structure: Reference structure (from StructureSelector.select())
    :param mobile_metadata: Dictionary containing metadata for mobile structure
    :param reference_metadata: Dictionary containing metadata for reference structure
    :param all_atoms: If False, use only CA atoms for RMSD calculations (default: False)
    :return: Dictionary containing RMSD results
    """
    logging.info("Starting antibody RMSD calculation")
    logging.info(f"Using {'all atoms' if all_atoms else 'only CA atoms'} for RMSD calculations")
    results = {}

    atom_selector = "" if all_atoms else " and name CA"

    # Step 1: Align on antigen using sequence_align
    if isinstance(mobile_metadata['antigen_chain'], str):
        mobile_metadata['antigen_chain'] = [mobile_metadata['antigen_chain']]
    if isinstance(reference_metadata['antigen_chain'], str):
        reference_metadata['antigen_chain'] = [reference_metadata['antigen_chain']]
    antigen_selector = lambda metadata: f"chain {'+'.join(map(str, metadata['antigen_chain']))}{atom_selector}"
    mobile_antigen = StructureSelector(mobile_structure).select(antigen_selector(mobile_metadata))
    reference_antigen = StructureSelector(reference_structure).select(antigen_selector(reference_metadata))
    
    logging.debug(f"Mobile antigen atoms: {len(list(mobile_antigen.get_atoms()))}")
    logging.debug(f"Reference antigen atoms: {len(list(reference_antigen.get_atoms()))}")
    
    antigen_alignment = sequence_align(mobile_antigen, reference_antigen)
    rotation_matrix, translation_vector = antigen_alignment['rotation'], antigen_alignment['translation']
    
    logging.info(f"Antigen alignment RMSD: {antigen_alignment['rmsd']:.4f}")
    
    # Apply transformation to the entire mobile structure
    StructureSelector(mobile_structure).apply_transformation(rotation_matrix, translation_vector)
    logging.info("Applied transformation to mobile structure")

    # Calculate RMSD for antigen
    results['Antigen RMSD'] = antigen_alignment['rmsd']

    # Step 2: Align and calculate RMSD for heavy and light chains
    logging.info("Step 2: Aligning heavy and light chains")
    for chain_type in ['heavy_chain', 'light_chain']:
        logging.info(f"Aligning {chain_type}")
        chain_selector = lambda metadata: f"chain {metadata[chain_type]}{atom_selector}"
        mobile_chain = StructureSelector(mobile_structure).select(chain_selector(mobile_metadata))
        reference_chain = StructureSelector(reference_structure).select(chain_selector(reference_metadata))
        
        logging.debug(f"Mobile {chain_type} atoms: {len(list(mobile_chain.get_atoms()))}")
        logging.debug(f"Reference {chain_type} atoms: {len(list(reference_chain.get_atoms()))}")
        
        if align_method == 'cealign':
            chain_alignment = cealign(mobile_chain, reference_chain)
        elif align_method == 'tmalign':
            chain_alignment = tmalign(mobile_chain, reference_chain)
        else:
            raise ValueError(f"Unknown alignment method: {align_method}")
        chain_letter = 'H' if chain_type == 'heavy_chain' else 'L'
        results[f"Chain {chain_letter} RMSD"] = chain_alignment['rmsd']
        logging.info(f"{chain_type.capitalize()} RMSD: {chain_alignment['rmsd']:.4f}")

    # Step 3: Calculate RMSD for CDR loops using naive alignment
    logging.info("Step 3: Calculating RMSD for CDR loops")
    naive_aligner = NaiveAligner()
    
    for cdr in ['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3']:
        if cdr not in mobile_metadata:
            raise ValueError(f"CDR selection metadata of {cdr} is missing in the mobile structure")
        if cdr not in reference_metadata:
            raise ValueError(f"CDR selection metadata of {cdr} is missing in the reference structure")
        if len(reference_metadata[cdr]) == 0:
            raise ValueError(f"CDR selection metadata of {cdr} is empty in the reference structure")
        try:
            logging.info(f"Aligning {cdr}")
            chain_type = 'heavy_chain' if cdr[3] == 'H' else 'light_chain'
            if len(mobile_metadata[cdr]) == 0:
                # Less severe error, we just skip this CDR (it will trigger the except block)
                raise ValueError(f"CDR selection metadata of {cdr} is empty")
            cdr_selector = lambda metadata: f"chain {metadata[chain_type]} and resi {'+'.join(map(str, metadata[cdr]))}{atom_selector}"
            mobile_cdr = StructureSelector(mobile_structure).select(cdr_selector(mobile_metadata))
            reference_cdr = StructureSelector(reference_structure).select(cdr_selector(reference_metadata))

            check, nb_atoms = sanity_check(len(mobile_metadata[cdr]), mobile_cdr)
            if not check:
                raise ValueError(f"Number of atoms in mobile structure ({str(nb_atoms)}) does not match the expected number ({str(len(mobile_metadata[cdr]))}). Selection was {cdr_selector(mobile_metadata)}")
            check, nb_atoms = sanity_check(len(reference_metadata[cdr]), reference_cdr)
            if not check:
                raise ValueError(f"Number of atoms in reference structure ({str(nb_atoms)}) does not match the expected number ({str(len(reference_metadata[cdr]))}). Selection was {cdr_selector(reference_metadata)}")
            
            mobile_coords = [atom.coord for atom in mobile_cdr.get_atoms()]
            reference_coords = [atom.coord for atom in reference_cdr.get_atoms()]
            
            logging.debug(f"Mobile {cdr} atoms: {len(mobile_coords)}")
            logging.debug(f"Reference {cdr} atoms: {len(reference_coords)}")
            
            alignment = naive_aligner.align(mobile_coords, reference_coords)
            rmsd = get_rmsd(*get_aligned_coords(mobile_coords, reference_coords, alignment['alignment']))
            results[f"Segment {cdr} RMSD"] = rmsd
            logging.info(f"{cdr} RMSD: {rmsd:.4f}")
        except Exception as e:
            logging.warning(f"Exception occured during alignment of {cdr}: {e}")
            results[f"{cdr}"] = np.nan

    # Step 4: Calculate RMSD for the entire Fab
    if calculate_entire_fab:
        # Can be long to align, so we skip it if not needed
        logging.info("Step 4: Calculating RMSD for entire Fab")
        fab_selector = lambda metadata: f"chain {metadata['heavy_chain']}+{metadata['light_chain']}{atom_selector}"
        mobile_fab = StructureSelector(mobile_structure).select(fab_selector(mobile_metadata))
        reference_fab = StructureSelector(reference_structure).select(fab_selector(reference_metadata))
        
        logging.debug(f"Mobile Fab atoms: {len(list(mobile_fab.get_atoms()))}")
        logging.debug(f"Reference Fab atoms: {len(list(reference_fab.get_atoms()))}")
        
        if align_method == 'cealign':
            fab_alignment = cealign(mobile_fab, reference_fab)
        elif align_method == 'tmalign':
            fab_alignment = tmalign(mobile_fab, reference_fab)
        else:
            raise ValueError(f"Unknown alignment method: {align_method}")
        fab_alignment = cealign(mobile_fab, reference_fab)
        results['Entire Fab RMSD'] = fab_alignment['rmsd']
        logging.info(f"Fab RMSD: {fab_alignment['rmsd']:.4f}")
    else:
        results['Entire Fab RMSD'] = np.nan
        logging.info("Skipping calculation of entire Fab RMSD")

    logging.info("Antibody RMSD calculation completed")
    return {k: round(v, 4) for k, v in results.items()}
