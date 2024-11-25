import json
import time
from .selection import *
from .rmsd import *
from .align import *

logging.basicConfig(level=logging.INFO)

mobile_metadata = json.load(open("examples/7mtb.json"))
reference_metadata = json.load(open("examples/7mtb_polyA.json"))
parser = PDB.PDBParser(QUIET=True)
mobile_structure = parser.get_structure("mobile", "examples/7mtb.pdb")
reference_structure = parser.get_structure("reference", "examples/7mtb_polyA.pdb")

for chain_type in ['heavy_chain', 'light_chain']:
    chain_name = 'H' if chain_type == 'heavy_chain' else 'L'
    print(f"Aligning {chain_name}")
    chain_selector = lambda metadata: f"chain {metadata[chain_type]} and name CA"
    mobile_chain = StructureSelector(mobile_structure).select(chain_selector(mobile_metadata))
    reference_chain = StructureSelector(reference_structure).select(chain_selector(reference_metadata))
    start_time = time.time()
    chain_alignment_cealign = cealign(mobile_chain, reference_chain)
    cealign_time = time.time() - start_time
    chain_alignment_tmalign = tmalign(mobile_chain, reference_chain)
    tmalign_time = time.time() - start_time - cealign_time
    print(f"\tCE-Align RMSD: {chain_alignment_cealign['rmsd']:.4f} ({cealign_time:.2f} s)")
    print(f"\tTM-Align RMSD: {chain_alignment_tmalign['rmsd']:.4f} ({tmalign_time:.2f} s)")

calculate_antibody_rmsd(mobile_structure, reference_structure, mobile_metadata, reference_metadata, align_method='tmalign')


parser = PDB.PDBParser()
structure = parser.get_structure("example", "examples/7w9f.pdb")
structure_to_compare = parser.get_structure("example", "examples/7w9f_polyA.pdb")

selector = StructureSelector(structure)
selector_to_compare = StructureSelector(structure_to_compare)

sel = "chain A and resn GLY+ALA+GLU"

# Select chains A and D
result = selector.select(sel)
print(list(result.get_chains()))
print(list(result[0].get_residues()))
print("Selected structure:", result)

# Get only the atom indices for chain C
indices = selector.select(sel, index_only=True)
print("Atom indices for chain A+D:", indices)

result_to_compare = selector_to_compare.select(sel)

 # Make selections
mobile_sel = selector.select("chain A and name CA")
target_sel = selector_to_compare.select("chain A and name CA")


# Perform alignments and RMSD calculations
align_result = align(mobile_sel, target_sel)
print(f"Align RMSD: {align_result['rmsd']:.2f} Å ({align_result['n_aligned']} atoms)")
selector.apply_transformation(align_result['rotation'], align_result['translation'])


mobile_sel = selector.select("chain A and name CA")
print(f"Atoms in mobile structure: {len(list(mobile_sel.get_residues()))}")
target_sel = selector_to_compare.select("chain A and name CA")
cealign_result = cealign(mobile_sel, target_sel, transform=False, window_size=8, max_gap=30)
align_result = align(mobile_sel, target_sel, method='superimposer')
align_sequence_result = align(mobile_sel, target_sel, method='sequence')
align_tm_result = align(mobile_sel, target_sel, method='tmalign')
np.array([atom.coord for atom in cealign_result.get_atoms()])
np.array([atom.coord for atom in mobile_sel.get_atoms()])
print(f"CE-Align RMSD: {cealign_result['rmsd']:.2f} Å ({cealign_result['n_aligned']} atoms)")


rms_cur_result = rms_cur(mobile_sel, target_sel)
print(f"RMS-cur RMSD: {rms_cur_result['rmsd']:.2f} Å ({rms_cur_result['n_atoms']} atoms)")


