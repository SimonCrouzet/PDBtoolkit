# PDBtools: Protein Data Bank Tools

PDBtools is a Python package for working with protein structures from the Protein Data Bank (PDB). It specializes in structural alignments and RMSD calculations, with a focus on antibody structures. The package uses a PyMOL-inspired syntax for structure selection, making it intuitive for users familiar with PyMOL.

## Installation

PDBtools requires the following packages:

- NumPy
- Biopython
- SciPy
- Numba

## Key Features

- Multiple structural alignment methods
- RMSD (Root Mean Square Deviation) calculation
- Antibody-specific RMSD calculations
- Structure selection with PyMOL-inspired syntax

## Usage

### Structure Selection

PDBtools uses a selection syntax inspired by PyMOL, making it easy for users familiar with PyMOL to work with the package. Here's an example:

```python
from pdbtools.selection import StructureSelector

selector = StructureSelector(structure)

# Select all atoms in chain A
chain_a = selector.select("chain A")

# Select residues 10-30 in chain A
domain = selector.select("chain A and resi 10-30")

# Select all alpha carbons
ca_atoms = selector.select("name CA")

# Select all atoms within 5Å of a ligand
near_ligand = selector.select("within 5 of resn LIG")
```

This PyMOL-inspired syntax allows for intuitive and flexible selection of structural elements.

### Aligning Structures

Here's a detailed example of how to align two structures:

```python
from pdbtools.align import align
from pdbtools.selection import StructureSelector
from Bio.PDB import PDBParser

# Parse PDB files
parser = PDBParser()
structure1 = parser.get_structure("mobile", "path/to/mobile_structure.pdb")
structure2 = parser.get_structure("target", "path/to/target_structure.pdb")

# Select specific parts of the structures using PyMOL-like syntax
selector1 = StructureSelector(structure1)
selector2 = StructureSelector(structure2)

mobile_structure = selector1.select("chain A and resi 1-100 and name CA")
target_structure = selector2.select("chain A and resi 1-100 and name CA")

# Perform alignment
result = align(mobile_structure, target_structure, method='cealign')

print(f"RMSD: {result['rmsd']:.2f} Å")
print(f"Number of aligned atoms: {result['n_aligned']}")

# Apply the transformation to the entire mobile structure if needed
selector1.apply_transformation(result['rotation'], result['translation'])

# You can now save the transformed structure or perform further analysis
```

This example demonstrates how to:
1. Load PDB structures
2. Select specific parts of the structures using PyMOL-inspired syntax
3. Perform an alignment using the CE algorithm
4. Access the alignment results
5. Apply the transformation to the entire structure

### Calculating Antibody RMSD

PDBtools includes a specialized function for calculating RMSD values for different parts of an antibody structure. Here's how to use it:

```python
from pdbtools.rmsd import calculate_antibody_rmsd
from pdbtools.selection import StructureSelector
from Bio.PDB import PDBParser

# Parse PDB files
parser = PDBParser()
mobile_structure = parser.get_structure("mobile", "path/to/mobile_antibody.pdb")
reference_structure = parser.get_structure("reference", "path/to/reference_antibody.pdb")

# Define metadata for each structure
mobile_metadata = {
    "heavy_chain": "H",
    "light_chain": "L",
    "antigen_chain": "A",
    "CDRH1": [26, 27, 28, 29, 30, 31, 32],
    "CDRH2": [52, 53, 54, 55, 56],
    "CDRH3": [95, 96, 97, 98, 99, 100, 101],
    "CDRL1": [24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
    "CDRL2": [50, 51, 52, 53, 54, 55],
    "CDRL3": [89, 90, 91, 92, 93, 94, 95, 96]
}

reference_metadata = {
    # ... similar to mobile_metadata, but for the reference structure
}

# Calculate RMSD
rmsd_results = calculate_antibody_rmsd(
    mobile_structure, 
    reference_structure, 
    mobile_metadata, 
    reference_metadata
)

# Print results
for region, rmsd in rmsd_results.items():
    print(f"{region}: {rmsd:.2f} Å")
```

This example shows how to:
1. Load antibody structures
2. Define metadata for each structure, including chain identifiers and CDR ranges as lists
3. Calculate RMSD values for different parts of the antibody
4. Access and display the results

The `calculate_antibody_rmsd` function provides RMSD values for:
- The entire Fab
- Heavy and light chains separately
- Each CDR loop
- The antigen (if present)

## License

PDBtools is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact & Contributing

We welcome contributions to PDBtools! If you have ideas for new features or improvements, please open an issue or submit a pull request on our GitHub repository.

For questions, suggestions, or issues, please contact the project maintainer.
You can also open an issue on our GitHub repository for bug reports or feature requests.
