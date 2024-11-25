import unittest
import json, os, re
from pathlib import Path
import numpy as np
from Bio import PDB
from typing import Dict, Any, List
import sys
from src.selection import *
from src.antibodies import *

# Define test paths relative to test file location
TEST_DIR = Path(__file__).parent / "files"
TEST_FILES = [
    TEST_DIR / "7w9f.pdb",
    TEST_DIR / "6vyh.pdb",
    TEST_DIR / "7m3n.pdb"
]

class TestAntibodySelector(unittest.TestCase):
    """Unit tests for the AntibodySelector class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data paths and load structures."""
        cls.structures = {}
        cls.reference_metadata = {}
        cls.parser = PDB.PDBParser(QUIET=True)
        
        for pdb_file in TEST_FILES:
            pdb_path = Path(pdb_file)
            json_path = pdb_path.with_suffix('.json')
            
            # Load structure
            structure_id = pdb_path.stem
            try:
                cls.structures[structure_id] = cls.parser.get_structure(
                    structure_id, 
                    pdb_file
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load PDB file {pdb_file}: {str(e)}")
            
            # Load reference data
            try:
                with open(json_path, 'r') as f:
                    cls.reference_metadata[structure_id] = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load JSON file {json_path}: {str(e)}")

    def setUp(self):
        """Set up individual test cases with both original and renumbered selectors."""
        self.selectors = {}
        for structure_id, structure in self.structures.items():
            self.selectors[structure_id] = {
                'original': AntibodySelector(structure, renumber_structure=False),
                'renumbered': AntibodySelector(structure, renumber_structure=True)
            }

    def test_original_metadata_accuracy(self):
        """Test that original structure metadata matches reference data."""
        for structure_id in self.structures:
            reference = self.reference_metadata[structure_id]
            selector = self.selectors[structure_id]['original']
            metadata = selector.get_metadata()
            
            # Print structure info
            print(f"\nTesting original structure: {structure_id}")
            print(f"Chain starts:")
            print(f"  Heavy chain - Reference: {reference['heavy_chain_start']}, "
                  f"Test: {metadata['heavy_chain_start']}")
            print(f"  Light chain - Reference: {reference['light_chain_start']}, "
                  f"Test: {metadata['light_chain_start']}")
            
            # Test all components
            components = [
                'heavy_chain', 'light_chain', 'antigen_chain',
                'CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3'
            ]
            
            for component in components:
                with self.subTest(structure=structure_id, component=component):
                    if component == 'antigen_chain':
                        # Handle antigen chain list/string conversion
                        ref_value = ([reference[component]] 
                                   if isinstance(reference[component], str)
                                   else reference[component])
                        test_value = ([metadata[component]]
                                    if isinstance(metadata[component], str)
                                    else metadata[component])
                    else:
                        ref_value = reference[component]
                        test_value = metadata[component]
                    
                    # For CDR components, print sequence info
                    if component.startswith('CDR'):
                        print(f"\n{component}:")
                        print(f"Positions: {test_value}")
                        print(f"Sequence found: {metadata.get(f'{component}_seq', '')}")
                    
                    self.assertEqual(
                        test_value,
                        ref_value,
                        f"\nComponent: {component}\n"
                        f"Test value: {test_value}\n"
                        f"Reference value: {ref_value}"
                    )

    def _has_insertion_codes(self, positions: List[Any]) -> bool:
        """Check if a list of positions contains duplicate numbers (indicating insertion codes)."""
        if isinstance(positions[0], list):  # Handle multi-domain case
            return any(len(set(domain)) != len(domain) for domain in positions)
        return len(set(positions)) != len(positions)

    def test_renumbered_structure(self):
        """
        Test that renumbered structure:
        1. Has no insertion codes in CDR positions
        2. Maintains same sequences as original structure
        """
        for structure_id in self.structures:
            orig_selector = self.selectors[structure_id]['original']
            renum_selector = self.selectors[structure_id]['renumbered']
            
            orig_metadata = orig_selector.get_metadata()
            renum_metadata = renum_selector.get_metadata()
            
            print(f"\nTesting renumbered structure: {structure_id}")
            
            # Test CDR components
            cdr_components = ['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3']
            
            for cdr in cdr_components:
                with self.subTest(structure=structure_id, component=cdr):
                    positions = renum_metadata[cdr]
                    print(f"\n{cdr}:")
                    print(f"Positions: {positions}")
                    print(f"Sequence: {renum_metadata.get(f'{cdr}_seq', '')}")
                    
                    # Verify no insertion codes
                    self.assertFalse(
                        self._has_insertion_codes(positions),
                        f"Found insertion codes in renumbered structure for {cdr}: {positions}"
                    )
                    
                    # Verify sequences match original
                    self.assertEqual(
                        renum_metadata.get(f'{cdr}_seq', ''),
                        orig_metadata.get(f'{cdr}_seq', ''),
                        f"Sequence mismatch between original and renumbered for {cdr}"
                    )

    def test_save_renumbered(self):
        """Test saving renumbered structure."""
        test_structure = next(iter(self.structures.values()))
        output_dir = TEST_DIR / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_output.pdb"
        
        try:
            selector = AntibodySelector(
                test_structure,
                renumber_structure=True,
                save_renumbered=str(output_path)
            )
            
            self.assertTrue(output_path.exists())
            
            saved_structure = self.parser.get_structure("test", output_path)
            self.assertIsNotNone(saved_structure)
            
            # Test structure properties
            for chain in saved_structure.get_chains():
                orig_chain = next(
                    c for c in test_structure.get_chains() 
                    if c.id == chain.id
                )
                
                # Check residue count matches
                orig_residues = [r for r in orig_chain.get_residues() if PDB.is_aa(r)]
                new_residues = [r for r in chain.get_residues() if PDB.is_aa(r)]
                self.assertEqual(
                    len(new_residues),
                    len(orig_residues),
                    "Residue count mismatch in saved structure"
                )
                
                # Verify sequential numbering and no insertion codes
                prev_num = 0
                for residue in new_residues:
                    self.assertEqual(
                        residue.id[2],
                        " ",
                        f"Found insertion code '{residue.id[2]}' in renumbered structure"
                    )
                    self.assertEqual(
                        residue.id[1],
                        prev_num + 1,
                        "Non-sequential numbering found"
                    )
                    prev_num = residue.id[1]
            
        finally:
            if output_path.exists():
                output_path.unlink()
            if output_dir.exists():
                output_dir.rmdir()


if __name__ == '__main__':
    unittest.main()