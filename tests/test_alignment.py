import unittest
import pymol
from pymol import cmd
from Bio.PDB import PDBParser
from src.selection import StructureSelector
from src.align import align, cealign
from src.rmsd import rms_cur
from src.tmalign import tmalign
import numpy as np
import time

class TestAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Initializing PyMOL...")
        start_time = time.time()
        pymol.finish_launching(['pymol', '-qc'])  # Quiet and no GUI
        print(f"PyMOL initialized in {time.time() - start_time:.2f} seconds")

    def setUp(self):
        print("\nSetting up test...")
        start_time = time.time()
        self.parser = PDBParser()
        print("Loading structures...")
        self.structure_fixed = self.parser.get_structure("7w9f", "examples/7w9f.pdb")
        self.structure_mobile = self.parser.get_structure("7w9f_polyA", "examples/7w9f_polyA.pdb")
        
        print("Loading structures into PyMOL...")
        cmd.load("examples/7w9f.pdb", "7w9f")
        cmd.load("examples/7w9f_polyA.pdb", "7w9f_polyA")
        print(f"Setup completed in {time.time() - start_time:.2f} seconds")

    def tearDown(self):
        print("Cleaning up PyMOL objects...")
        cmd.delete('all')

    def compare_alignment_results(self, custom_result, pymol_result, method_name):
        if isinstance(pymol_result, float):
            pymol_rmsd = pymol_result
            pymol_n_aligned = np.nan
        elif isinstance(pymol_result, list) or isinstance(pymol_result, tuple):
            pymol_rmsd = pymol_result[0]
            pymol_n_aligned = pymol_result[1]
        elif isinstance(pymol_result, dict):
            pymol_rmsd = pymol_result['RMSD']
            pymol_n_aligned = pymol_result['alignment_length']
        else:
            raise ValueError(f"Unknown PyMOL result type: {type(pymol_result)}")
        custom_rmsd = custom_result['rmsd']
        custom_n_aligned = custom_result.get('alignment_length', custom_result.get('n_aligned', 0))

        print(f"{method_name} Custom RMSD: {custom_rmsd:.4f}, PyMOL RMSD: {pymol_rmsd:.4f}")
        print(f"{method_name} Custom n_aligned: {custom_n_aligned}, PyMOL n_aligned: {pymol_n_aligned}")

        rmsd_match = np.isclose(custom_rmsd, pymol_rmsd, atol=0.01)
        n_aligned_match = custom_n_aligned == pymol_n_aligned

        if rmsd_match and n_aligned_match:
            print(f"✅ {method_name} test PASSED")
            return True
        else:
            print(f"❌ {method_name} test FAILED")
            if not rmsd_match:
                print(f"  RMSD mismatch: Custom {custom_rmsd:.4f} vs PyMOL {pymol_rmsd:.4f}")
            if not n_aligned_match:
                print(f"  N_aligned mismatch: Custom {custom_n_aligned} vs PyMOL {pymol_n_aligned}")
            return False

    def run_alignment_test(self, method_name, fixed_chain, mobile_chain, alignment_match=None):
        print(f"\nRunning {method_name} for {mobile_chain} to {fixed_chain}...")
        start_time = time.time()

        print("Selecting chains...")
        selector_fixed = StructureSelector(self.structure_fixed)
        fixed = selector_fixed.select(f"chain {fixed_chain}")
        selector_mobile = StructureSelector(self.structure_mobile)
        mobile = selector_mobile.select(f"chain {mobile_chain}")

        print(f"Running custom {method_name}...")
        custom_start = time.time()
        if method_name == "align":
            custom_result = align(mobile, fixed)
        elif method_name == "cealign":
            custom_result = cealign(mobile, fixed)
        elif method_name == "tmalign":
            custom_result = tmalign(mobile, fixed)
        elif method_name == "rms_cur":
            custom_result = rms_cur(mobile, fixed, alignment_match)
        custom_time = time.time() - custom_start
        print(f"Custom {method_name} completed in {custom_time:.2f} seconds")

        print(f"Running PyMOL {method_name}...")
        pymol_start = time.time()
        if method_name == "align":
            pymol_result = cmd.align(f"7w9f_polyA and chain {mobile_chain}", f"7w9f and chain {fixed_chain}")
        elif method_name == "cealign":
            pymol_result = cmd.cealign(f"7w9f and chain {fixed_chain}", f"7w9f_polyA and chain {mobile_chain}", window=8, gap_max=30, d0=3.0, d1=4.0)
        elif method_name == "tmalign":
            pymol_result = cmd.super(f"7w9f_polyA and chain {mobile_chain}", f"7w9f and chain {fixed_chain}")
        elif method_name == "rms_cur":
            if alignment_match:
                cmd.select("mobile_match", f"7w9f_polyA and chain {mobile_chain} and resi {'+'.join(str(i) for i, _ in alignment_match)}")
                cmd.select("target_match", f"7w9f and chain {fixed_chain} and resi {'+'.join(str(i) for _, i in alignment_match)}")
                pymol_result = cmd.rms_cur("mobile_match", "target_match")
            else:
                pymol_result = cmd.rms_cur(f"7w9f_polyA and chain {mobile_chain}", f"7w9f and chain {fixed_chain}")
        pymol_time = time.time() - pymol_start
        print(f"PyMOL {method_name} completed in {pymol_time:.2f} seconds")

        success = self.compare_alignment_results(custom_result, pymol_result, method_name)
        print(f"{method_name} test completed in {time.time() - start_time:.2f} seconds")
        return success

    def test_align_chain_E(self):
        self.assertTrue(self.run_alignment_test("align", "E", "E"))

    def test_cealign_chain_E(self):
        self.assertTrue(self.run_alignment_test("cealign", "E", "E"))

    def test_tmalign_chain_E(self):
        self.assertTrue(self.run_alignment_test("tmalign", "E", "E"))

    def test_rms_cur_chain_E(self):
        self.assertTrue(self.run_alignment_test("rms_cur", "E", "E"))

    def test_cealign_chain_A_to_H(self):
        result = self.run_alignment_test("cealign", "A", "H")
        self.assertTrue(result)
        return result

    def test_tmalign_chain_A_to_H(self):
        self.assertTrue(self.run_alignment_test("tmalign", "A", "H"))

    def test_rms_cur_chain_A_to_H(self):
        cealign_result = self.test_cealign_chain_A_to_H()
        if cealign_result:
            alignment_match = cealign_result.get('alignment', [])
            self.assertTrue(self.run_alignment_test("rms_cur", "A", "H", alignment_match))
        else:
            self.fail("Couldn't get alignment match from cealign for A to H")

    def test_cealign_chain_B_to_L(self):
        result = self.run_alignment_test("cealign", "B", "L")
        self.assertTrue(result)
        return result

    def test_tmalign_chain_B_to_L(self):
        self.assertTrue(self.run_alignment_test("tmalign", "B", "L"))

    def test_rms_cur_chain_B_to_L(self):
        cealign_result = self.test_cealign_chain_B_to_L()
        if cealign_result:
            alignment_match = cealign_result.get('alignment', [])
            self.assertTrue(self.run_alignment_test("rms_cur", "B", "L", alignment_match))
        else:
            self.fail("Couldn't get alignment match from cealign for B to L")

if __name__ == '__main__':
    unittest.main(verbosity=2)