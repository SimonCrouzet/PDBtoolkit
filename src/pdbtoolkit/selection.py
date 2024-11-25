from copy import deepcopy
import logging
from typing import Optional, Dict
from Bio import PDB
import re

import numpy as np

from copy import deepcopy


class StructureSelector:
    def __init__(self, structure):
        self.structure = structure

    def select(self, selection_string, index_only=False, state=1, include_insertions=False):
        logging.debug(f"Selecting atoms with: {selection_string}, state: {state}, include_insertions: {include_insertions}")
        parsed_selection = self._parse(selection_string)
        
        if index_only:
            return [atom.serial_number for atom in self._get_atoms(state) if self._evaluate(parsed_selection, atom, include_insertions)]
        else:
            return self._create_new_structure(parsed_selection, state, include_insertions)
        
    def apply_transformation(self, rotation_matrix, translation_vector, state=1):
        """
        Apply a transformation (rotation and translation) to the selected structure.
        
        :param rotation_matrix: 3x3 rotation matrix
        :param translation_vector: 3D translation vector
        """
        for atom in self._get_atoms(state):
            atom.transform(rotation_matrix, translation_vector)

    def get_coordinates(self, state=1):
        """
        Return a numpy array of atomic coordinates.
        
        :param state: State to use for coordinate extraction (default: 1)
        :return: numpy array of shape (n_atoms, 3)
        """
        return np.array([atom.coord for atom in self._get_atoms(state)])
    
    def get_structure(self):
        return self.structure

    def _get_atoms(self, state=1):
        """
        Get atoms from the specified state.
        
        :param state: State to use for atom selection (default: 1)
        :return: iterator of atoms
        """
        if len(self.structure) > 1:
            logging.warning(f"Multiple models found ({len(self.structure)}), using the model {state}")
        count_atoms = 0
        for model in self.structure:
            if model.id == state - 1:  # Biopython uses 0-based indexing for models
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            count_atoms += 1
                            yield atom
                return  # Stop after processing the specified state

    def _parse(self, selection_string):
        tokens = re.findall(r'\(|\)|and|or|not|\S+', selection_string)
        return self._parse_or(tokens)

    def _parse_or(self, tokens):
        left = self._parse_and(tokens)
        while tokens and tokens[0] == 'or':
            tokens.pop(0)
            right = self._parse_and(tokens)
            left = {'op': 'or', 'left': left, 'right': right}
        return left

    def _parse_and(self, tokens):
        left = self._parse_not(tokens)
        while tokens and tokens[0] == 'and':
            tokens.pop(0)
            right = self._parse_not(tokens)
            left = {'op': 'and', 'left': left, 'right': right}
        return left

    def _parse_not(self, tokens):
        if tokens and tokens[0] == 'not':
            tokens.pop(0)
            return {'op': 'not', 'expr': self._parse_not(tokens)}
        return self._parse_atom(tokens)

    def _parse_atom(self, tokens):
        if not tokens:
            raise ValueError("Unexpected end of selection string")
            
        if tokens[0] == '(':
            tokens.pop(0)
            if not tokens:
                raise ValueError("Unclosed parenthesis in selection string")
            expr = self._parse_or(tokens)
            if not tokens:
                raise ValueError("Missing closing parenthesis in selection string")
            if tokens[0] != ')':
                raise ValueError(f"Expected ')', found '{tokens[0]}'")
            tokens.pop(0)
            return expr
        
        if len(tokens) < 2:
            raise ValueError(f"Incomplete selector: {tokens}")
            
        selector_type = tokens.pop(0)
        selector_value = tokens.pop(0)
        return {'type': selector_type, 'value': selector_value}

    def _evaluate(self, node, atom, include_insertions):
        if 'op' in node:
            if node['op'] == 'and':
                return self._evaluate(node['left'], atom, include_insertions) and self._evaluate(node['right'], atom, include_insertions)
            elif node['op'] == 'or':
                return self._evaluate(node['left'], atom, include_insertions) or self._evaluate(node['right'], atom, include_insertions)
            elif node['op'] == 'not':
                return not self._evaluate(node['expr'], atom, include_insertions)
        else:
            return self._evaluate_leaf(node, atom, include_insertions)

    def _evaluate_leaf(self, leaf, atom, include_insertions):
        selector_type = leaf['type']
        selector_value = leaf['value']

        if selector_type == 'chain':
            return atom.parent.parent.id in self._parse_items(selector_value)
        elif selector_type == 'resi':
            return self._compare_residue_id(atom.parent.id, self._parse_range(selector_value), include_insertions)
        elif selector_type == 'resn':
            return atom.parent.resname in self._parse_items(selector_value)
        elif selector_type == 'name':
            return atom.name in self._parse_items(selector_value)
        elif selector_type == 'elem':
            return atom.element in self._parse_items(selector_value)
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")

    def _parse_items(self, items_str):
        return set(items_str.split('+'))

    def _parse_range(self, range_str):
        result = set()
        for part in range_str.split('+'):
            if '-' in part:
                start, end = part.split('-')
                start_num, start_insert = self._split_residue_id(start)
                end_num, end_insert = self._split_residue_id(end)
                for i in range(start_num, end_num + 1):
                    if i == start_num and start_insert:
                        result.add((i, start_insert))
                    elif i == end_num and end_insert:
                        result.add((i, end_insert))
                    else:
                        result.add((i, ''))
            else:
                num, insert = self._split_residue_id(part)
                result.add((num, insert))
        return result
    
    def _split_residue_id(self, res_id):
        match = re.match(r'(\d+)(\w?)', res_id)
        if match:
            return int(match.group(1)), match.group(2)
        else:
            raise ValueError(f"Invalid residue ID format: {res_id}")

    def _compare_residue_id(self, atom_res_id, parsed_range, include_insertions):
        atom_res_num, atom_insert = atom_res_id[1], atom_res_id[2]
        for range_res_num, range_insert in parsed_range:
            if atom_res_num == range_res_num:
                if range_insert.strip() == '':
                    # Always select residues without insertion codes
                    if atom_insert.strip() == '':
                        return True
                    # If include_insertions is True, also select residues with insertion codes
                    elif include_insertions:
                        return True
                elif range_insert == atom_insert:
                    return True
        return False

    def _create_new_structure(self, parsed_selection, state=1, include_insertions=False):
        new_structure = PDB.Structure.Structure(self.structure.id)
        new_model = PDB.Model.Model(0)  # Create a single model
        new_structure.add(new_model)

        atoms_to_add = list(self._get_atoms(state))
       
        for atom in atoms_to_add:
            if self._evaluate(parsed_selection, atom, include_insertions):
                chain_id = atom.parent.parent.id
                residue_id = atom.parent.id
               
                if chain_id not in new_model:
                    new_model.add(PDB.Chain.Chain(chain_id))
               
                chain = new_model[chain_id]
                if residue_id not in chain:
                    new_residue = PDB.Residue.Residue(residue_id, atom.parent.resname, atom.parent.segid)
                    chain.add(new_residue)
               
                new_atom = atom.copy()
                chain[residue_id].add(new_atom)

        logging.debug(f"New structure created with {len(list(new_structure.get_atoms()))} atoms")
        return new_structure
    
    @classmethod
    def index_as_antibody(cls, instance: 'StructureSelector', 
                         scheme: str = 'chothia',
                         allow_multidomains: bool = False,
                         specify_variable_chains: Optional[Dict[str, str]] = None) -> 'AntibodySelector':
        """
        Convert a StructureSelector instance to an AntibodySelector instance.
        This allows for antibody-specific analysis while preserving the current structure state.
        
        Parameters:
        -----------
        instance : StructureSelector
            The StructureSelector instance to convert
        scheme : str, default='chothia'
            The numbering scheme to use ('chothia', 'kabat', 'imgt', or 'aho')
        allow_multidomains : bool, default=False
            Whether to allow multiple variable domains in a single chain
        specify_variable_chains : Optional[Dict[str, str]], default=None
            Optional dictionary to manually specify variable chain IDs
            Must be in the format {'heavy': 'H', 'light': 'L'}
            
        Returns:
        --------
        AntibodySelector
            A new AntibodySelector instance with the same structure state
            
        Examples:
        --------
        >>> selector = StructureSelector(structure)
        >>> # Do some structure manipulation...
        >>> ab_selector = StructureSelector.index_as_antibody(selector)
        >>> # Now can use antibody-specific methods
        >>> cdr3 = ab_selector.select_cdr('3', 'heavy')
        
        Notes:
        ------
        This method creates a new AntibodySelector instance while preserving
        the current state of the structure. This is useful when you need to
        switch to antibody-specific analysis after general structure manipulation.
        """

        from .antibodies import AntibodySelector  # Import only when method is called
        
        # Create new AntibodySelector instance with copied structure
        antibody_selector = AntibodySelector(
            deepcopy(instance.structure),
            scheme=scheme,
            allow_multidomains=allow_multidomains,
            specify_variable_chains=specify_variable_chains
        )
        
        return antibody_selector

# struct = PDB.PDBParser(QUIET=True).get_structure('mobile', '/work/lpdi/users/sjcrouze/RFdiffusion/benchmark/data/pred/7m3n_cryoEM_variable_mean_w100_zscore_Fab_9.pdb')
# metadata = AbMeta
# struct = PDB.PDBParser(QUIET=True).get_structure('reference', '/work/lpdi/users/sjcrouze/Data/Benchmark/7lrs.pdb')
# selector = StructureSelector(struct)
# insert_sel = [95,96,97,98,99,100,"100A","100B","100C","100D","100E","100F",101,102]
# new_sel = [24,25,26,27,"27A",28,29,30,31,32,33,34]
# resi_str = '+'.join([str(i) for i in insert_sel])
# resi_str = '+'.join([str(i) for i in new_sel])
# selected_structure = selector.select(f"resi {resi_str} and chain E")
# len(list(selected_structure.get_residues()))
# len(new_sel)