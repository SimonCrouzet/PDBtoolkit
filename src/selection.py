from copy import deepcopy
import logging
from Bio import PDB
import re

import numpy as np

class StructureSelector:
    def __init__(self, structure):
        self.structure = structure

    def select(self, selection_string, index_only=False, state=1):
        logging.debug(f"Selecting atoms with: {selection_string}, state: {state}")
        parsed_selection = self._parse(selection_string)
        
        if index_only:
            return [atom.serial_number for atom in self._get_atoms(state) if self._evaluate(parsed_selection, atom)]
        else:
            return self._create_new_structure(parsed_selection, state)
        
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
        if tokens and tokens[0] == '(':
            tokens.pop(0)
            expr = self._parse_or(tokens)
            assert tokens.pop(0) == ')'
            return expr
        
        selector_type = tokens.pop(0)
        selector_value = tokens.pop(0)
        return {'type': selector_type, 'value': selector_value}

    def _evaluate(self, node, atom):
        if 'op' in node:
            if node['op'] == 'and':
                return self._evaluate(node['left'], atom) and self._evaluate(node['right'], atom)
            elif node['op'] == 'or':
                return self._evaluate(node['left'], atom) or self._evaluate(node['right'], atom)
            elif node['op'] == 'not':
                return not self._evaluate(node['expr'], atom)
        else:
            return self._evaluate_leaf(node, atom)

    def _evaluate_leaf(self, leaf, atom):
        selector_type = leaf['type']
        selector_value = leaf['value']

        if selector_type == 'chain':
            return atom.parent.parent.id in self._parse_items(selector_value)
        elif selector_type == 'resi':
            return atom.parent.id[1] in self._parse_range(selector_value)
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
                start, end = map(int, part.split('-'))
                result.update(range(start, end + 1))
            else:
                result.add(int(part))
        return result

    def _create_new_structure(self, parsed_selection, state=1):
        new_structure = PDB.Structure.Structure(self.structure.id)
        new_model = PDB.Model.Model(0)  # Create a single model
        new_structure.add(new_model)

        atoms_to_add = list(self._get_atoms(state))
       
        for atom in atoms_to_add:
            if self._evaluate(parsed_selection, atom):
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


