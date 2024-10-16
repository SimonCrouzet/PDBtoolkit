from copy import deepcopy
from Bio import PDB
import re

import numpy as np

class StructureSelector:
    def __init__(self, structure):
        self.structure = structure

    def select(self, selection_string, index_only=False):
        parsed_selection = self._parse(selection_string)
        
        if index_only:
            return [atom.serial_number for atom in self.structure.get_atoms() if self._evaluate(parsed_selection, atom)]
        else:
            return self._create_new_structure(parsed_selection)
        
    def apply_transformation(self, rotation_matrix, translation_vector):
        """
        Apply a transformation (rotation and translation) to the selected structure.
        
        :param rotation_matrix: 3x3 rotation matrix
        :param translation_vector: 3D translation vector
        """
        for atom in self.structure.get_atoms():
            atom.transform(rotation_matrix, translation_vector)

    def get_coordinates(self):
        """
        Return a numpy array of atomic coordinates.
        
        :return: numpy array of shape (n_atoms, 3)
        """
        return np.array([atom.coord for atom in self.structure.get_atoms()])

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

    def _create_new_structure(self, parsed_selection):
        new_structure = PDB.Structure.Structure(self.structure.id)
        
        for model in self.structure:
            new_model = PDB.Model.Model(model.id)
            for chain in model:
                new_chain = PDB.Chain.Chain(chain.id)
                for residue in chain:
                    new_residue = PDB.Residue.Residue(residue.id, residue.resname, residue.segid)
                    for atom in residue:
                        if self._evaluate(parsed_selection, atom):
                            new_residue.add(atom.copy())
                    if len(new_residue):
                        new_chain.add(new_residue)
                if len(new_chain):
                    new_model.add(new_chain)
            if len(new_model):
                new_structure.add(new_model)
        
        return new_structure

