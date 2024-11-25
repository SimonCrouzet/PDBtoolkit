import re, os
from Bio import PDB
from Bio.PDB import PDBParser, Structure, Chain, Model
from Bio.Data.IUPACData import protein_letters_3to1
import abnumber
import logging
from typing import List, Union, Optional, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

from .selection import StructureSelector

class AntibodySelector(StructureSelector):
    """
    A specialized selector for antibody structures that extends the base StructureSelector
    and integrates antibody metadata functionality. Provides methods for analyzing and
    selecting specific antibody regions like CDRs and handling antibody-specific
    numbering schemes.
    """
    def __init__(self, structure: PDB.Structure.Structure, scheme: str = 'chothia', 
                 allow_multidomains: bool = False, 
                 specify_variable_chains: Optional[Dict[str, str]] = None,
                 renumber_structure: bool = True,
                 save_renumbered: Optional[str] = None):
        """
        Initialize the AntibodySelector.
        
        Parameters:
        -----------
        structure : Bio.PDB.Structure.Structure
            The antibody structure to analyze
        scheme : str, optional
            Numbering scheme to use (default: 'chothia')
        allow_multidomains : bool, optional
            Whether to allow multiple domains (default: False)
        specify_variable_chains : Optional[Dict[str, str]], optional
            Dictionary specifying chain IDs for variable regions
        renumber_structure : bool, optional
            Whether to renumber residues sequentially removing insertion codes (default: True)
        save_renumbered : Optional[str], optional
            Path where to save the renumbered structure (default: None)
            Only used if renumber_structure is True
            Can be either a .pdb or .cif file path
        """
        # Store original structure and configuration
        self.original_structure = structure
        self.renumber_structure = renumber_structure
        self.scheme = scheme
        self.allow_multidomains = allow_multidomains

        # Mappings between original and renumbered positions
        self.new_to_original = defaultdict(dict)  # {chain_id: {new_num: (orig_num, insert_code)}}
        self.original_to_new = defaultdict(dict)  # {chain_id: {(orig_num, insert_code): new_num}}
        
        # Create working structures
        if renumber_structure:
            self.internal_structure = self._renumber_structure(structure)
            if save_renumbered is not None:
                self._save_structure(self.internal_structure, save_renumbered)
        else:
            self.internal_structure = structure.copy()
            if save_renumbered is not None:
                logging.warning("save_renumbered option ignored because renumber_structure is False")
        
        # Initialize with the appropriate structure
        super().__init__(self.internal_structure)
        
        # Validate specify_variable_chains if provided
        if specify_variable_chains is not None:
            self._validate_variable_chains_dict(specify_variable_chains)
            self.specify_variable_chains = specify_variable_chains
        else:
            self.specify_variable_chains = None
            
        self._initialize_metadata()

    def _save_structure(self, structure: PDB.Structure.Structure, filepath: str):
        """
        Save the structure to a file.
        
        Parameters:
        -----------
        structure : Bio.PDB.Structure.Structure
            Structure to save
        filepath : str
            Path where to save the structure (.pdb or .cif extension)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Get file extension
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        # Initialize IO writer
        io = PDB.PDBIO()
        io.set_structure(structure)
        
        # Save structure based on file extension
        if ext == '.pdb':
            io.save(filepath)
        elif ext == '.cif':
            from Bio.PDB import MMCIFIO
            io = MMCIFIO()
            io.set_structure(structure)
            io.save(filepath)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Use .pdb or .cif")

    def _renumber_structure(self, structure: PDB.Structure.Structure) -> PDB.Structure.Structure:
        """
        Create a new structure with sequential residue numbering, removing insertion codes.
        """
        new_structure = PDB.Structure.Structure(structure.id)
        new_model = PDB.Model.Model(0)
        new_structure.add(new_model)
        
        for chain in structure.get_chains():
            new_chain = PDB.Chain.Chain(chain.id)
            new_model.add(new_chain)
            
            # Get only amino acid residues and sort them
            residues = [res for res in chain if PDB.is_aa(res)]
            residues.sort(key=lambda x: (x.id[1], x.id[2]))
            
            # Renumber residues sequentially
            for new_number, residue in enumerate(residues, start=1):
                new_residue = residue.copy()
                original_pos = (residue.id[1], residue.id[2])
                new_id = (residue.id[0], new_number, " ")
                new_residue.id = new_id
                new_chain.add(new_residue)
                
                # Store mappings in both directions
                self.new_to_original[chain.id][new_number] = original_pos
                self.original_to_new[chain.id][original_pos] = new_number
        
        return new_structure

    def _get_residue_indices_from_struct(self) -> Dict[str, List[int]]:
        """Get residue indices for each chain in the structure."""
        chain_indices = {}
        for chain in self.internal_structure.get_chains():
            chain_indices[chain.id] = [
                residue.id[1] for residue in chain if PDB.is_aa(residue)
            ]
        return chain_indices

    def _extract_sequences(self) -> Dict[str, str]:
        """Extract amino acid sequences from each chain in the structure."""
        sequences = {}
        for chain in self.internal_structure.get_chains():
            seq = "".join([
                protein_letters_3to1.get(residue.resname.capitalize(), 'X')
                for residue in chain if PDB.is_aa(residue)
            ])
            sequences[chain.id] = seq
        return sequences

    def _validate_variable_chains_dict(self, chains_dict: Dict[str, str]):
        """Validate the specify_variable_chains dictionary format and content."""
        # Check dictionary keys
        required_keys = {'heavy', 'light'}
        if set(chains_dict.keys()) != required_keys:
            raise ValueError(
                f"specify_variable_chains must contain exactly these keys: {required_keys}"
            )
        
        # Get available chain IDs from structure
        available_chains = {chain.id for chain in self.internal_structure.get_chains()}
        
        # Check if specified chains exist in structure
        for chain_type, chain_id in chains_dict.items():
            if not isinstance(chain_id, str):
                raise ValueError(
                    f"Chain ID for {chain_type} must be a string, got {type(chain_id)}"
                )
            if chain_id not in available_chains:
                raise ValueError(
                    f"Specified {chain_type} chain '{chain_id}' not found in structure. "
                    f"Available chains: {available_chains}"
                )
            
        # Check for duplicate chain IDs
        if chains_dict['heavy'] == chains_dict['light']:
            raise ValueError(
                f"Heavy and light chain IDs must be different, "
                f"got '{chains_dict['heavy']}' for both"
            )

    def _initialize_metadata(self):
        """Initialize antibody metadata."""
        self.chain_indices = self._get_residue_indices_from_struct()
        self.sequences = self._extract_sequences()
        
        # Initialize chain and CDR attributes
        self.antigen_chain = []
        self.heavy_chain: List[str] = []
        self.light_chain: List[str] = []
        self.heavy_chain_start: List[int] = []
        self.light_chain_start: List[int] = []
        self.numbered_heavy_chain = []
        self.numbered_light_chain = []
        
        # Initialize CDR lists
        self.CDRH1: List[List[int]] = []
        self.CDRH2: List[List[int]] = []
        self.CDRH3: List[List[int]] = []
        self.CDRL1: List[List[int]] = []
        self.CDRL2: List[List[int]] = []
        self.CDRL3: List[List[int]] = []
        
        # Initialize CDR sequence lists
        self.CDRH1_seq: List[str] = []
        self.CDRH2_seq: List[str] = []
        self.CDRH3_seq: List[str] = []
        self.CDRL1_seq: List[str] = []
        self.CDRL2_seq: List[str] = []
        self.CDRL3_seq: List[str] = []
        
        self._process_chains()
        self._process_cdrs()
        self._handle_single_domain()

    def _process_chains(self):
        """Process chains and identify heavy, light, and antigen chains."""
        if self.specify_variable_chains:
            self._process_specified_chains()
        else:
            self._process_automatic_chains()

    def _process_automatic_chains(self):
        """Automatically identify chain types."""
        for chain_id, sequence in self.sequences.items():
            logging.debug(f"Processing chain {chain_id}")
            try:
                chains_seq = abnumber.Chain.multiple_domains(sequence, scheme=self.scheme)
                for chain_seq in chains_seq:
                    if chain_seq.is_heavy_chain():
                        self.heavy_chain.append(chain_id)
                        self.heavy_chain_start.append(self.chain_indices[chain_id][0])
                        self.numbered_heavy_chain.append(chain_seq)
                    elif chain_seq.is_light_chain():
                        self.light_chain.append(chain_id)
                        self.light_chain_start.append(self.chain_indices[chain_id][0])
                        self.numbered_light_chain.append(chain_seq)
                    else:
                        self.antigen_chain.append(chain_id)
            except Exception as e:
                logging.debug(f"Chain {chain_id} identified as antigen: {str(e)}")
                self.antigen_chain.append(chain_id)

        # Validate chain identification
        if ((len(self.heavy_chain) > 1 or len(self.light_chain) > 1) 
            and not self.allow_multidomains):
            raise ValueError("Multiple domains found - use allow_multidomains=True")
        
        if not self.heavy_chain or not self.light_chain:
            raise ValueError("Could not identify both heavy and light chains")

    def _process_cdrs(self):
        """Process CDR regions using the working structure's numbering."""
        self._process_heavy_chain_cdrs()
        self._process_light_chain_cdrs()

    def _process_heavy_chain_cdrs(self):
        """Process CDRs for heavy chains."""
        for h_chain_start, numb_h_chain in zip(self.heavy_chain_start, self.numbered_heavy_chain):
            cdr1, cdr2, cdr3 = [], [], []
            cdr1_seq, cdr2_seq, cdr3_seq = [], [], []
            
            chain_id = (self.heavy_chain if isinstance(self.heavy_chain, str) 
                       else self.heavy_chain[0])
            
            for pos, aa in numb_h_chain:
                pos_str = str(pos)
                if pos_str[0] in ['H', 'L']:
                    pos_str = pos_str[1:]
                    
                # Parse position and get the number
                match = re.match(r'(\d+)(\w?)', pos_str)
                if match:
                    num = int(match.group(1))
                else:
                    num = int(pos_str)
                
                region = pos.get_region()
                if region == 'CDR1':
                    cdr1.append(num)
                    cdr1_seq.append(aa)
                elif region == 'CDR2':
                    cdr2.append(num)
                    cdr2_seq.append(aa)
                elif region == 'CDR3':
                    cdr3.append(num)
                    cdr3_seq.append(aa)
                    
            self.CDRH1.append(sorted(cdr1))
            self.CDRH2.append(sorted(cdr2))
            self.CDRH3.append(sorted(cdr3))
            self.CDRH1_seq.append(''.join(cdr1_seq))
            self.CDRH2_seq.append(''.join(cdr2_seq))
            self.CDRH3_seq.append(''.join(cdr3_seq))

    def _process_light_chain_cdrs(self):
        """Process CDRs for light chains."""
        for l_chain_start, numb_l_chain in zip(self.light_chain_start, self.numbered_light_chain):
            cdr1, cdr2, cdr3 = [], [], []
            cdr1_seq, cdr2_seq, cdr3_seq = [], [], []
            
            chain_id = (self.light_chain if isinstance(self.light_chain, str) 
                       else self.light_chain[0])
            
            for pos, aa in numb_l_chain:
                pos_str = str(pos)
                if pos_str[0] in ['H', 'L']:
                    pos_str = pos_str[1:]
                    
                # Parse position and get the number
                match = re.match(r'(\d+)(\w?)', pos_str)
                if match:
                    num = int(match.group(1))
                else:
                    num = int(pos_str)
                
                region = pos.get_region()
                if region == 'CDR1':
                    cdr1.append(num)
                    cdr1_seq.append(aa)
                elif region == 'CDR2':
                    cdr2.append(num)
                    cdr2_seq.append(aa)
                elif region == 'CDR3':
                    cdr3.append(num)
                    cdr3_seq.append(aa)
                    
            self.CDRL1.append(sorted(cdr1))
            self.CDRL2.append(sorted(cdr2))
            self.CDRL3.append(sorted(cdr3))
            self.CDRL1_seq.append(''.join(cdr1_seq))
            self.CDRL2_seq.append(''.join(cdr2_seq))
            self.CDRL3_seq.append(''.join(cdr3_seq))

    def _process_specified_chains(self):
        """Process only the specified heavy and light chains."""
        heavy_chain_id = self.specify_variable_chains['heavy']
        light_chain_id = self.specify_variable_chains['light']
        
        # Process heavy chain
        try:
            heavy_seq = self.sequences[heavy_chain_id]
            chains_seq = abnumber.Chain.multiple_domains(heavy_seq, scheme=self.scheme)
            heavy_valid = False
            for chain_seq in chains_seq:
                if chain_seq.is_heavy_chain():
                    self.heavy_chain.append(heavy_chain_id)
                    self.heavy_chain_start.append(self.chain_indices[heavy_chain_id][0])
                    self.numbered_heavy_chain.append(chain_seq)
                    heavy_valid = True
                    break
            if not heavy_valid:
                raise ValueError(f"Specified heavy chain '{heavy_chain_id}' is not valid")
        except Exception as e:
            raise ValueError(f"Error processing heavy chain '{heavy_chain_id}': {str(e)}")
            
        # Process light chain
        try:
            light_seq = self.sequences[light_chain_id]
            chains_seq = abnumber.Chain.multiple_domains(light_seq, scheme=self.scheme)
            light_valid = False
            for chain_seq in chains_seq:
                if chain_seq.is_light_chain():
                    self.light_chain.append(light_chain_id)
                    self.light_chain_start.append(self.chain_indices[light_chain_id][0])
                    self.numbered_light_chain.append(chain_seq)
                    light_valid = True
                    break
            if not light_valid:
                raise ValueError(f"Specified light chain '{light_chain_id}' is not valid")
        except Exception as e:
            raise ValueError(f"Error processing light chain '{light_chain_id}': {str(e)}")
        
        # Add all other chains as antigen chains
        self.antigen_chain = [
            chain_id for chain_id in self.sequences.keys()
            if chain_id not in [heavy_chain_id, light_chain_id]
        ]

    def _handle_single_domain(self):
        """Convert lists to single values when only one domain is present."""
        if len(self.heavy_chain) == 1 and len(self.light_chain) == 1:
            self.heavy_chain = self.heavy_chain[0]
            self.light_chain = self.light_chain[0]
            self.heavy_chain_start = self.heavy_chain_start[0]
            self.light_chain_start = self.light_chain_start[0]
            self.numbered_heavy_chain = self.numbered_heavy_chain[0]
            self.numbered_light_chain = self.numbered_light_chain[0]
            self.CDRH1 = self.CDRH1[0]
            self.CDRH2 = self.CDRH2[0]
            self.CDRH3 = self.CDRH3[0]
            self.CDRL1 = self.CDRL1[0]
            self.CDRL2 = self.CDRL2[0]
            self.CDRL3 = self.CDRL3[0]
            self.CDRH1_seq = self.CDRH1_seq[0]
            self.CDRH2_seq = self.CDRH2_seq[0]
            self.CDRH3_seq = self.CDRH3_seq[0]
            self.CDRL1_seq = self.CDRL1_seq[0]
            self.CDRL2_seq = self.CDRL2_seq[0]
            self.CDRL3_seq = self.CDRL3_seq[0]

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get a dictionary containing all metadata about the antibody structure.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing chain information and CDR definitions
        """
        return {
            'scheme': self.scheme,
            'heavy_chain': self.heavy_chain,
            'light_chain': self.light_chain,
            'antigen_chain': self.antigen_chain,
            'heavy_chain_start': self.heavy_chain_start,
            'light_chain_start': self.light_chain_start,
            'CDRH1': self.CDRH1,
            'CDRH2': self.CDRH2,
            'CDRH3': self.CDRH3,
            'CDRL1': self.CDRL1,
            'CDRL2': self.CDRL2,
            'CDRL3': self.CDRL3,
            'CDRH1_seq': self.CDRH1_seq,
            'CDRH2_seq': self.CDRH2_seq,
            'CDRH3_seq': self.CDRH3_seq,
            'CDRL1_seq': self.CDRL1_seq,
            'CDRL2_seq': self.CDRL2_seq,
            'CDRL3_seq': self.CDRL3_seq
        }

    def select_cdr(self, cdr: str, chain_type: str) -> Structure:
        """
        Select a specific CDR region from the structure.
        
        Parameters:
        -----------
        cdr : str
            The CDR to select ('1', '2', or '3')
        chain_type : str
            Which chain to select from ('heavy' or 'light')
                
        Returns:
        --------
        Bio.PDB.Structure.Structure
            A new structure containing only the selected CDR region
        """
        if cdr not in ['1', '2', '3']:
            raise ValueError("CDR must be '1', '2', or '3'")
        
        if chain_type not in ['heavy', 'light']:
            raise ValueError("chain_type must be 'heavy' or 'light'")

        # Get CDR residues and chain ID
        if chain_type == 'heavy':
            cdr_residues = getattr(self, f'CDRH{cdr}')
            chain_id = (self.heavy_chain if isinstance(self.heavy_chain, str) 
                       else self.heavy_chain[0])
        else:
            cdr_residues = getattr(self, f'CDRL{cdr}')
            chain_id = (self.light_chain if isinstance(self.light_chain, str) 
                       else self.light_chain[0])

        # Convert residues list to selection string
        if isinstance(cdr_residues[0], list):
            residue_ranges = ['+'.join(map(str, domain_residues)) 
                            for domain_residues in cdr_residues]
            range_string = '+'.join(residue_ranges)
        else:
            range_string = '+'.join(map(str, cdr_residues))

        # Create selection string
        selection_string = f"chain {chain_id} and resi {range_string}"
        logging.debug(f"Selection string: {selection_string}")
        
        return self.select(selection_string)

    def get_cdr_coordinates(self, cdr: str, chain_type: str) -> np.ndarray:
        """
        Get the coordinates of a specific CDR region.
        
        Parameters:
        -----------
        cdr : str
            The CDR to select ('1', '2', or '3')
        chain_type : str
            Which chain to select from ('heavy' or 'light')
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_atoms, 3) containing the coordinates of the CDR atoms
        """
        cdr_structure = self.select_cdr(cdr, chain_type)
        return np.array([atom.coord for atom in cdr_structure.get_atoms()])