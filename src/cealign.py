import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd
from numba import jit, prange

class CEAligner:
    def __init__(self, window_size=8, max_gap=30, d0=3.0, d1=4.0, use_guide_atoms=True):
        self.window_size = window_size
        self.max_gap = max_gap
        self.d0 = d0
        self.d1 = d1
        self.max_kept = 10
        self.reference_coords = None
        self.reference_atoms = None
        self.use_guide_atoms = use_guide_atoms
        self.gap_open = 5  # Updated based on the paper
        self.gap_extend = 0.5  # Updated based on the paper
        self.optimization_threshold = 3.5  # Z-score threshold for optimization

    def set_reference(self, structure):
        if self.use_guide_atoms:
            self.reference_coords = np.array([atom.coord for atom in structure.get_atoms() if atom.name in ['CA', "C4'"]])
            self.reference_atoms = [atom for atom in structure.get_atoms() if atom.name in ['CA', "C4'"]]
        else:
            self.reference_coords = np.array([atom.coord for atom in structure.get_atoms()])
            self.reference_atoms = list(structure.get_atoms())

    def align(self, mobile_structure, transform=False):
        if self.reference_coords is None:
            raise ValueError("Reference structure not set. Call set_reference() first.")

        if self.use_guide_atoms:
            mobile_coords = np.array([atom.coord for atom in mobile_structure.get_atoms() if atom.name in ['CA', "C4'"]])
            mobile_atoms = [atom for atom in mobile_structure.get_atoms() if atom.name in ['CA', "C4'"]]
        else:
            mobile_coords = np.array([atom.coord for atom in mobile_structure.get_atoms()])
            mobile_atoms = list(mobile_structure.get_atoms())

        self.len1 = len(self.reference_coords)
        self.len2 = len(mobile_coords)

        dm1 = self.calc_distance_matrix(self.reference_coords)
        dm2 = self.calc_distance_matrix(mobile_coords)
        sim_matrix = self.calc_similarity_matrix(dm1, dm2, self.window_size, self.d0)
        paths = self.find_path(sim_matrix, dm1, dm2, self.window_size, self.max_gap, self.d0, self.d1)
        path_scores = [self.path_score(path, self.reference_coords, mobile_coords, self.window_size) for path in paths]
        best_paths = [path for _, path in sorted(zip(path_scores, paths))[:self.max_kept]]
        result = self.find_best(best_paths, self.reference_coords, mobile_coords, mobile_atoms)

        if transform:
            for atom, new_coord in zip(mobile_atoms, result['aligned_coords']):
                atom.coord = new_coord

        return result

    @staticmethod
    @jit(nopython=True, parallel=True)
    def calc_distance_matrix(coords):
        n = coords.shape[0]
        dist_matrix = np.zeros((n, n))
        for i in prange(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix

    @staticmethod
    @jit(nopython=True)
    def calc_similarity_matrix(dm1, dm2, window_size, d0):
        len1, len2 = dm1.shape[0], dm2.shape[0]
        sim_matrix = np.full((len1, len2), -1.0)
        win_sum = (window_size - 1) * (window_size - 2) / 2

        for i in range(len1 - window_size + 1):
            for j in range(len2 - window_size + 1):
                score = np.sum(np.abs(dm1[i:i+window_size-1, i+1:i+window_size] - 
                                      dm2[j:j+window_size-1, j+1:j+window_size]))
                sim_matrix[i, j] = score / win_sum

        return sim_matrix

    @staticmethod
    @jit(nopython=True)
    def find_path(sim_matrix, dm1, dm2, window_size, max_gap, d0, d1):
        len1, len2 = sim_matrix.shape
        smaller = min(len1, len2)
        win_cache = np.array([(i+1)*i*window_size/2 + (i+1)*((window_size-1)*(window_size-2)/2) for i in range(smaller)])
        
        path_buffer = []
        score_buffer = []
        len_buffer = []
        max_kept = 20

        for ia in range(len1):
            if ia > len1 - window_size * (max(len_buffer) if len_buffer else 0):
                break

            for ib in range(len2):
                if sim_matrix[ia, ib] < d0 * 1.2:
                    cur_path = [(ia, ib)]
                    cur_path_length = 1
                    cur_total_score = 0.0

                    while True:
                        ja_start = cur_path[-1][0] + window_size
                        jb_start = cur_path[-1][1] + window_size
                        ja_end = min(len1, ja_start + max_gap + 1)
                        jb_end = min(len2, jb_start + max_gap + 1)

                        if ja_start >= ja_end or jb_start >= jb_end:
                            break

                        best_score = 1e10  # Use a large number instead of infinity
                        best_ja = -1
                        best_jb = -1

                        for ja in range(ja_start, ja_end):
                            for jb in range(jb_start, jb_end):
                                score = sim_matrix[ja, jb]
                                if score <= d0 * 1.2 and score != -1.0 and score < best_score:
                                    best_score = score
                                    best_ja = ja
                                    best_jb = jb

                        if best_ja == -1 or best_jb == -1:
                            break

                        cur_path.append((best_ja, best_jb))
                        cur_path_length += 1

                        score1 = (best_score * window_size * (cur_path_length - 1) + 
                                  sim_matrix[best_ja, best_jb] * ((window_size-1)*(window_size-2)/2)) / \
                                 (window_size * (cur_path_length - 1) + ((window_size-1)*(window_size-2)/2))

                        score2 = (sim_matrix[ia, ib] if cur_path_length == 2 else cur_total_score) * win_cache[cur_path_length - 2] + \
                                 score1 * (win_cache[cur_path_length - 1] - win_cache[cur_path_length - 2])
                        score2 /= win_cache[cur_path_length - 1]

                        cur_total_score = score2

                        if cur_total_score > d1 * 1.2:
                            break

                    if cur_path_length > min(len_buffer) if len_buffer else 0 or \
                       (cur_path_length == min(len_buffer) if len_buffer else 0 and cur_total_score < max(score_buffer) if score_buffer else 1e10):
                        if len(path_buffer) < max_kept:
                            path_buffer.append(cur_path)
                            score_buffer.append(cur_total_score)
                            len_buffer.append(cur_path_length)
                        else:
                            # Find the index of the maximum score manually
                            max_index = 0
                            max_score = score_buffer[0]
                            for i in range(1, len(score_buffer)):
                                if score_buffer[i] > max_score:
                                    max_index = i
                                    max_score = score_buffer[i]
                            
                            if cur_total_score < max_score:
                                path_buffer[max_index] = cur_path
                                score_buffer[max_index] = cur_total_score
                                len_buffer[max_index] = cur_path_length

        return path_buffer

    def reconstruct_alignment(self, path, mobile_atoms):
        alignment = []
        for p in path:
            for i in range(self.window_size):
                alignment.append((self.reference_atoms[p[0]+i].get_parent().id[1],
                                  mobile_atoms[p[1]+i].get_parent().id[1]))
        return alignment

    def find_best(self, best_paths, coords1, coords2, mobile_atoms):
        best_result = None
        best_rmsd = float('inf')

        for path in best_paths:
            alignment = self.reconstruct_alignment(path, mobile_atoms)
            optimized_alignment = self.optimize_final_path(alignment, coords1, coords2)
            rmsd = self.calculate_rmsd(coords1, coords2, optimized_alignment)

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_result = {
                    'rmsd': rmsd,
                    'alignment_length': len(optimized_alignment),
                    'alignment': optimized_alignment,
                    'rotation': self.calculate_rotation(coords1, coords2, optimized_alignment),
                    'translation': self.calculate_translation(coords1, coords2, optimized_alignment),
                }

        return best_result

    def optimize_final_path(self, alignment, coords1, coords2):
        # Step 1: Evaluate and select the best path (already done in find_best)

        # Step 2: Gap relocation
        alignment = self.relocate_gaps(alignment, coords1, coords2)

        # Step 3: Iterative optimization using dynamic programming
        d0 = 2.0
        prev_alignment_length = len(alignment)
        prev_rmsd = self.calculate_rmsd(coords1, coords2, alignment)

        while True:
            dist_matrix = self.calc_distance_matrix_for_dp(coords1, coords2, alignment)
            new_alignment = self.dynamic_programming_align(dist_matrix, d0)
            new_rmsd = self.calculate_rmsd(coords1, coords2, new_alignment)

            if len(new_alignment) < 0.95 * prev_alignment_length or new_rmsd > 1.1 * prev_rmsd:
                break

            alignment = new_alignment
            prev_alignment_length = len(alignment)
            prev_rmsd = new_rmsd
            d0 += 0.5

        return alignment

    def relocate_gaps(self, alignment, coords1, coords2):
        new_alignment = []
        for i, (a, b) in enumerate(alignment):
            if a == '-' or b == '-':
                best_pos = i
                best_rmsd = float('inf')
                for j in range(max(0, i - self.window_size // 2), min(len(alignment), i + self.window_size // 2 + 1)):
                    temp_alignment = alignment[:j] + [(a, b)] + alignment[j:]
                    rmsd = self.calculate_rmsd(coords1, coords2, temp_alignment)
                    if rmsd < best_rmsd:
                        best_pos = j
                        best_rmsd = rmsd
                new_alignment.insert(best_pos, (a, b))
            else:
                new_alignment.append((a, b))
        return new_alignment

    def calc_distance_matrix_for_dp(self, coords1, coords2, alignment):
        rotation, translation = self.calculate_rotation(coords1, coords2, alignment), self.calculate_translation(coords1, coords2, alignment)
        transformed_coords2 = self.apply_transformation(coords2, rotation, translation)
        return np.sqrt(np.sum((coords1[:, np.newaxis] - transformed_coords2[np.newaxis, :])**2, axis=2))

    def dynamic_programming_align(self, dist_matrix, d0):
        m, n = dist_matrix.shape
        score_matrix = np.zeros((m+1, n+1))
        traceback = np.zeros((m+1, n+1), dtype=int)
        
        score_matrix[1:, 0] = self.gap_open + np.arange(m) * self.gap_extend
        score_matrix[0, 1:] = self.gap_open + np.arange(n) * self.gap_extend
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                match = score_matrix[i-1, j-1] + (d0 - dist_matrix[i-1, j-1])
                delete = score_matrix[i-1, j] - (self.gap_extend if traceback[i-1, j] == 1 else self.gap_open)
                insert = score_matrix[i, j-1] - (self.gap_extend if traceback[i, j-1] == 2 else self.gap_open)
                
                score_matrix[i, j] = max(match, delete, insert)
                traceback[i, j] = np.argmax([match, delete, insert])
        
        alignment = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and traceback[i, j] == 0:
                alignment.append((i-1, j-1))
                i -= 1
                j -= 1
            elif i > 0 and traceback[i, j] == 1:
                alignment.append((i-1, '-'))
                i -= 1
            elif j > 0:
                alignment.append(('-', j-1))
                j -= 1
            else:
                break
        
        return list(reversed(alignment))

    def get_aligned_coords(self, coords1, coords2, alignment):
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

    def calculate_rmsd(self, coords1, coords2, alignment):
        aligned_coords1, aligned_coords2 = self.get_aligned_coords(coords1, coords2, alignment)
        if len(aligned_coords1) == 0:
            return float('inf')  # Return infinity for completely misaligned structures
        return np.sqrt(np.mean(np.sum((aligned_coords1 - aligned_coords2)**2, axis=1)))

    def calculate_rotation(self, coords1, coords2, alignment):
        aligned_coords1, aligned_coords2 = self.get_aligned_coords(coords1, coords2, alignment)
        
        centroid1 = np.mean(aligned_coords1, axis=0)
        centroid2 = np.mean(aligned_coords2, axis=0)
        
        H = (aligned_coords2 - centroid2).T @ (aligned_coords1 - centroid1)
        U, S, Vt = np.linalg.svd(H)
        return Vt.T @ U.T

    def calculate_translation(self, coords1, coords2, alignment):
        aligned_coords1, aligned_coords2 = self.get_aligned_coords(coords1, coords2, alignment)
        
        centroid1 = np.mean(aligned_coords1, axis=0)
        centroid2 = np.mean(aligned_coords2, axis=0)
        
        rotation = self.calculate_rotation(coords1, coords2, alignment)
        return centroid1 - rotation @ centroid2

    def apply_transformation(self, coords, rotation, translation):
        return (rotation @ coords.T).T + translation

    @staticmethod
    @jit(nopython=True)
    def path_score(path, coords1, coords2, window_size):
        total_distance = 0.0
        for p in path:
            c1 = coords1[p[0]:p[0]+window_size]
            c2 = coords2[p[1]:p[1]+window_size]
            total_distance += np.sum((c1 - c2)**2)
        return np.sqrt(total_distance / (len(path) * window_size))