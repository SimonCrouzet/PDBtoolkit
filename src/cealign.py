import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd

class CEAligner:
    def __init__(self, window_size=8, max_gap=30, d0=3.0, d1=4.0, use_guide_atoms=True):
        self.window_size = window_size
        self.max_gap = max_gap
        self.d0 = d0
        self.d1 = d1
        self.max_kept = 20
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
        sim_matrix = self.calc_similarity_matrix(dm1, dm2)
        paths = self.find_path(sim_matrix, dm1, dm2)
        result = self.find_best(paths, self.reference_coords, mobile_coords, mobile_structure)

        if transform:
            for atom, new_coord in zip(mobile_atoms, result['aligned_coords']):
                atom.coord = new_coord

        return result

    def calc_distance_matrix(self, coords):
        return squareform(pdist(coords))

    def calc_similarity_matrix(self, dm1, dm2):
        sim_matrix = np.full((self.len1, self.len2), -1.0)
        win_sum = (self.window_size - 1) * (self.window_size - 2) / 2

        for i in range(self.len1 - self.window_size + 1):
            for j in range(self.len2 - self.window_size + 1):
                score = np.sum(np.abs(dm1[i:i+self.window_size-1, i+1:i+self.window_size] - 
                                      dm2[j:j+self.window_size-1, j+1:j+self.window_size]))
                sim_matrix[i, j] = score / win_sum

        return sim_matrix

    def find_path(self, sim_matrix, dm1, dm2):
        smaller = min(self.len1, self.len2)
        win_cache = np.array([(i+1)*i*self.window_size/2 + (i+1)*((self.window_size-1)*(self.window_size-2)/2) for i in range(smaller)])
        
        path_buffer = []
        score_buffer = []
        len_buffer = []

        for ia in range(self.len1):
            if ia > self.len1 - self.window_size * (max(len_buffer) - 1 if len_buffer else 0):
                break

            for ib in range(self.len2):
                if sim_matrix[ia, ib] < self.d0 * 1.2:  # Allow 20% more flexibility
                    cur_path = [(ia, ib)]
                    cur_path_length = 1
                    cur_total_score = 0.0

                    while True:
                        ja_range = cur_path[-1][0] + self.window_size + np.arange(self.max_gap + 1)
                        jb_range = cur_path[-1][1] + self.window_size + np.arange(self.max_gap + 1)

                        valid_ja = ja_range[ja_range < self.len1 - self.window_size]
                        valid_jb = jb_range[jb_range < self.len2 - self.window_size]

                        if len(valid_ja) == 0 or len(valid_jb) == 0:
                            break

                        ja_grid, jb_grid = np.meshgrid(valid_ja, valid_jb)
                        scores = sim_matrix[ja_grid, jb_grid]

                        valid_scores = (scores <= self.d0 * 1.2) & (scores != -1.0)
                        if not np.any(valid_scores):
                            break

                        best_indices = np.unravel_index(np.argmin(scores[valid_scores]), scores[valid_scores].shape)
                        ja, jb = ja_grid[valid_scores][best_indices], jb_grid[valid_scores][best_indices]

                        cur_path.append((ja, jb))
                        cur_path_length += 1

                        score1 = (scores[valid_scores][best_indices] * self.window_size * (cur_path_length - 1) + 
                                  sim_matrix[ja, jb] * ((self.window_size-1)*(self.window_size-2)/2)) / \
                                 (self.window_size * (cur_path_length - 1) + ((self.window_size-1)*(self.window_size-2)/2))

                        score2 = (sim_matrix[ia, ib] if cur_path_length == 2 else cur_total_score) * win_cache[cur_path_length - 2] + \
                                 score1 * (win_cache[cur_path_length - 1] - win_cache[cur_path_length - 2])
                        score2 /= win_cache[cur_path_length - 1]

                        cur_total_score = score2

                        if cur_total_score > self.d1 * 1.2:  # Allow 20% more flexibility
                            break

                    if cur_path_length > min(len_buffer) if len_buffer else 0 or \
                       (cur_path_length == min(len_buffer) if len_buffer else 0 and cur_total_score < max(score_buffer) if score_buffer else float('inf')):
                        if len(path_buffer) < self.max_kept:
                            path_buffer.append(cur_path)
                            score_buffer.append(cur_total_score)
                            len_buffer.append(cur_path_length)
                        else:
                            max_index = score_buffer.index(max(score_buffer))
                            path_buffer[max_index] = cur_path
                            score_buffer[max_index] = cur_total_score
                            len_buffer[max_index] = cur_path_length

        return path_buffer

    def reconstruct_alignment(self, path):
        alignment = []
        for p in path:
            for i in range(self.window_size):
                alignment.append((self.reference_atoms[p[0]+i].get_parent().id[1],
                                  self.reference_atoms[p[1]+i].get_parent().id[1]))
        return alignment

    def find_best(self, paths, coords1, coords2, mobile_structure):
        best_paths = sorted(paths, key=lambda p: self.path_score(p, coords1, coords2))[:self.max_kept]
        
        best_result = None
        best_rmsd = float('inf')

        for path in best_paths:
            alignment = self.reconstruct_alignment(path)
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
        coords1_aligned = []
        coords2_aligned = []
        for align1, align2 in alignment:
            if align1 != '-' and align2 != '-':
                coords1_aligned.append(coords1[align1])
                coords2_aligned.append(coords2[align2])
        return np.array(coords1_aligned), np.array(coords2_aligned)

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
        aligned_coords1 = np.array([coords1[i] for i, j in alignment if i != '-' and j != '-'])
        aligned_coords2 = np.array([coords2[j] for i, j in alignment if i != '-' and j != '-'])
        
        centroid1 = np.mean(aligned_coords1, axis=0)
        centroid2 = np.mean(aligned_coords2, axis=0)
        
        rotation = self.calculate_rotation(coords1, coords2, alignment)
        return centroid1 - rotation @ centroid2

    def apply_transformation(self, coords, rotation, translation):
        return (rotation @ coords.T).T + translation

    def path_score(self, path, coords1, coords2):
        c1 = np.array([coords1[p[0]:p[0]+self.window_size] for p in path]).reshape(-1, 3)
        c2 = np.array([coords2[p[1]:p[1]+self.window_size] for p in path]).reshape(-1, 3)
        return self.calculate_rmsd(c1, c2, list(zip(range(len(c1)), range(len(c2)))))