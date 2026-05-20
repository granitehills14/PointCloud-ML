from scipy.spatial import cKDTree
import numpy as np

def majority_vote(neighbor_labels):
    return np.array([
        np.bincount(row).argmax()
        for row in neighbor_labels
    ])

def extrapolate_classification(pc_glcs_classified, neighborhood):
    k = neighborhood["k"]
    unclassified_labels = neighborhood["unclassified_labels"]
    
    labels = pc_glcs_classified[:,3]

    target_mask = np.isin(labels, neighborhood["unclassified_labels"])
    candidate_mask = ~target_mask
    
    candidate_xyz = pc_glcs_classified[candidate_mask, :3]
    target_xyz = pc_glcs_classified[target_mask, :3]
    candidate_labels = labels[candidate_mask]

    tree = cKDTree(candidate_xyz)
    dist, idx = tree.query(target_xyz, k=k)

    neighbor_labels = candidate_labels[idx]
    filled_labels = majority_vote(neighbor_labels)

    pc_glcs_filled = pc_glcs_classified.copy()
    pc_glcs_filled[target_mask,3] = filled_labels

    return pc_glcs_filled