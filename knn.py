from scipy.spatial import cKDTree
import numpy as np
import open3d as o3d

def majority_vote(neighbor_labels):
    return np.array([
        np.bincount(row).argmax()
        for row in neighbor_labels
    ])


def extrapolate_classification(pc_glcs_classified, neighborhood):
    input_k = neighborhood["k"]
    unclassified_labels = neighborhood["unclassified_labels"]
    batch_size = neighborhood["batch_size"]
    
    labels = pc_glcs_classified.point.classification.cpu().numpy().reshape(-1)
    labels = labels.astype(np.int32, copy=False)

    unique, counts = np.unique(labels, return_counts=True)
    print("Pre-kNN label counts:")
    for u, c in zip(unique, counts):
        print(f"  label {u}: {c:,}")

    target_mask = np.isin(labels, unclassified_labels)
    candidate_mask = ~target_mask

    print(f"kNN target labels: {unclassified_labels}")
    print(f"kNN target points: {np.count_nonzero(target_mask):,}")
    print(f"kNN candidate points: {np.count_nonzero(candidate_mask):,}")

    if not np.any(target_mask):
        # return pc_glcs_classified.clone()
        if not np.any(target_mask):
            print("No target points matched unclassified_labels. kNN fill skipped.")
        return pc_glcs_classified.clone()

    if not np.any(candidate_mask):
        raise SystemExit("No classified candidate points available for kNN extrapolation. Exiting.")

    num_candidates = np.count_nonzero(candidate_mask)

    if num_candidates < 3:
        raise ValueError("Need at least 3 classified candidate points for kNN extrapolation.")

    k = int(input_k)

    if k < 3:
        k = 3

    if k % 2 == 0:
        k += 1

    if k > num_candidates:
        k = num_candidates

    if k % 2 == 0:
        k -= 1
    
    print(f"Extrapolating Classification with neighborhood: {k} points.")

    xyz = pc_glcs_classified.point.positions.cpu().numpy()
    candidate_xyz = xyz[candidate_mask]
    candidate_labels = labels[candidate_mask]

    tree = cKDTree(candidate_xyz)
    target_indices = np.flatnonzero(target_mask)

    pc_glcs_filled = pc_glcs_classified.clone()
    new_labels = labels.copy()

    num_batches = int(np.ceil(len(target_indices) / batch_size))

    for batch_start in range(0, len(target_indices), batch_size):
        batch_num = batch_start // batch_size + 1
        print(f"kNN batch {batch_num:,} / {num_batches:,}")
        batch_indices = target_indices[batch_start:batch_start + batch_size]
        batch_xyz = xyz[batch_indices]

        _, idx = tree.query(batch_xyz, k=k)

        neighbor_labels = candidate_labels[idx]
        filled_labels = majority_vote(neighbor_labels)

        new_labels[batch_indices] = filled_labels


    pc_glcs_filled.point.classification = o3d.core.Tensor(
        new_labels.reshape(-1, 1),
        dtype=pc_glcs_classified.point.classification.dtype,
        device=pc_glcs_classified.point.classification.device
    )

    n_changed = np.count_nonzero(new_labels != labels)
    print(f"kNN changed {n_changed:,} point labels.")

    return pc_glcs_filled