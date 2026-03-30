from typing import Dict, Tuple
import numpy as np


def recall_at_k(
    sim_matrix: np.ndarray,
    text_to_image: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute text-to-image Recall@K.

    Args:
        sim_matrix: [N_text, N_image] similarity scores (higher = more similar).
        text_to_image: [N_text] ground-truth image index for each query.
        ks: Tuple of K values to evaluate.

    Returns:
        Dict mapping "R@K" → percentage (e.g., {"R@1": 54.2, "R@5": 78.0, "R@10": 84.2}).
    """
    # Rank images for each query (descending similarity)
    ranked = np.argsort(-sim_matrix, axis=1)  # [N_text, N_image]
    results = {}
    for k in ks:
        # Check whether the ground-truth image appears in the top-k
        hits = (ranked[:, :k] == text_to_image[:, None]).any(axis=1)
        results[f"R@{k}"] = float(hits.mean() * 100)
    return results
