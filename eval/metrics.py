from typing import Dict, Tuple
import numpy as np


def recall_at_k(
    sim_matrix: np.ndarray,
    text_to_image: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Text-to-image Recall@K.

    Args:
        sim_matrix: [N_text, N_image] similarity scores.
        text_to_image: [N_text] ground-truth image index per query.
        ks: K values to evaluate.

    Returns:
        {"R@1": 54.2, "R@5": 78.0, ...}
    """
    ranked = np.argsort(-sim_matrix, axis=1)
    return {
        f"R@{k}": float((ranked[:, :k] == text_to_image[:, None]).any(axis=1).mean() * 100)
        for k in ks
    }
