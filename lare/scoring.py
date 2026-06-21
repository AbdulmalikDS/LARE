from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RetrievalScore:
    score: float
    global_sim: float
    best_region_sim: float


def csls_rerank(sim: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Cross-domain Similarity Local Scaling (Conneau et al., 2018).

    Corrects hubness bias by penalising embeddings that are uniformly
    close to many neighbours in the other modality:

        CSLS(q, i) = 2·cos(q, i) - mean_kNN_img(q) - mean_kNN_txt(i)

    Args:
        sim: [N_query, N_gallery] similarity matrix.
        k: Number of neighbours for the mean kNN estimate.

    Returns:
        Reranked similarity matrix of the same shape.
    """
    topk_per_query = np.sort(sim, axis=1)[:, -k:]
    r_T = topk_per_query.mean(axis=1, keepdims=True)
    topk_per_gallery = np.sort(sim, axis=0)[-k:, :]
    r_S = topk_per_gallery.mean(axis=0, keepdims=True)
    return 2 * sim - r_T - r_S


class ConfidenceGatedScoring:
    """
    Paper Eq. 4:
      S = s_g                                    if s_g >= tau
      S = (1 - alpha)*s_g + alpha*s_r            if s_r > s_g, alpha = min(2*(s_r-s_g), 0.5)
      S = s_g                                    otherwise

    Optionally applies CSLS hub-correction after gated fusion.
    """

    def __init__(self, tau: float = 0.25, csls_k: int = 10):
        self.tau = tau
        self.csls_k = csls_k

    def score(
        self,
        text_emb: np.ndarray,
        global_emb: np.ndarray,
        region_embs: List[np.ndarray],
    ) -> RetrievalScore:
        s_g = float(np.dot(text_emb, global_emb))
        s_r = max((float(np.dot(text_emb, r)) for r in region_embs), default=s_g)
        return RetrievalScore(
            score=float(self._fuse(np.float64(s_g), np.float64(s_r))),
            global_sim=s_g,
            best_region_sim=s_r,
        )

    def score_matrix(
        self,
        text_embs: np.ndarray,
        global_embs: np.ndarray,
        region_embs_list: List[List[np.ndarray]],
    ) -> np.ndarray:
        """Returns [N_text, N_image] similarity matrix."""
        sim = global_embs @ text_embs.T  # [N_image, N_text]

        for i, regions in enumerate(region_embs_list):
            if not regions:
                continue
            s_r = np.array(regions) @ text_embs.T  # [n_r, N_text]
            s_r = s_r.max(axis=0)                   # [N_text]
            sim[i] = self._fuse(sim[i], s_r)

        sim_t2i = sim.T  # [N_text, N_image]
        if self.csls_k > 0:
            sim_t2i = csls_rerank(sim_t2i, k=self.csls_k)
        return sim_t2i

    def _fuse(self, s_g: np.ndarray, s_r: np.ndarray) -> np.ndarray:
        """Paper Eq. 4, vectorized over scalars or arrays alike."""
        use_region = (s_g < self.tau) & (s_r > s_g)
        alpha = np.minimum(2.0 * np.where(use_region, s_r - s_g, 0.0), 0.5)
        return np.where(use_region, (1.0 - alpha) * s_g + alpha * s_r, s_g)
