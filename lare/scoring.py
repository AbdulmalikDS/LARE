from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RetrievalScore:
    score: float
    global_sim: float
    best_region_sim: float


class ConfidenceGatedScoring:
    """
    Paper Eq. 4:
      S = s_g                                    if s_g >= tau
      S = (1 - alpha)*s_g + alpha*s_r            if s_r > s_g, alpha = min(2*(s_r-s_g), 0.5)
      S = s_g                                    otherwise
    """

    def __init__(self, tau: float = 0.25):
        self.tau = tau

    def score(
        self,
        text_emb: np.ndarray,
        global_emb: np.ndarray,
        region_embs: List[np.ndarray],
    ) -> RetrievalScore:
        s_g = float(np.dot(text_emb, global_emb))
        s_r = max((float(np.dot(text_emb, r)) for r in region_embs), default=s_g)
        return RetrievalScore(
            score=self._gate(s_g, s_r),
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
            s_g = sim[i]

            use_region = (s_g < self.tau) & (s_r > s_g)
            alpha = np.minimum(2.0 * np.where(use_region, s_r - s_g, 0.0), 0.5)
            sim[i] = np.where(use_region, (1.0 - alpha) * s_g + alpha * s_r, s_g)

        return sim.T  # [N_text, N_image]

    def _gate(self, s_g: float, s_r: float) -> float:
        if s_g >= self.tau:
            return s_g
        if s_r > s_g:
            alpha = min(2.0 * (s_r - s_g), 0.5)
            return (1.0 - alpha) * s_g + alpha * s_r
        return s_g
