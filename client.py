# -*- coding: utf-8 -*-
"""
GenPark Physical AI Action Chunking with Transformers (ACT) Policy Interpolator
Standard Library only: math, collections, typing.
Temporal ensembling with exponential recency weighting and sliding window smoothing
for jerky multi-chunk robot joint angle execution.
"""

import math
from typing import Dict, List, Tuple, Any

class GenParkEmbodiedActionChunkingACTPolicyInterpolator:
    def __init__(self, chunk_size: int = 16, decay_rate: float = 0.15):
        self.chunk_size = chunk_size
        self.decay_rate = decay_rate
        # Map: step_idx -> list of predicted action vectors for this step
        # Each item: (predicted_val_list, weight)
        self.predictions_at_step: Dict[int, List[Tuple[List[float], float]]] = {}

    def ingest_action_chunk(self, emitted_step: int, trajectory_chunk: List[List[float]]) -> Dict[str, Any]:
        """
        Ingest a predicted chunk of future actions emitted at emitted_step.
        Each action is a list of DoF floats.
        """
        for offset, action in enumerate(trajectory_chunk):
            target_step = emitted_step + offset
            # Exponential decay weight: w = exp(-decay_rate * offset)
            weight = math.exp(-self.decay_rate * offset)
            if target_step not in self.predictions_at_step:
                self.predictions_at_step[target_step] = []
            self.predictions_at_step[target_step].append((action, weight))

        return {
            "emitted_step": emitted_step,
            "chunk_length": len(trajectory_chunk),
            "horizon_span": (emitted_step, emitted_step + len(trajectory_chunk) - 1),
            "total_cached_steps": len(self.predictions_at_step)
        }

    def get_ensembled_action(self, step: int) -> Dict[str, Any]:
        """
        Computes normalized weighted average of all overlapping predictions for the given step.
        """
        preds = self.predictions_at_step.get(step, [])
        if not preds:
            return {"step": step, "has_action": False, "action": []}

        num_dofs = len(preds[0][0])
        total_weight = sum(w for _, w in preds)

        ensembled_action = [0.0] * num_dofs
        for action_vec, w in preds:
            for d in range(num_dofs):
                ensembled_action[d] += action_vec[d] * w

        if total_weight > 0:
            ensembled_action = [round(val / total_weight, 5) for val in ensembled_action]

        return {
            "step": step,
            "has_action": True,
            "action": ensembled_action,
            "ensemble_sources_count": len(preds),
            "total_weight": round(total_weight, 4)
        }

    def evaluate_smoothness(self, executed_actions: List[List[float]]) -> Dict[str, Any]:
        """
        Measures smoothness via discrete acceleration variance / total variation.
        """
        if len(executed_actions) < 3:
            return {"smoothness_score": 1.0, "total_variation": 0.0}

        num_dofs = len(executed_actions[0])
        total_var = 0.0
        acc_var = 0.0

        for t in range(1, len(executed_actions)):
            for d in range(num_dofs):
                total_var += abs(executed_actions[t][d] - executed_actions[t-1][d])

        for t in range(1, len(executed_actions) - 1):
            for d in range(num_dofs):
                acc = (executed_actions[t+1][d] - 2*executed_actions[t][d] + executed_actions[t-1][d])
                acc_var += acc * acc

        score = round(1.0 / (1.0 + acc_var), 4)
        return {
            "smoothness_score": score,
            "total_variation": round(total_var, 4),
            "acceleration_variance": round(acc_var, 4),
            "step_count": len(executed_actions)
        }
