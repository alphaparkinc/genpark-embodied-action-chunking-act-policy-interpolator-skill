# -*- coding: utf-8 -*-
from client import GenParkEmbodiedActionChunkingACTPolicyInterpolator

def main():
    interpolator = GenParkEmbodiedActionChunkingACTPolicyInterpolator(chunk_size=4, decay_rate=0.2)
    print("=== GenPark Physical AI ACT Policy Temporal Ensembling ===")

    # Emit chunk at t=0
    c0 = [[0.0, 0.0], [0.1, 0.05], [0.2, 0.1], [0.3, 0.15]]
    interpolator.ingest_action_chunk(emitted_step=0, trajectory_chunk=c0)

    # Emit chunk at t=1 (overlapping prediction with slight variance)
    c1 = [[0.11, 0.048], [0.21, 0.098], [0.31, 0.148], [0.4, 0.2]]
    interpolator.ingest_action_chunk(emitted_step=1, trajectory_chunk=c1)

    print("Ensembled Actions:")
    history = []
    for t in range(5):
        action = interpolator.get_ensembled_action(t)
        if action["has_action"]:
            history.append(action["action"])
            print(f"  Step {t}: action={action['action']}, sources={action['ensemble_sources_count']}")

    smooth = interpolator.evaluate_smoothness(history)
    print(f"Trajectory Smoothness Score: {smooth['smoothness_score']} (Variance: {smooth['acceleration_variance']})")

if __name__ == "__main__":
    main()
