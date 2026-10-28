# -*- coding: utf-8 -*-
import sys, json
from client import GenParkEmbodiedActionChunkingACTPolicyInterpolator

interpolator = GenParkEmbodiedActionChunkingACTPolicyInterpolator()

def run_tool(name, args):
    if name == "ingest_action_chunk":
        step = int(args.get("emitted_step", 0))
        chunk = args.get("trajectory_chunk", [])
        return interpolator.ingest_action_chunk(step, chunk)
    elif name == "get_ensembled_action":
        step = int(args.get("step", 0))
        return interpolator.get_ensembled_action(step)
    elif name == "evaluate_smoothness":
        actions = args.get("executed_actions", [])
        return interpolator.evaluate_smoothness(actions)
    else:
        raise ValueError(f"Unknown tool: {name}")

def main():
    if "--test" in sys.argv:
        print("[TEST] Running GenPark ACT Policy Interpolator Self-Test...")
        chunk1 = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
        chunk2 = [[0.22, 0.31], [0.31, 0.42], [0.41, 0.52]]
        interpolator.ingest_action_chunk(0, chunk1)
        interpolator.ingest_action_chunk(1, chunk2)
        res = interpolator.get_ensembled_action(1)
        assert res["has_action"] is True
        assert res["ensemble_sources_count"] == 2
        print("[TEST] All tests passed successfully!")
        return

    for line in sys.stdin:
        if not line.strip(): continue
        try:
            req = json.loads(line)
            res = run_tool(req.get("name"), req.get("arguments", {}))
            print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": res}))
        except Exception as e:
            print(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
