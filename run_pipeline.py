"""Reproduce every number in the README: split -> train (leak-free and leaky) -> evaluate with the classical baseline -> export and benchmark.

    python run_pipeline.py

About 45 minutes on a 4-thread laptop CPU: training, then the INT8 layer search.
"""

import json
import time

from tiles import data, evaluate, export, train


def main() -> None:
    started = time.perf_counter()
    if not data.SPLITS_CSV.exists():
        data.build_splits()
    train.main(["--arch", "mobilenet_v3_small"])
    train.main(["--arch", "mobilenet_v3_small", "--split", "random"])  # leakage check only
    evaluate.main([])
    export.main([])
    seconds = round(time.perf_counter() - started, 1)
    (data.ROOT / "reports" / "pipeline_summary.json").write_text(json.dumps({"seconds": seconds}, indent=2))
    print(f"Pipeline finished in {seconds} s")


if __name__ == "__main__":
    main()
