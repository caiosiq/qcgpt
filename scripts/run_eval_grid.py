import os
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--num_rows", type=int, default=None)
    parser.add_argument("--num_cols", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--max_gates", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    env = os.environ.copy()
    root = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH','')}"

    out_dir = args.out_dir or env.get("OUT_DIR", "model_evaluations")
    run_name = args.run_name or env.get("RUN_NAME", None)
    if run_name is None:
        from datetime import datetime
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt = args.ckpt or env.get("CKPT", "checkpoints/rl_finetuned.pt")
    num_rows = args.num_rows or int(env.get("NUM_ROWS", "16"))
    num_cols = args.num_cols or int(env.get("NUM_COLS", "16"))
    max_len = args.max_len or int(env.get("MAX_LEN", "32"))
    max_gates = args.max_gates or int(env.get("MAX_GATES", "6"))

    cmd = [
        "python3",
        str(root / "scripts" / "eval_grid.py"),
        "--ckpt",
        ckpt,
        "--num_rows",
        str(num_rows),
        "--num_cols",
        str(num_cols),
        "--max_len",
        str(max_len),
        "--max_gates_ref",
        str(max_gates),
        "--out_dir",
        out_dir,
        "--run_name",
        run_name,
    ]

    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()