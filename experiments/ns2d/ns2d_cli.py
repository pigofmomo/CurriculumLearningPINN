"""Command-line multi-seed runner for the NS2D benchmark. / NS2D基准的命令行多随机种子运行器。"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))
PINN_PRO_DIR = ROOT_DIR / "pinn_pro"
if str(PINN_PRO_DIR) not in sys.path:
	sys.path.insert(0, str(PINN_PRO_DIR))

import torch

from experiments.ns2d.ns2d import BASE_DIR, NS2D, NS2DConfig
from experiments.utils import summarize_batch
import pinn_pro


class _Tee:
	"""Mirror console output to a log file. / 将终端输出同步写入日志文件。"""
	def __init__(self, *streams):
		self.streams = streams

	def write(self, text):
		for stream in self.streams:
			stream.write(text)
			stream.flush()

	def flush(self):
		for stream in self.streams:
			stream.flush()


@contextmanager
def tee_output(log_path: Path):
	log_path.parent.mkdir(parents=True, exist_ok=True)
	with log_path.open("w", encoding="utf-8") as log_file:
		original_stdout = sys.stdout
		original_stderr = sys.stderr
		sys.stdout = _Tee(original_stdout, log_file)
		sys.stderr = _Tee(original_stderr, log_file)
		try:
			yield
		finally:
			sys.stdout = original_stdout
			sys.stderr = original_stderr


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run NS2D with CLI-reconfigurable reweight settings.")
	parser.add_argument("--cuda-device", type=int, choices=[0, 1], default=1, dest="cuda_device", help="CUDA device index to use when CUDA is available.")
	parser.add_argument("--reweight-epsi", type=float, default=1.0, dest="decay_epsi")
	parser.add_argument("--num-subdomains", type=int, nargs=2, default=[5, 5], metavar=("NX", "NY"))
	parser.add_argument("--reweight-every", type=int, default=500, dest="reweight_every")
	parser.add_argument("--reweight-causal-begin", type=int, default=1000, dest="reweight_causal_begin")
	parser.add_argument("--reweight-causal-end", type=int, default=5000, dest="reweight_causal_end")
	parser.add_argument("--reweight-adaptive-begin", type=int, default=100000, dest="reweight_adaptive_begin")
	parser.add_argument("--reweight-adaptive-end", type=int, default=100000, dest="reweight_adaptive_end")
	parser.add_argument("--scale", type=float, default=2.0)
	parser.add_argument("--grad-norms-scale", type=float, default=1.0, dest="grad_norms_scale")
	parser.add_argument("--low-fre-n", type=int, default=3, dest="low_fre_n")
	parser.add_argument("--low-fre-data-weight", type=float, default=0.0, dest="low_fre_data_weight")
	parser.add_argument("--frame-data-weight", type=float, default=0.0, dest="frame_data_weight")
	parser.add_argument("--log", dest="log", action="store_true", default=True, help="Enable logging flag in reweight_config.")
	parser.add_argument("--no-log", dest="log", action="store_false", help="Disable logging flag in reweight_config.")
	return parser


def build_reweight_config(args: argparse.Namespace) -> dict:
	"""Collect CLI options into the reweight config. / 将命令行参数整理为重加权配置。"""
	return {
		"decay_epsi": args.decay_epsi,
		"num_subdomains": list(args.num_subdomains),
		"reweight_every": args.reweight_every,
		"reweight_causal_begin": args.reweight_causal_begin,
		"reweight_causal_end": args.reweight_causal_end,
		"reweight_adaptive_begin": args.reweight_adaptive_begin,
		"reweight_adaptive_end": args.reweight_adaptive_end,
		"log": args.log,
		"scale": args.scale,
		"grad_norms_scale": args.grad_norms_scale,
		"low_fre_n": args.low_fre_n,
		"low_fre_data_weight": args.low_fre_data_weight,
		"frame_data_weight": args.frame_data_weight,
	}


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
	run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_base_dir = BASE_DIR / "multi_runs_cli" / f"batch_{run_tag}"
	run_base_dir.mkdir(parents=True, exist_ok=True)
	# Run multiple independent seeds in one CLI invocation. / 一次命令行调用运行多个独立随机种子。
	seeds = [0, 1, 2]

	for run_idx, seed in enumerate(seeds):
		config = NS2DConfig()
		config.seed = seed
		config.reweight_config = build_reweight_config(args)

		model = NS2D(config)
		run_dir = run_base_dir / f"run_{run_idx:02d}_seed_{seed}"
		run_dir.mkdir(parents=True, exist_ok=True)
		log_path = run_dir / "train.log"

		# Keep each seed in an isolated run directory and log. / 每个随机种子使用独立目录和日志。
		with tee_output(log_path):
			print(f"Using device: {device}")
			if device.type == "cuda":
				torch.cuda.set_device(device)
			print(f"Run base dir: {run_dir}")
			print(f"Log path: {log_path}")
			print(f"Seed: {seed}")
			print(f"Reweight config: {config.reweight_config}")
			pinn_runner = pinn_pro.PINNWeightedSamples(model, run_dir)
			pinn_runner.train_and_evaluate()

	summarize_batch(run_base_dir)

if __name__ == "__main__":
	main()
