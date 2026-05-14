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
	seeds = [0, 1, 2]

	for run_idx, seed in enumerate(seeds):
		config = NS2DConfig()
		config.seed = seed
		config.reweight_config = build_reweight_config(args)

		model = NS2D(config)
		run_dir = run_base_dir / f"run_{run_idx:02d}_seed_{seed}"
		run_dir.mkdir(parents=True, exist_ok=True)
		log_path = run_dir / "train.log"

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


####### bc weight 100 最终数据
# pinn batch_20260510_152326 adam_5000_lbfgs_3000_seed_*_decay_epsi_0.0_data_weight_0.0frame_weight_0.0_reweight_causal_1000_5000_reweight_adaptive_100000_100000
# p_l2_relative             0.09832239303    0.0001692071937
# p_max_abs                  0.3844263232    0.0002016795355
# p_mse                   0.0004553224768    1.292360775e-08
# pde_residual              0.03032994146    5.750425448e-06
# u_l2_relative             0.07094406235    0.0001976572606
# u_max_abs                 0.08170308661     0.000404507963
# u_mse                   0.0007454498569    8.142799128e-08
# v_l2_relative              0.1055923775     0.000235574669
# v_max_abs                  0.1024957284    0.0002127218247
# v_mse                   0.0007831645569    5.031624683e-08


  
# pinn-c1 w/o bridge
# batch_20260511_033201 adam_5000_lbfgs_3000_seed_*_decay_epsi_1.0_data_weight_0.0frame_weight_0.0_reweight_causal_1000_3000_reweight_adaptive_100000_100000
# p_l2_relative              0.1129734387    0.0006226547692
# p_max_abs                   0.380987324    0.0003676287769
# p_mse                   0.0006196094552    6.484031402e-08
# pde_residual              0.03184721929    1.908215463e-05
# u_l2_relative             0.09017026498    0.0006052882283
# u_max_abs                  0.1024591063    0.0008515024043
# u_mse                    0.001244996335    3.874296694e-07
# v_l2_relative               0.132715703     0.001216852479
# v_max_abs                  0.1286017457      0.00137794493
# v_mse                    0.001295284222     3.78795879e-07

# pinn-c1 with bridge
# batch_20260510_212355 adam_5000_lbfgs_3000_seed_*_decay_epsi_1.0_data_weight_10.0frame_weight_0.0_reweight_causal_1000_3000_reweight_adaptive_100000_100000
# p_l2_relative             0.09054548786    0.0002829395001
# p_max_abs                  0.3815583663    0.0004155296861
# p_mse                   0.0003925972974    2.249833624e-08
# pde_residual              0.03026673322    1.900832646e-05
# u_l2_relative             0.06439930535    0.0002582633736
# u_max_abs                 0.07403461957    0.0003462354674
# u_mse                   0.0006278497699    1.030458279e-07
# v_l2_relative             0.09661490822     0.000458201521
# v_max_abs                 0.09377810685    0.0004405348836
# v_mse                   0.0006736083726    9.410858064e-08


# pinn-c2
# "grad_norms_scale": 0.5,"scale": 3.0
# batch_20260511_033452 adam_5000_lbfgs_3000_seed_*_decay_epsi_1.0_data_weight_10.0frame_weight_0.0_reweight_causal_1000_3000_reweight_adaptive_6000_8000
# p_l2_relative             0.07421626578    5.082280523e-05
# p_max_abs                  0.3949213303    0.0004650299191
# p_mse                   0.0002573152607     2.53556378e-09
# pde_residual              0.02115930369    1.230713292e-05
# u_l2_relative             0.04548975261    2.228807398e-05
# u_max_abs                 0.05332851139    5.469969819e-05
# u_mse                   0.0002980828579    4.024033435e-09
# v_l2_relative             0.06658186708    4.208078413e-05
# v_max_abs                 0.05920611732    7.187187445e-05
# v_mse                    0.000307838229    3.777786764e-09