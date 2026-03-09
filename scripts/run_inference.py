"""
Run AP1000 Digital Twin Inference Pipeline (upgraded).

Pipeline:
    Input parameters
    → Neural operator (DeepONet / DeepONetFourier / Transolver / Clifford)
    → [Optional] Diffusion turbulence super-resolution
    → Feature extraction
    → LOCAC classifier
    → Risk probability + WSS map

Usage:
    python scripts/run_inference.py
    python scripts/run_inference.py --operator deeponet_fourier
    python scripts/run_inference.py --mode time_series
    python scripts/run_inference.py --velocity 5.0 --break-size 3.0 --temperature 305
    python scripts/run_inference.py --wss           # compute wall shear stress
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import pickle
import torch
import yaml

from src.deeponet.model import DeepONet
from src.deeponet.deeponet_fourier import DeepONetFourier
from src.feature_translation.translator import FeatureTranslator
from src.physics.wall_shear_calculator import WallShearCalculator
from src.core.model_factory import print_version_banner, operator_param_count
from src.core.model_versions import get_tier_label


# ---------------------------------------------------------------------------
# Operator factory
# ---------------------------------------------------------------------------

def load_operator(operator_name: str, config: dict, checkpoint_path: Path,
                  device: torch.device) -> torch.nn.Module:
    """Load a trained neural operator from checkpoint."""
    if operator_name == "deeponet_fourier":
        model = DeepONetFourier.from_legacy_config(config)
    elif operator_name == "deeponet":
        model = DeepONet(config)
    elif operator_name == "transolver":
        from src.operators.transolver_operator import TransolverOperator
        model = TransolverOperator.from_config(config)
    elif operator_name == "clifford":
        from src.operators.clifford_operator import CliffordNeuralOperator
        model = CliffordNeuralOperator.from_config(config)
    else:
        raise ValueError(f"Unknown operator: {operator_name!r}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------------------------

class DigitalTwinInference:
    """
    End-to-end AP1000 LOCAC digital twin inference.

    Args:
        config_path:          Path to configs/config.yaml.
        model_config_path:    Path to configs/model_config.yaml.
        deeponet_path:        Path to trained operator checkpoint.
        locac_detector_path:  Path to trained LOCAC detector .pkl.
        scalers_path:         Path to scalers.pkl produced by preprocessing.
        operator:             One of deeponet | deeponet_fourier | transolver | clifford.
        use_diffusion:        If True, apply diffusion turbulence refinement.
        compute_wss:          If True, compute wall shear stress.
    """

    def __init__(
        self,
        config_path:         Path,
        model_config_path:   Path,
        deeponet_path:       Path,
        locac_detector_path: Path,
        scalers_path:        Path,
        operator:            str  = "deeponet_fourier",
        use_diffusion:       bool = False,
        compute_wss:         bool = False,
    ) -> None:
        with open(config_path)       as f: self.config       = yaml.safe_load(f)
        with open(model_config_path) as f: self.model_config = yaml.safe_load(f)
        self.config.update(self.model_config)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load neural operator
        self.operator    = load_operator(operator, self.config, deeponet_path,
                                         self.device)
        self.operator_nm = operator

        print_version_banner(
            version  = operator,
            device   = self.device,
            n_params = operator_param_count(self.operator),
        )

        # Load scalers
        with open(scalers_path, "rb") as f:
            self.scalers = pickle.load(f)

        # Load LOCAC detector
        with open(locac_detector_path, "rb") as f:
            locac_data          = pickle.load(f)
            self.locac_detector = locac_data["model"]
            self.locac_scaler   = locac_data["scaler"]

        # Feature translator
        self.translator = FeatureTranslator(config_path)

        # Load trunk coordinates from HDF5
        h5_path = (
            project_root / "data" / "deeponet_dataset" / "deeponet_dataset.h5"
        )
        with h5py.File(h5_path, "r") as f:
            self.trunk_coords_norm = torch.FloatTensor(
                f["train"]["trunk"][:]
            ).to(self.device)
        self.trunk_coords = self.scalers["trunk"].inverse_transform(
            self.trunk_coords_norm.cpu().numpy()
        )

        self.field_names = self.config["deeponet"]["output_fields"]

        # Optional modules
        self.diffusion_model = None
        if use_diffusion:
            self._load_diffusion()

        self.wss_calc = WallShearCalculator() if compute_wss else None

    # ------------------------------------------------------------------
    def _load_diffusion(self) -> None:
        diff_path = (
            project_root / "results" / "models" / "diffusion_model.pth"
        )
        if not diff_path.exists():
            print("WARNING: Diffusion model not found, skipping.")
            return
        from src.generative.diffusion_turbulence_model import (
            DiffusionTurbulenceModel,
        )
        cfg       = self.config.get("diffusion_turbulence", {})
        n_nodes   = self.trunk_coords.shape[0]
        self.diffusion_model = DiffusionTurbulenceModel(
            n_nodes=n_nodes,
            n_fields=cfg.get("n_fields", 4),
            cond_dim=cfg.get("cond_dim", 8),
            n_diff_steps=cfg.get("n_diff_steps", 1000),
        ).to(self.device).eval()
        ckpt = torch.load(diff_path, map_location=self.device, weights_only=False)
        self.diffusion_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print("✓ Diffusion turbulence model loaded")

    # ------------------------------------------------------------------
    def denormalize_field(self, data: np.ndarray, field_name: str) -> np.ndarray:
        scaler = self.scalers["targets"][field_name]
        return scaler.inverse_transform(data.reshape(-1, 1)).reshape(data.shape)

    # ------------------------------------------------------------------
    def run_inference(
        self,
        velocity:    float,
        break_size:  float,
        temperature: float,
        verbose:     bool = True,
        n_diffusion: int  = 0,
    ) -> dict:
        """
        Execute the full inference pipeline for one parameter set.

        Args:
            velocity:    Coolant inlet velocity (m/s).
            break_size:  Break size (% pipe diameter).
            temperature: Coolant temperature (°C).
            verbose:     Print step results.
            n_diffusion: Number of turbulence realisations (0 = disabled).

        Returns:
            dict with fields, features, locac_probability, locac_detected,
            inference_time_ms, coordinates, and optionally wss + turbulence_samples.
        """
        if verbose:
            print(f"\n{'='*60}\nInference: v={velocity} m/s  "
                  f"break={break_size}%  T={temperature}°C\n{'='*60}")

        # Normalise branch input
        branch_norm = self.scalers["branch"].transform(
            [[velocity, break_size, temperature]]
        )
        branch = torch.FloatTensor(branch_norm).to(self.device)

        # Operator forward pass
        t0 = time.perf_counter()
        with torch.no_grad():
            predictions = self.operator(branch, self.trunk_coords_norm)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if verbose:
            print(f"✓ {self.operator_nm}: {elapsed_ms:.3f} ms")

        pred_np = predictions.cpu().numpy()[0]           # [n_fields, N]
        fields  = {
            fn: self.denormalize_field(pred_np[i], fn)
            for i, fn in enumerate(self.field_names)
        }

        # Optional diffusion refinement
        turbulence_samples = None
        if self.diffusion_model is not None and n_diffusion > 0:
            with torch.no_grad():
                vel_tensor = predictions[:, :, :]        # [1, n_fields, N]
                tke_tensor = predictions[:, 2, :]        # [1, N] (turbulence_k)
                turbulence_samples = self.diffusion_model.sample(
                    vel_tensor, tke_tensor, n_samples=n_diffusion
                ).cpu().numpy()
            if verbose:
                print(f"✓ Diffusion: {n_diffusion} turbulence realisations")

        # Feature extraction
        features = self.translator.extract_features(fields, self.trunk_coords)
        if verbose:
            print("✓ Features:")
            for k, v in features.items():
                print(f"    {k}: {v:.4f}")

        # LOCAC classification
        fv = np.array([
            features["average_pressure"],
            features["mass_flow_rate"],
            features["avg_temperature"],
            features["pressure_drop"],
            features["max_turbulence"],
            features["temperature_difference"],
            features["velocity_std"],
        ]).reshape(1, -1)
        fv_scaled  = self.locac_scaler.transform(fv)
        locac_prob = self.locac_detector.predict_proba(fv_scaled)[0, 1]
        locac_det  = bool(locac_prob > 0.5)

        if verbose:
            status = "⚠ LOCAC DETECTED!" if locac_det else "✓ NORMAL"
            print(f"\n{status}  probability={locac_prob:.4f}")

        result = {
            "input_params":      {"velocity": velocity, "break_size": break_size,
                                  "temperature": temperature},
            "fields":            fields,
            "features":          features,
            "locac_probability": float(locac_prob),
            "locac_detected":    locac_det,
            "inference_time_ms": elapsed_ms,
            "coordinates":       self.trunk_coords,
            "operator":          self.operator_nm,
        }

        # Optional WSS
        if self.wss_calc is not None:
            wss_res = self.wss_calc.compute_wss(
                fields["velocity_magnitude"], self.trunk_coords
            )
            risk    = self.wss_calc.assess_corrosion_risk(wss_res)
            result["wss"]             = wss_res
            result["corrosion_risk"]  = risk
            if verbose:
                print(f"✓ WSS: mean={wss_res['mean_wss']:.3f} Pa  "
                      f"risk={risk['risk_level']}")

        if turbulence_samples is not None:
            result["turbulence_samples"] = turbulence_samples

        return result

    # ------------------------------------------------------------------
    def run_time_series(self, param_sequence: list, duration: float = 60) -> list:
        """Run inference over a sequence of parameter sets."""
        dt   = duration / len(param_sequence)
        results = []
        for i, (v, b, t) in enumerate(param_sequence):
            r         = self.run_inference(v, b, t, verbose=False)
            r["time"] = i * dt
            results.append(r)
            status = "⚠ LOCAC!" if r["locac_detected"] else "OK"
            print(f"t={i*dt:5.1f}s  prob={r['locac_probability']:.4f}  {status}")
        return results

    # ------------------------------------------------------------------
    def benchmark(self, cfd_ref: float = 3600.0, n_repeats: int = 10) -> dict:
        """Measure inference speed vs. CFD reference time."""
        params  = [[5.0, 0.0, 305.0], [4.5, 2.5, 300.0], [5.5, 5.0, 310.0]]
        times   = []
        for p in params * (n_repeats // 3 + 1):
            r = self.run_inference(*p, verbose=False)
            times.append(r["inference_time_ms"])

        mean_ms = sum(times) / len(times)
        speedup = cfd_ref / (mean_ms / 1000)
        print(f"\n{'='*50}\nBenchmark: {self.operator_nm}")
        print(f"  CFD ref.   : {cfd_ref:.0f} s")
        print(f"  Inference  : {mean_ms:.3f} ms")
        print(f"  Speedup    : {speedup:,.0f}×")
        if speedup >= 1000:
            print("  ✓ >1000× target achieved!")
        return {"mean_ms": mean_ms, "speedup": speedup}


# ---------------------------------------------------------------------------
# Convenience test functions
# ---------------------------------------------------------------------------

def test_single_case(pipeline: DigitalTwinInference) -> None:
    test_cases = [
        (5.0, 0.0, 305.0, "Normal operation"),
        (4.5, 5.0, 290.0, "Partial LOCAC"),
        (3.5, 9.0, 280.0, "Large LOCAC"),
    ]
    for v, b, t, label in test_cases:
        r = pipeline.run_inference(v, b, t, verbose=False)
        print(f"{label:20s}  prob={r['locac_probability']:.4f}  "
              f"{'LOCAC' if r['locac_detected'] else 'normal'}")


def test_time_series(pipeline: DigitalTwinInference) -> None:
    import numpy as np
    n_steps = 30
    velocities   = np.linspace(5.0, 3.5, n_steps)
    break_sizes  = np.linspace(0.0, 8.0, n_steps)
    temperatures = np.linspace(305.0, 285.0, n_steps)
    params = list(zip(velocities, break_sizes, temperatures))
    results = pipeline.run_time_series(params, duration=60)
    try:
        import matplotlib.pyplot as plt
        times = [r["time"] for r in results]
        probs = [r["locac_probability"] for r in results]
        plt.figure(figsize=(12, 4))
        plt.plot(times, probs, "b-", linewidth=2, label="LOCAC Probability")
        plt.axhline(0.5, color="r", linestyle="--", label="Threshold")
        plt.fill_between(times, 0, 1,
                         where=[p > 0.5 for p in probs],
                         alpha=0.3, color="red", label="LOCAC Detected")
        plt.xlabel("Time (s)"); plt.ylabel("Probability")
        plt.title("AP1000 LOCAC Detection — Time Series")
        plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(0, 1)
        plot_path = project_root / "results" / "plots" / "locac_time_series.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150); plt.close()
        print(f"✓ Time-series plot saved to {plot_path}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AP1000 Digital Twin Inference (upgraded)"
    )
    p.add_argument("--model-version", default=None,
                   dest="model_version",
                   choices=["deeponet", "deeponet_fourier",
                            "transolver", "clifford", "mamba", "diffusion"],
                   help="Select model architecture (overrides --operator and config)")
    p.add_argument("--operator",     default="deeponet_fourier",
                   choices=["deeponet", "deeponet_fourier",
                            "transolver", "clifford"],
                   help="Operator name (legacy, prefer --model-version)")
    p.add_argument("--mode",         choices=["single", "time_series", "benchmark"],
                   default="single")
    p.add_argument("--velocity",     type=float, default=5.0)
    p.add_argument("--break-size",   type=float, default=2.0, dest="break_size")
    p.add_argument("--temperature",  type=float, default=305.0)
    p.add_argument("--diffusion",    type=int,   default=0,
                   help="Number of turbulence realisations (0 = disabled)")
    p.add_argument("--wss",          action="store_true",
                   help="Compute wall shear stress")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path       = project_root / "configs" / "config.yaml"
    model_config_path = project_root / "configs" / "model_config.yaml"
    models_dir        = project_root / "results" / "models"
    scalers_path      = (
        project_root / "data" / "deeponet_dataset" / "scalers.pkl"
    )

    # Resolve operator: CLI --model-version > CLI --operator > config
    operator = args.model_version or args.operator
    if operator == "deeponet_fourier" and not args.model_version:
        # Check model_config for a non-default value
        try:
            with open(model_config_path) as _f:
                _mc = yaml.safe_load(_f)
            operator = _mc.get("model_version") or _mc.get("operator") or operator
        except Exception:
            pass

    # Resolve the correct checkpoint for the requested architecture.
    # train_operator.py saves  → results/models/<operator>_best.pth
    # train_deeponet.py saves  → results/models/best_model.pth
    _CKPT_MAP = {
        "transolver":      models_dir / "transolver_best.pth",
        "clifford":        models_dir / "clifford_best.pth",
        "mamba":           models_dir / "mamba_best.pth",
        "diffusion":       models_dir / "diffusion_model.pth",
        "deeponet_fourier": models_dir / "best_model.pth",
        "deeponet":        models_dir / "best_model.pth",
    }
    operator_ckpt = _CKPT_MAP.get(operator, models_dir / "best_model.pth")
    locac_ckpt    = models_dir / "locac_detector.pkl"

    if not operator_ckpt.exists():
        print(f"ERROR: No checkpoint found for '{operator}' at {operator_ckpt}")
        if operator in ("transolver", "clifford"):
            print(f"  Train it first:  python scripts/train_operator.py --model-version {operator}")
        elif operator in ("mamba", "diffusion"):
            print(f"  Train it first:  python scripts/train_deeponet.py --model-version {operator}")
        else:
            print("  Train it first:  python scripts/train_deeponet.py")
        return
    for p_req in [locac_ckpt, scalers_path]:
        if not p_req.exists():
            print(f"ERROR: Required file not found: {p_req}")
            print("Train models first:  train_deeponet.py + train_locac_model.py")
            return

    pipeline = DigitalTwinInference(
        config_path         = config_path,
        model_config_path   = model_config_path,
        deeponet_path       = operator_ckpt,
        locac_detector_path = locac_ckpt,
        scalers_path        = scalers_path,
        operator            = operator,
        use_diffusion       = args.diffusion > 0,
        compute_wss         = args.wss,
    )

    if args.mode == "single":
        result = pipeline.run_inference(
            args.velocity, args.break_size, args.temperature,
            n_diffusion=args.diffusion,
        )
        if args.wss and "wss" in result:
            out = project_root / "results" / "predictions" / "wss_map.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            pipeline.wss_calc.save_wss_map(result["wss"], out)
    elif args.mode == "time_series":
        test_time_series(pipeline)
    elif args.mode == "benchmark":
        pipeline.benchmark()


if __name__ == "__main__":
    print("=" * 60 + "\nSTEP 8: Run Inference Pipeline (Upgraded)\n" + "=" * 60)
    main()
