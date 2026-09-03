"""Audit the evidence used by Journal/completed_todo/paper.tex.

Run from the repository root with:
    conda run -n minor_proj python Journal/scripts/edit_013_completed_paper_audit.py

This script performs no training and does not modify checkpoints or datasets.
It writes a compact, reproducible inventory to
Journal/completed_todo/artifact_audit.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import h5py
from PIL import Image
import sklearn
import torch
import yaml


os.environ.setdefault("MPLCONFIGDIR", "/tmp/minorproj-matplotlib")


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "Journal" / "completed_todo" / "artifact_audit.json"
MODELS = ROOT / "results" / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_from(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(artifact, dict) and "model_state_dict" in artifact:
        return artifact["model_state_dict"], artifact
    if isinstance(artifact, dict) and "state_dict" in artifact:
        return artifact["state_dict"], artifact
    if isinstance(artifact, dict):
        return artifact, {}
    raise TypeError(f"Unsupported checkpoint object in {path}")


def checkpoint_record(path: Path, model: torch.nn.Module) -> dict[str, Any]:
    state, metadata = state_dict_from(path)
    incompat = model.load_state_dict(state, strict=False)
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_tensors": len(state),
        "missing_keys": list(incompat.missing_keys),
        "unexpected_keys": list(incompat.unexpected_keys),
        "metadata_keys": sorted(metadata.keys()),
        "epoch": metadata.get("epoch"),
        "operator": metadata.get("operator"),
    }


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        return sum(1 for _ in reader)


def main() -> None:
    from src.deeponet.deeponet_fourier import DeepONetFourier
    from src.operators.clifford_operator import CliffordNeuralOperator
    from src.operators.transolver_operator import TransolverOperator
    sys.path.insert(0, str(ROOT / "Journal" / "scripts"))
    from edit_005_train_baseline_deeponet import (
        BRANCH_DIMS,
        N_OUTPUTS,
        TRUNK_DIMS,
        VanillaDeepONet,
    )

    with (ROOT / "configs" / "model_config.yaml").open() as stream:
        model_cfg = yaml.safe_load(stream)

    dataset_path = ROOT / "data" / "deeponet_dataset" / "deeponet_dataset.h5"
    with h5py.File(dataset_path, "r") as h5:
        dataset = {
            split: {name: list(h5[split][name].shape) for name in ("branch", "trunk", "targets")}
            for split in ("train", "val", "test")
        }

    checkpoints = {
        "baseline_deeponet": checkpoint_record(
            MODELS / "baseline_deeponet_best.pth",
            VanillaDeepONet(BRANCH_DIMS, TRUNK_DIMS, N_OUTPUTS),
        ),
        "deeponet_fourier": checkpoint_record(
            MODELS / "deeponet_fourier_best.pth", DeepONetFourier(model_cfg)
        ),
        "transolver": checkpoint_record(
            MODELS / "transolver_best.pth", TransolverOperator.from_config(model_cfg)
        ),
        "clifford": checkpoint_record(
            MODELS / "clifford_best.pth", CliffordNeuralOperator.from_config(model_cfg)
        ),
    }

    classifier_path = MODELS / "locac_detector.pkl"
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with classifier_path.open("rb") as stream:
            classifier = pickle.load(stream)
    classifier_record = {
        "path": str(classifier_path.relative_to(ROOT)),
        "bytes": classifier_path.stat().st_size,
        "sha256": sha256(classifier_path),
        "keys": sorted(classifier.keys()),
        "metrics": classifier.get("metrics"),
        "load_warnings": sorted({str(item.message) for item in caught_warnings}),
    }

    nppad_root = ROOT / "data" / "nppad" / "operation_csv_data"
    nppad = {}
    for class_name in ("Normal", "LOCAC"):
        files = sorted((nppad_root / class_name).glob("*.csv"))
        nppad[class_name] = {
            "files": len(files),
            "rows": sum(count_csv_rows(path) for path in files),
        }

    figure_root = ROOT / "Journal" / "completed_todo" / "figures"
    figures = {}
    for path in sorted(figure_root.glob("*.png")):
        with Image.open(path) as image:
            figures[path.name] = {
                "pixels": list(image.size),
                "format": image.format,
                "dpi": list(image.info.get("dpi", ())),
                "sha256": sha256(path),
            }

    baseline_metrics_path = MODELS / "baseline_deeponet_results.json"
    field_metrics_path = ROOT / "Journal" / "scripts" / "edit_010_table_vii_metrics.json"
    ablation_path = ROOT / "Journal" / "scripts" / "edit_008d_ablation_results.json"
    transolver_seed42_source = (
        ROOT / "results" / "multiseed_sweep" / "20260902_231150"
        / "seed_42_transolver_metrics.json"
    )
    transolver_seed42_path = (
        ROOT / "Journal" / "completed_todo" / "evidence"
        / "seed_42_transolver_metrics.json"
    )

    manuscript_path = ROOT / "Journal" / "completed_todo" / "paper.tex"
    manuscript = manuscript_path.read_text(encoding="utf-8")
    cited_keys = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", manuscript)
        for key in group.split(",")
    }
    bibliography_keys = set(re.findall(r"\\bibitem\{([^}]+)\}", manuscript))
    included_graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", manuscript)
    forbidden_markers = [
        marker
        for marker in (r"\todo", "[VERIFY", "[Input", r"\color{red}", r"\color{green}")
        if marker.lower() in manuscript.lower()
    ]
    manuscript_record = {
        "path": str(manuscript_path.relative_to(ROOT)),
        "sha256": sha256(manuscript_path),
        "cited_keys": sorted(cited_keys),
        "bibliography_keys": sorted(bibliography_keys),
        "cited_but_missing": sorted(cited_keys - bibliography_keys),
        "uncited_bibliography": sorted(bibliography_keys - cited_keys),
        "included_graphics": included_graphics,
        "missing_graphics": sorted(
            graphic
            for graphic in included_graphics
            if not (manuscript_path.parent / graphic).exists()
        ),
        "forbidden_markers": forbidden_markers,
    }

    report = {
        "command": "conda run -n minor_proj python Journal/scripts/edit_013_completed_paper_audit.py",
        "torch_version": torch.__version__,
        "sklearn_runtime_version": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "dataset": {"path": str(dataset_path.relative_to(ROOT)), "shapes": dataset},
        "checkpoints": checkpoints,
        "classifier": classifier_record,
        "nppad": nppad,
        "figures": figures,
        "manuscript": manuscript_record,
        "metric_artifacts": {
            "baseline": json.loads(baseline_metrics_path.read_text()),
            "deeponet_fourier": json.loads(field_metrics_path.read_text()),
            "transolver_seed42": {
                "path": str(transolver_seed42_path.relative_to(ROOT)),
                "sha256": sha256(transolver_seed42_path),
                "source_path": str(transolver_seed42_source.relative_to(ROOT)),
                "source_available": transolver_seed42_source.exists(),
                "source_sha256": (
                    sha256(transolver_seed42_source)
                    if transolver_seed42_source.exists() else None
                ),
                "source_content_matches_archive": (
                    transolver_seed42_source.exists()
                    and json.loads(transolver_seed42_source.read_text())
                    == json.loads(transolver_seed42_path.read_text())
                ),
                "record": json.loads(transolver_seed42_path.read_text()),
            },
            "translation_ablation": json.loads(ablation_path.read_text()),
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")

    incompatible = [
        name
        for name, record in checkpoints.items()
        if record["missing_keys"] or record["unexpected_keys"]
    ]
    if incompatible:
        raise RuntimeError(f"Incompatible checkpoints: {incompatible}")
    if manuscript_record["cited_but_missing"] or manuscript_record["missing_graphics"]:
        raise RuntimeError(f"Manuscript dependency failure: {manuscript_record}")
    if manuscript_record["forbidden_markers"]:
        raise RuntimeError(
            f"Unresolved manuscript markers: {manuscript_record['forbidden_markers']}"
        )

    print(f"Audit written to {OUTPUT.relative_to(ROOT)}")
    print(f"Dataset splits: {dataset}")
    print("Checkpoint parameters:")
    for name, record in checkpoints.items():
        compatible = not record["missing_keys"] and not record["unexpected_keys"]
        print(f"  {name}: {record['trainable_parameters']:,} (compatible={compatible})")
    print(f"NPPAD rows: Normal={nppad['Normal']['rows']:,}, LOCAC={nppad['LOCAC']['rows']:,}")
    print(f"Classifier metrics: {classifier_record['metrics']}")
    print(f"Figures checked: {len(figures)}")
    print(
        "Manuscript checks: "
        f"{len(cited_keys)} cited keys, "
        f"{len(bibliography_keys)} bibliography entries, "
        "no missing dependencies or unresolved markers"
    )


if __name__ == "__main__":
    main()
