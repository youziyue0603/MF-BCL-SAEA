"""Audit the local reconstruction against the explicit manuscript contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "paper_contract.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manuscript_path(contract: dict) -> Path:
    declared = (ROOT / contract["manuscript"]["source"]).resolve()
    if declared.exists():
        return declared
    candidates = sorted(ROOT.parent.glob("*/1.0.pdf"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Unable to identify the authoritative 1.0 manuscript from {declared}.")


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    manuscript = manuscript_path(contract)
    locked = contract["paper_locked_protocol"]
    required = {
        "design_dimension": 14,
        "objective_dimension": 2,
        "solver": "Abaqus 2024",
        "element": "CAX4RT",
        "population_size": 50,
        "maximum_generations": 200,
        "high_fidelity_budget": 200,
        "high_fidelity_upgrades_per_generation": 5,
        "ensemble_members": 5,
        "ensemble_epochs": 50,
        "ewc_lambda": 1000,
        "independent_run_count": 15,
    }
    mismatch = {key: (locked.get(key), value) for key, value in required.items() if locked.get(key) != value}
    if mismatch:
        raise SystemExit(f"Contract mismatch: {mismatch}")
    print("MF-BCL-SAEA contract audit: PASS")
    print(f"Authoritative PDF: {manuscript}")
    print(f"PDF SHA-256: {sha256(manuscript)}")
    print("Boundary: source-only Abaqus decks, constraints, raw records and stopping rule remain unresolved.")


if __name__ == "__main__":
    main()
