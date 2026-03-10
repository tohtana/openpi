from __future__ import annotations

from pathlib import Path

from scripts.libero.validate_track_a_contracts import validate_phase


FIXTURES = Path(__file__).parent / "fixtures"


def test_validate_phase1_success() -> None:
    root = FIXTURES / "contracts_phase1"
    errors, warnings = validate_phase("phase1", root, strict=False)
    assert errors == []
    assert warnings == []


def test_validate_all_success() -> None:
    root = FIXTURES / "contracts_full"
    errors, warnings = validate_phase("all", root, strict=True)
    assert errors == []
    assert warnings == []


def test_validate_phase3_requires_phase2_pass(tmp_path: Path) -> None:
    src = FIXTURES / "contracts_full"
    dst = tmp_path / "contracts"
    dst.mkdir(parents=True)

    # Copy fixture tree using local filesystem APIs available via pathlib.
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        out = dst / rel
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.write_bytes(path.read_bytes())

    phase2 = dst / "artifacts/state/phase2_handoff.json"
    phase2.write_text(
        phase2.read_text(encoding="utf-8").replace('"status": "pass"', '"status": "blocked"'),
        encoding="utf-8",
    )

    errors, _ = validate_phase("phase3", dst, strict=False)
    assert any("Phase 3 requires phase2_handoff status=pass" in e for e in errors)
