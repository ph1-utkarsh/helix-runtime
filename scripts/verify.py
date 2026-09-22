"""Validate the headline Helix evidence without rerunning benchmarks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
metal = json.loads((ROOT / "helix/experiments/EXP-HEL-003/metrics.json").read_text())
advanced = json.loads((ROOT / "helix/experiments/EXP-HEL-005/metrics.json").read_text())
assert metal["all_within_1e-5"]
assert max(row["max_abs_error"] for row in metal["rows"]) < 1e-5
assert advanced["quantization"]["reduction_percent"] == 71.875
assert advanced["speculative"]["matched"]["exact_target_output"]
assert advanced["speculative"]["adversarial"]["exact_target_output"]
assert advanced["paid_spend"] == 0
print("HELIX VERIFY: PASS")
