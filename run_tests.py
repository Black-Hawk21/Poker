#!/usr/bin/env python3
"""
Run All Tests
=============
Executes every test suite and prints a combined summary.
Usage:  python run_tests.py
"""

import subprocess, sys, time

SUITES = [
    ("Phase 1 — Poker Engine",    "test_phase1.py"),
    ("Phase 2 — Decision Engine", "test_phase2.py"),
    ("Phase 3 — Range Inference", "test_phase3.py"),
    ("Phase 4 — Spectator Learn", "test_phase4.py"),
    ("Phase 5 — Exploitation",    "test_phase5.py"),
    ("Game Environment",          "test_environment.py"),
    ("Phase 6 — RNG Inference",   "test_phase6.py"),
]

total_passed = 0
total_failed = 0
results = []
t0_all = time.time()

for label, script in SUITES:
    print(f"\n{'━' * 60}")
    print(f"  Running: {label}  ({script})")
    print(f"{'━' * 60}")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        timeout=300,
    )
    dt = time.time() - t0

    # Parse pass/fail from exit code
    ok = proc.returncode == 0
    results.append((label, ok, dt))

dt_all = time.time() - t0_all

print(f"\n{'━' * 60}")
print(f"  COMBINED RESULTS")
print(f"{'━' * 60}")

all_ok = True
for label, ok, dt in results:
    status = "✓ PASS" if ok else "✗ FAIL"
    if not ok:
        all_ok = False
    print(f"  {status}  {label:<30}  ({dt:.1f}s)")

print(f"\n  Total time: {dt_all:.1f}s")
print(f"{'━' * 60}\n")
sys.exit(0 if all_ok else 1)
