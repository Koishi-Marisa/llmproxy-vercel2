"""Wait for the new build to go live, then run the full suite against it.

The commit under test changes the Gemini route so that a request with no
credentials returns 401 (it used to be 500 for a malformed header and 422 for a
missing one). That is a remote-visible marker of the new build, so we poll it
until it flips, then run verify_proxy.py and test_all_models.py against the
deployment.
"""
import json
import subprocess
import sys
import time

import httpx

BASE = "https://ai.xiaowan.dpdns.org"
PY = r".\.venv\Scripts\python.exe"


def gemini_ok():
    """New build: no credentials on the Gemini route -> 401 (was 500/422)."""
    try:
        r = httpx.post(f"{BASE}/gemini/chat/completions",
                       json={"model": "gemini-1.5-flash",
                             "messages": [{"role": "user", "content": "hi"}]},
                       timeout=30)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}"
    return r.status_code == 401, f"HTTP {r.status_code}"


print(f"Waiting for the new build at {BASE} ...")
live = False
for attempt in range(1, 41):
    ok, detail = gemini_ok()
    print(f"  attempt {attempt}: {detail}")
    if ok:
        live = True
        break
    time.sleep(8)

if not live:
    print("TIMED OUT waiting for the new build")
    sys.exit(2)

print("\nNew build detected. Running suites.\n")
results = {}
for label, args in [
    ("verify_proxy", ["verify_proxy.py", "--base", BASE]),
    ("all_models", ["test_all_models.py", BASE]),
]:
    print("=" * 70)
    print(f"RUNNING {label}")
    print("=" * 70)
    p = subprocess.run([PY] + args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    print(p.stdout)
    if p.stderr.strip():
        print("STDERR:", p.stderr[-2000:])
    results[label] = p.returncode
    print(f"{label} exit={p.returncode}\n")

print("=" * 70)
print("SUMMARY:", {k: ("PASS" if v == 0 else f"FAIL({v})") for k, v in results.items()})
print("=" * 70)
sys.exit(0 if all(v == 0 for v in results.values()) else 1)
