"""Submit a case and report whether the workflow invoked the LLM.

Usage:
    python scripts/test_case_llm_usage.py
    python scripts/test_case_llm_usage.py --api-base http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import time
from urllib.request import Request, urlopen


def _request_json(url: str, method: str = "GET", body: dict | None = None) -> dict:
    payload = None
    headers = {}
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url, data=payload, headers=headers, method=method)
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit a case and print LLM usage during that case")
    parser.add_argument("--api-base", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="Polling timeout")
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")
    case_payload = {
        "failure_type": "leak_test_fail",
        "failure_description": "LLM usage probe case for workflow verification.",
        "part_number": "HYD-VALVE-200",
        "serial_number": "SN-LLM-0001",
        "test_name": "Helium_Leak_Test",
        "test_value": 2.3e-6,
        "spec_lower": 0,
        "spec_upper": 1e-6,
    }

    submitted = _request_json(f"{api_base}/cases", method="POST", body=case_payload)
    case_id = submitted["case_id"]
    print(f"Submitted case: {case_id}")

    deadline = time.time() + args.timeout_seconds
    status_payload = {}
    while time.time() < deadline:
        status_payload = _request_json(f"{api_base}/cases/{case_id}")
        status = status_payload.get("status")
        llm_usage = status_payload.get("llm_usage")
        # The case can flip to completed slightly before post-processing fields are written.
        if status in {"completed", "failed"} and llm_usage is not None:
            break
        time.sleep(1.5)
    else:
        print("Timed out waiting for case completion")
        return 1

    print("Case status payload:")
    print(json.dumps(status_payload, indent=2))

    llm_usage = status_payload.get("llm_usage") or {}
    calls = int(llm_usage.get("calls_during_case", 0))
    print(f"LLM calls during case: {calls}")

    # Return non-zero when case finished with zero model calls.
    return 0 if calls > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
