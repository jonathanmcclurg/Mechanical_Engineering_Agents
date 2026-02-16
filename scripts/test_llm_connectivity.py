"""Simple API-level LLM connectivity test.

Usage:
    python scripts/test_llm_connectivity.py
    python scripts/test_llm_connectivity.py --api-base http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def run(api_base: str) -> int:
    endpoint = api_base.rstrip("/") + "/health/llm"
    request = Request(endpoint, method="GET")

    try:
        with urlopen(request, timeout=20) as response:
            status_code = response.getcode()
            body = response.read().decode("utf-8")
    except HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print(f"LLM connectivity test failed ({e.code}) at {endpoint}")
        print(error_body)
        return 1
    except URLError as e:
        print(f"Unable to reach API at {endpoint}: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error calling {endpoint}: {e}")
        return 1

    if status_code != 200:
        print(f"Unexpected status code: {status_code}")
        print(body)
        return 1

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        print("API returned non-JSON payload:")
        print(body)
        return 1

    print("LLM connectivity check response:")
    print(json.dumps(payload, indent=2))

    if payload.get("status") != "ok":
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check API-to-LLM connectivity")
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000",
        help="Base URL for the RCA API",
    )
    args = parser.parse_args()
    return run(args.api_base)


if __name__ == "__main__":
    sys.exit(main())
