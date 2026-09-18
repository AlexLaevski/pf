#!/usr/bin/env python3
"""Strip secrets out of logs before sharing them.

    ./scripts/redact.py --env /etc/spreadbot.env < spreadbot.log > safe.log
    journalctl -u spreadbot | ./scripts/redact.py --env /etc/spreadbot.env

The strongest pass is the one driven by ``--env``: every value in that file is
replaced wherever it appears, so even a key that leaked into a log by accident
does not survive. Pattern matching catches the rest.

This reduces risk; it does not remove it. Read the output before sending it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Pattern, Tuple

# (pattern, replacement) applied in order.
PATTERNS: List[Tuple[Pattern[str], str]] = [
    # Hex private keys and long hex blobs.
    (re.compile(r"\b0x[a-fA-F0-9]{32,}\b"), "0x<REDACTED>"),
    (re.compile(r"\b[a-fA-F0-9]{64,}\b"), "<REDACTED_HEX>"),
    # Bearer / auth tokens.
    (re.compile(r"(?i)\b(authorization|auth|bearer|token)([=:\s\"']+)\S+"), r"\1\2<REDACTED>"),
    # PEM blocks.
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
     "<REDACTED_PRIVATE_KEY>"),
    # Anything that looks like an assignment of a key-ish name.
    (re.compile(r"(?i)\b(\w*(?:private_key|secret|passwd|password|api_key)\w*)([=:\s\"']+)\S+"),
     r"\1\2<REDACTED>"),
]

MIN_SECRET_LENGTH = 6


def env_values(path: Path) -> List[str]:
    """Every non-trivial value in an env file, longest first."""
    values: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        value = line.split("=", 1)[1].strip().strip("'\"")
        if len(value) >= MIN_SECRET_LENGTH:
            values.append(value)
    # Longest first so a value containing another is replaced whole.
    return sorted(set(values), key=len, reverse=True)


def redact(text: str, secrets: List[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "<REDACTED_ENV>")
    for pattern, replacement in PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", help="file to read (default: stdin)")
    parser.add_argument("--env", action="append", default=[], help="env file whose values to redact")
    parser.add_argument("--check", action="store_true", help="exit 1 if anything was redacted")
    args = parser.parse_args(argv)

    secrets: List[str] = []
    for env_path in args.env:
        path = Path(env_path)
        if not path.exists():
            print(f"redact: {path} not found", file=sys.stderr)
            return 2
        secrets.extend(env_values(path))
    secrets = sorted(set(secrets), key=len, reverse=True)

    source = Path(args.path).read_text() if args.path else sys.stdin.read()
    cleaned = redact(source, secrets)
    sys.stdout.write(cleaned)

    if cleaned != source:
        print("redact: secrets were removed from this output", file=sys.stderr)
        if args.check:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
