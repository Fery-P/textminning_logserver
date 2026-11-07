"""Parsing Apache/NGINX combined access logs into a structured DataFrame."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Apache/NGINX "combined" log format. Trailing fields (e.g. response time) are ignored.
LOG_PATTERN = re.compile(
    r'^(?P<ip>\S+)\s+(?P<ident>\S+)\s+(?P<auth>\S+)\s+'
    r'\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<request>[^"]*)"\s+'
    r'(?P<status>\d{3})\s+(?P<bytes>\S+)\s+'
    r'"(?P<referer>[^"]*)"\s+'
    r'"(?P<user_agent>[^"]*)"'
    r'(?:\s+.*)?$'
)

TIMESTAMP_FORMAT = "%d/%b/%Y:%H:%M:%S %z"


def parse_line(line: str) -> dict | None:
    """Parse one log line. Returns None when the line does not match the format."""
    match = LOG_PATTERN.match(line)
    if not match:
        return None
    rec = match.groupdict()
    parts = rec["request"].split()
    method = parts[0].upper() if len(parts) >= 2 else None
    url = parts[1] if len(parts) >= 2 else None
    protocol = parts[2] if len(parts) >= 3 else None
    return {
        "ip": rec["ip"],
        "timestamp": rec["timestamp"],
        "http_method": method,
        "url": url,
        "protocol": protocol,
        "status_code": rec["status"],
        "response_size": int(rec["bytes"]) if rec["bytes"].isdigit() else None,
        "referer": rec["referer"],
        "user_agent": rec["user_agent"],
    }


def parse_log(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Parse a log file.

    Returns
    -------
    parsed : DataFrame of successfully parsed records (with ``line_number`` and ``datetime``)
    failed : DataFrame of lines that could not be parsed
    summary : dict with counts used in the paper
    """
    parsed, failed = [], []
    total = empty = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line_number, raw in enumerate(fh, start=1):
            total += 1
            line = raw.strip()
            if not line:
                empty += 1
                continue
            rec = parse_line(line)
            if rec is None:
                failed.append({"line_number": line_number, "raw_log": line})
            else:
                rec["line_number"] = line_number
                parsed.append(rec)

    df = pd.DataFrame(parsed)
    df["datetime"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT, errors="coerce")
    # Records without the attributes needed for modelling are removed.
    required = ["http_method", "url", "status_code", "user_agent", "datetime"]
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    summary = {
        "raw_file_bytes": Path(path).stat().st_size,
        "total_response_bytes": int(df["response_size"].fillna(0).sum()),
        "unique_referer": int(df["referer"].nunique()),
        "total_lines": total,
        "empty_lines": empty,
        "parsed_records": before,
        "failed_lines": len(failed),
        "dropped_missing_required": before - len(df),
        "final_records": len(df),
        "start": str(df["datetime"].min()),
        "end": str(df["datetime"].max()),
        "unique_ip": int(df["ip"].nunique()),
        "unique_url": int(df["url"].nunique()),
        "unique_user_agent": int(df["user_agent"].nunique()),
        "duplicate_records": int(df.drop(columns=["line_number"]).duplicated().sum()),
    }
    return df, pd.DataFrame(failed, columns=["line_number", "raw_log"]), summary
