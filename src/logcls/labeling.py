"""Transparent heuristic labeling of HTTP access-log records.

Labels are *indicators for review*, not verified incidents. The HTTP status code is
deliberately NOT used by any rule, so that the status-code feature used by the
context-aware model is not part of the label definition.

Priority: (1) suspicious endpoint, (2) scanner/exploit User-Agent,
(3) known monitoring/crawler User-Agent, (4) unknown HTTP method -> Review,
(5) everything else -> Normal_candidate.
"""
from __future__ import annotations

import re
from urllib.parse import unquote

import pandas as pd

LABELS = ["Monitoring_Crawler", "Normal_candidate", "Suspicious"]

# (rule_id, description, regex) applied to the lower-cased, URL-decoded request target.
SUSPICIOUS_URL_RULES = [
    ("U1", "Environment/config file probe (.env, .git, .svn, wp-config)",
     r"/\.env(?:[./~_-]|$)|/\.git(?:/|$)|/\.svn(?:/|$)|wp-config"),
    ("U2", "WordPress probing (wp-login, wp-admin, xmlrpc, wlwmanifest, REST batch)",
     r"wp-login\.php|/wp-admin|xmlrpc\.php|wlwmanifest\.xml|wp-json/batch|rest_route=/batch"),
    ("U3", "PHP information disclosure probe (phpinfo, info.php, test.php, ?=PHP...)",
     r"phpinfo|/(?:info|php_info|phpversion|server-info|version|test|p|pi|i|php)\.(?:php|txt)$|\?=php[a-f0-9]{8}-"),
    ("U4", "Dependency/build file probe (composer, phpunit)",
     r"composer\.(?:json|lock)|installed\.json|phpunit"),
    ("U5", "Joomla JCE exploit probe",
     r"com_jce|/jce\.xml"),
    ("U6", "Administration panel probe (/admin, /administrator, phpMyAdmin)",
     r"/administrator(?:/|$)|/admin(?:/|\.php|$)|phpmyadmin"),
]

# Standardised discovery paths that legitimate clients request; never treated as suspicious.
BENIGN_WELL_KNOWN = r"/\.well-known/(?:passkey-endpoints|traffic-advice|security\.txt|change-password)"

SCANNER_UA_RULE = ("A1", "Scanner/exploit tool User-Agent",
                   r"php2shell|cve-\d{4}|sqlmap|nikto|nuclei|zgrab|masscan|wpscan|acunetix|nmap")

CRAWLER_UA_RULE = ("C1", "Monitoring, search-engine, AI, social-preview or headless crawler User-Agent",
                   r"bot|crawler|spider|statuscake|uptime|headlesschrome|externalhit|whatsapp|"
                   r"prefetch proxy|hostlifecycle|autodetect|slurp|preview")

KNOWN_METHODS = {"GET", "POST", "HEAD", "OPTIONS", "PUT", "DELETE", "PATCH"}


def _norm_url(url: str) -> str:
    return unquote(str(url)).lower()


def label_record(url: str, user_agent: str, method: str) -> tuple[str, str]:
    """Return (label, rule_id) for one request."""
    u = _norm_url(url)
    ua = str(user_agent).lower()
    if not re.search(BENIGN_WELL_KNOWN, u):
        for rule_id, _, pattern in SUSPICIOUS_URL_RULES:
            if re.search(pattern, u):
                return "Suspicious", rule_id
    if re.search(SCANNER_UA_RULE[2], ua):
        return "Suspicious", SCANNER_UA_RULE[0]
    if re.search(CRAWLER_UA_RULE[2], ua):
        return "Monitoring_Crawler", CRAWLER_UA_RULE[0]
    if str(method).upper() not in KNOWN_METHODS:
        return "Review", "M1"
    return "Normal_candidate", "N1"


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    res = [label_record(u, a, m) for u, a, m in zip(out["url"], out["user_agent"], out["http_method"])]
    out["activity_label"] = [r[0] for r in res]
    out["label_rule"] = [r[1] for r in res]
    return out


def rules_table() -> pd.DataFrame:
    rows = [(r, d, "Suspicious") for r, d, _ in SUSPICIOUS_URL_RULES]
    rows += [(SCANNER_UA_RULE[0], SCANNER_UA_RULE[1], "Suspicious"),
             (CRAWLER_UA_RULE[0], CRAWLER_UA_RULE[1], "Monitoring_Crawler"),
             ("M1", "HTTP method outside GET/POST/HEAD/OPTIONS/PUT/DELETE/PATCH", "Review (excluded)"),
             ("N1", "No rule matched", "Normal_candidate")]
    return pd.DataFrame(rows, columns=["rule", "description", "label"])
