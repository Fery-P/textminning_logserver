import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logcls.labeling import label_record  # noqa: E402
from logcls.parsing import parse_line  # noqa: E402

LINE = ('1.2.3.4 - - [22/Oct/2026:15:54:59 +0700] "GET /.env HTTP/2" 403 199 "-" '
        '"Mozilla/5.0 (Windows NT 10.0; Win64; x64)"')


def test_parse_line():
    rec = parse_line(LINE)
    assert rec["http_method"] == "GET" and rec["url"] == "/.env" and rec["status_code"] == "403"


def test_parse_invalid():
    assert parse_line("not a log line") is None


def test_labels():
    assert label_record("/.env.backup", "Mozilla/5.0", "GET")[0] == "Suspicious"
    assert label_record("//blog/wp-includes/wlwmanifest.xml", "Mozilla/5.0", "GET")[0] == "Suspicious"
    assert label_record("/", "php2shell-scan/2.0", "HEAD")[0] == "Suspicious"
    assert label_record("/", "StatusCake_Pagespeed_Indev", "GET")[0] == "Monitoring_Crawler"
    assert label_record("/.well-known/passkey-endpoints", "Mozilla/5.0", "GET")[0] == "Normal_candidate"
    assert label_record("/berita", "Mozilla/5.0 (iPhone)", "GET")[0] == "Normal_candidate"
    assert label_record("/", "Mozilla/5.0", "PROPFIND")[0] == "Review"
