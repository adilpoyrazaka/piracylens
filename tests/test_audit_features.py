"""Tests for the audit feature table on synthetic data, with hand-computed expectations."""

import dataclasses
import datetime as dt
from pathlib import Path

import duckdb
import pytest

from piracylens.audit.features import INPUT_COLUMNS, OUTPUT_COLUMNS, build_features
from piracylens.audit.preregistration import load_preregistration
from piracylens.data.aliases import AliasRow

DESIGN = Path(__file__).resolve().parents[1] / "configs" / "audit_preregistration.yaml"
T = dt.datetime(2023, 6, 1, 12, 0, 0)  # filing time of the main target request (train)
DAY = dt.timedelta(days=1)

OWNERS = [
    AliasRow("studio a", "studio", "Studio A", "web", "verified", "us", "film", "us"),
    AliasRow("studio a tv", "studio", "Studio A TV", "web", "verified", "us", "tv", "us"),
    AliasRow("channel b", "channel", "Channel B", "web", "verified", "ru", "tv", "ru"),
]
REPORTERS = [
    AliasRow("agency x", "agency", "Agency X", "reporter", "verified"),
    AliasRow("agency x llc", "agency", "Agency X LLC", "reporter", "verified"),
]


def _request(rid, when, owner_group, reporter_key, reporter_group, market="us"):
    return {
        "request_id": rid,
        "requested_at_utc": when,
        "owner_group": owner_group,
        "market": market,
        "content_type": "film",
        "region": "us",
        "reporter_key": reporter_key,
        "reporter_group": reporter_group,
    }


def _row(rid, domain, removed=0, not_in_index=0, no_action=0, pending=0):
    return (rid, domain, removed, not_in_index, no_action, pending)


# Main target request 1 at T, with three rows: two decided, one only not-in-index.
SCOPED = [
    _request(1, T, "studio", "studio a tv", "studio a tv"),  # self-filed through another brand
    _request(2, T + DAY, "channel", "agency x llc", "agency"),
    _request(3, dt.datetime(2025, 3, 1), "channel", "studio a", "studio a"),  # other group's key
    _request(9, dt.datetime(2020, 1, 1), "studio", "agency x", "agency"),  # history only
]
SCOPED_ROWS = [
    _row(1, " X.com ", removed=2, no_action=1),
    _row(1, "y.com", removed=1),
    _row(1, "z.com", not_in_index=4),
    _row(2, "Sub.Ex-ample1.xn--p1ai", no_action=3),
    _row(3, "x.com", removed=1, no_action=1),
    _row(9, "x.com", removed=1),
]
# Full-table history from every owner (request ids 100+ are out of scope).
HISTORY = [
    (100, T - 45 * DAY, [_row(100, "x.com", no_action=2)]),  # decided, label 1, counts
    (101, T - 20 * DAY, [_row(101, "x.com", removed=1)]),  # decided, too recent
    (102, T - 30 * DAY, [_row(102, "y.com", no_action=1)]),  # exactly at the gap: counts
    (103, T - 30 * DAY + dt.timedelta(seconds=1), [_row(103, "y.com", removed=1)]),  # too recent
    (104, T - 400 * DAY, [_row(104, "y.com", not_in_index=1)]),  # undecided: age only
    (105, T, [_row(105, "z.com", removed=1)]),  # same timestamp: not earlier
    (106, dt.datetime(2018, 1, 1), [_row(106, "x.com", no_action=1)]),  # before history start
]


_INSERT_D = (
    "INSERT INTO d (request_id, domain, urls_removed, urls_not_in_index, urls_no_action, "
    "urls_pending) VALUES (?, ?, ?, ?, ?, ?)"
)


def _write(tmp: Path, history=HISTORY, scoped_rows=SCOPED_ROWS) -> dict[str, Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    p = {
        "sr": tmp / "scoped_requests.parquet",
        "sd": tmp / "scoped_domains.parquet",
        "dom": tmp / "domains.parquet",
        "req": tmp / "requests.parquet",
    }
    times = {r["request_id"]: r["requested_at_utc"] for r in SCOPED}
    window = dt.datetime(2021, 1, 1)
    con.execute(
        "CREATE TABLE sr (request_id BIGINT, requested_at_utc TIMESTAMP, in_window BOOLEAN, "
        "owner_group VARCHAR, market VARCHAR, content_type VARCHAR, region VARCHAR, "
        "reporter_key VARCHAR, reporter_group VARCHAR)"
    ).fetchall()
    for r in SCOPED:
        con.execute(
            "INSERT INTO sr VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                r["request_id"],
                r["requested_at_utc"],
                r["requested_at_utc"] >= window,
                r["owner_group"],
                r["market"],
                r["content_type"],
                r["region"],
                r["reporter_key"],
                r["reporter_group"],
            ],
        ).fetchall()
    con.execute(
        "CREATE TABLE d (request_id BIGINT, domain VARCHAR, urls_removed BIGINT, "
        "urls_not_in_index BIGINT, urls_no_action BIGINT, urls_pending BIGINT, "
        "from_abuser BOOLEAN DEFAULT false)"
    ).fetchall()
    # Same columns as the real requests table, so ambiguous column names fail here too.
    con.execute(
        "CREATE TABLE q (request_id BIGINT, requested_at_utc TIMESTAMP, owner_name VARCHAR, "
        "reporter_name VARCHAR, urls_removed BIGINT DEFAULT 0, urls_not_in_index BIGINT "
        "DEFAULT 0, urls_no_action BIGINT DEFAULT 0, urls_pending BIGINT DEFAULT 0, "
        "from_abuser BOOLEAN DEFAULT false)"
    ).fetchall()
    for row in scoped_rows:
        con.execute(_INSERT_D, list(row)).fetchall()
    for rid, when in times.items():
        con.execute("INSERT INTO q (request_id, requested_at_utc) VALUES (?, ?)", [rid, when])
    for rid, when, rows in history:
        con.execute("INSERT INTO q (request_id, requested_at_utc) VALUES (?, ?)", [rid, when])
        for row in rows:
            con.execute(_INSERT_D, list(row)).fetchall()
    con.execute(f"COPY sr TO '{p['sr']}' (FORMAT parquet)").fetchall()
    con.execute(
        f"COPY (SELECT d.* EXCLUDE (from_abuser), q.requested_at_utc, "
        f"q.requested_at_utc >= TIMESTAMP '2021-01-01' "
        f"AS in_window FROM d JOIN q USING (request_id) "
        f"WHERE request_id IN (SELECT request_id FROM sr)) TO '{p['sd']}' (FORMAT parquet)"
    ).fetchall()
    con.execute(f"COPY d TO '{p['dom']}' (FORMAT parquet)").fetchall()
    con.execute(f"COPY q TO '{p['req']}' (FORMAT parquet)").fetchall()
    return p


def _build(tmp: Path, **kw) -> dict[tuple, dict]:
    p = _write(tmp, **kw)
    out = tmp / "features.parquet"
    build_features(
        load_preregistration(DESIGN),
        OWNERS,
        REPORTERS,
        p["sr"],
        p["sd"],
        p["dom"],
        p["req"],
        out,
        threads=2,
        memory_limit="1GB",
    )
    con = duckdb.connect()
    cols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{out}')").fetchall()]
    assert tuple(cols) == OUTPUT_COLUMNS
    rows = con.execute(f"SELECT * FROM read_parquet('{out}')").fetchall()
    return {(r[0], r[1]): dict(zip(cols, r, strict=True)) for r in rows}


@pytest.fixture
def feats(tmp_path):
    return _build(tmp_path)


def test_only_decided_window_rows_are_targets(feats):
    assert set(feats) == {(1, "x.com"), (1, "y.com"), (2, "sub.ex-ample1.xn--p1ai"), (3, "x.com")}
    assert feats[(1, "x.com")]["split"] == "train"
    assert feats[(3, "x.com")]["split"] == "validation"


def test_past_rate_uses_only_rows_at_least_the_gap_earlier(feats):
    x = feats[(1, "x.com")]
    # Counts request 100 (45 days, label 1) and the 2020 scoped request 9 (label 0);
    # request 101 (20 days) is too recent, request 106 is before the history start.
    assert x["domain_past_requests"] == 2 and x["domain_past_rate"] == pytest.approx(0.5)
    assert x["has_domain_history"] is True
    # Exactly 30 days earlier counts; one second later does not; undecided rows never count.
    y = feats[(1, "y.com")]
    assert y["domain_past_requests"] == 1 and y["domain_past_rate"] == pytest.approx(1.0)


def test_row_without_history_has_missing_rate_and_flag(feats):
    s = feats[(2, "sub.ex-ample1.xn--p1ai")]
    assert s["domain_past_rate"] is None and s["has_domain_history"] is False
    assert s["domain_past_requests"] == 0


def test_later_rows_of_the_same_domain_see_earlier_targets(feats):
    # Request 3 (2025-03-01) sees requests 9, 100, 101 and 1 on x.com; labels 0, 1, 0, 1.
    later = feats[(3, "x.com")]
    assert later["domain_past_requests"] == 4 and later["domain_past_rate"] == pytest.approx(0.5)


def test_changing_anything_after_the_cutoff_leaves_features_unchanged(tmp_path):
    base = _build(tmp_path / "a")
    flipped_history = [
        (rid, when, [_row(r[0], r[1], no_action=0 if r[4] else 5) for r in rows])
        if when > T - 30 * DAY
        else (rid, when, rows)
        for rid, when, rows in HISTORY
    ]
    flipped_scoped = [
        _row(r[0], r[1], removed=1, no_action=0 if r[4] else 1) if r[0] in (2, 3) else r
        for r in SCOPED_ROWS
    ]
    changed = _build(tmp_path / "b", history=flipped_history, scoped_rows=flipped_scoped)
    for col in INPUT_COLUMNS:
        for key in [(1, "x.com"), (1, "y.com")]:
            assert base[key][col] == changed[key][col], (key, col)


def test_request_size_counts_every_row(feats):
    x = feats[(1, "x.com")]
    assert x["domains_in_request"] == 3  # includes the not-in-index-only row
    assert x["urls_in_request"] == 3 + 1 + 4
    assert x["urls_for_domain"] == 3


def test_domain_age_uses_strictly_earlier_appearances(feats):
    assert feats[(1, "y.com")]["domain_age_days"] == pytest.approx(400.0)  # undecided row counts
    assert feats[(1, "x.com")]["has_domain_age"] is True
    # Earliest x.com in history is 2020-01-01 (2018 is before the history start).
    expected = (T - dt.datetime(2020, 1, 1)).total_seconds() / 86400
    assert feats[(1, "x.com")]["domain_age_days"] == pytest.approx(expected)
    s = feats[(2, "sub.ex-ample1.xn--p1ai")]
    assert s["has_domain_age"] is False and s["domain_age_days"] is None


def test_same_timestamp_does_not_count_as_earlier(tmp_path):
    # z.com appears only at T: in request 1 (made decided here) and in request 105.
    rows = [_row(1, "z.com", removed=1) if r[1] == "z.com" else r for r in SCOPED_ROWS]
    rows.append(_row(2, "z.com", removed=1))
    feats = _build(tmp_path, scoped_rows=rows)
    first = feats[(1, "z.com")]
    assert first["has_domain_age"] is False and first["domain_age_days"] is None
    later = feats[(2, "z.com")]  # one day after T
    assert later["has_domain_age"] is True and later["domain_age_days"] == pytest.approx(1.0)
    assert later["has_domain_history"] is False  # decided only one day earlier


def test_domain_shape(feats):
    s = feats[(2, "sub.ex-ample1.xn--p1ai")]
    assert s["tld"] == "xn--p1ai"
    assert s["domain_length"] == len("sub.ex-ample1.xn--p1ai")
    assert s["domain_labels"] == 3
    assert s["domain_hyphens"] == 3
    assert s["domain_digit_share"] == pytest.approx(2 / 22)
    assert s["domain_punycode"] is True
    assert feats[(1, "x.com")]["tld"] == "com" and feats[(1, "x.com")]["domain_punycode"] is False


def test_owner_and_reporter_sides(feats):
    x = feats[(1, "x.com")]
    assert x["self_filed"] is True  # reporter is another brand key of the same owner group
    assert x["reporter_brand"] == "studio a tv"  # not an agency alias: the key itself
    s = feats[(2, "sub.ex-ample1.xn--p1ai")]
    assert s["self_filed"] is False and s["reporter_brand"] == "Agency X LLC"
    assert s["owner_group"] == "channel" and s["market"] == "us"
    assert feats[(3, "x.com")]["self_filed"] is False  # a key of a different group


def test_labels(feats):
    assert feats[(1, "x.com")]["label_primary"] is True
    assert feats[(1, "x.com")]["label_sensitivity"] is False  # 1 of 3 decided URLs
    assert feats[(2, "sub.ex-ample1.xn--p1ai")]["label_sensitivity"] is True
    assert feats[(1, "y.com")]["label_primary"] is False


def test_dry_run_writes_nothing(tmp_path, capsys):
    p = _write(tmp_path)
    out = tmp_path / "features.parquet"
    n = build_features(
        load_preregistration(DESIGN),
        OWNERS,
        REPORTERS,
        p["sr"],
        p["sd"],
        p["dom"],
        p["req"],
        out,
        dry_run=True,
        threads=2,
        memory_limit="1GB",
    )
    assert n == 4 and not out.exists()
    assert "dry run: nothing written" in capsys.readouterr().out


def test_design_that_differs_from_the_builder_stops(tmp_path):
    p = load_preregistration(DESIGN)
    fewer = dataclasses.replace(p, inputs=p.inputs[:-1])
    with pytest.raises(SystemExit, match="design inputs differ"):
        build_features(fewer, OWNERS, REPORTERS, *[tmp_path / "x"] * 5)
    other_rule = dataclasses.replace(p, tld_rule="public_suffix")
    with pytest.raises(SystemExit, match="not implemented"):
        build_features(other_rule, OWNERS, REPORTERS, *[tmp_path / "x"] * 5)
