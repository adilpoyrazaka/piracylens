# PiracyLens

[![CI](https://github.com/adilpoyrazaka/piracylens/actions/workflows/ci.yml/badge.svg)](https://github.com/adilpoyrazaka/piracylens/actions/workflows/ci.yml)

Machine learning prototype for four anti-piracy tasks: detecting modified video copies, auditing copyright takedown requests, forecasting takedown volume, and clustering pirate domains.

> **Status:** work in progress. The data pipeline for the Google Transparency Report copyright dataset is in place. No models have been trained yet, so this README reports no results.

## Requirements

- Linux or WSL2, Python 3.12.9
- Disk: about 105 GB free on the Linux filesystem during conversion (about 90 GB of extracted CSVs plus about 14 GB of Parquet). The CSVs can be deleted after a successful conversion.
- Memory: the DuckDB steps are configured for 16 threads and a 14 GB memory limit.

## Setup

```bash
git clone git@github.com:adilpoyrazaka/piracylens.git
cd piracylens
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

## Data

### Snapshot

All figures in this repository are produced from this snapshot of the Google Transparency Report bulk export for copyright removals from Web Search:

| Field | Value |
| --- | --- |
| File | `google-websearch-copyright-removals.zip` |
| Source | https://storage.googleapis.com/transparencyreport/google-websearch-copyright-removals.zip |
| Downloaded | 2026-10-26 |
| Size | 15,739,542,261 bytes |
| SHA-256 | `6ce3aaf6a31bc834660b1d8644c5796fa42a67f910da5671fb48908659f5707d` |
| Requests covered | 2011-03-10 to 2026-10-24 |

Verify a download against this snapshot:

```bash
echo "6ce3aaf6a31bc834660b1d8644c5796fa42a67f910da5671fb48908659f5707d  google-websearch-copyright-removals.zip" | sha256sum -c -
```

Google updates this export regularly. A new download is a different snapshot and will not reproduce these figures exactly.

### Terms of use

Google publishes this dataset without an explicit data license. This repository does not redistribute it: raw files and derived Parquet files stay local (`data/` is git-ignored). Only code and aggregate figures are published.

### 1. Extract

```bash
mkdir -p data/raw data/interim data/processed
cd data/raw
unzip -q /path/to/google-websearch-copyright-removals.zip
cd ../..
ls -l data/raw/google-websearch-copyright-removals/
```

Expected file sizes (bytes):

| File | Size |
| --- | --- |
| `requests.csv` | 7,738,066,978 |
| `domains.csv` | 36,409,779,308 |
| `urls-no-action-taken.csv` | 45,524,457,061 |
| `README.txt` | 2,341 |

On the reference machine (WSL2, reading the zip from the Windows drive) extraction took about 15 minutes.

### 2. Convert CSV to Parquet

```bash
for t in requests domains urls_no_action; do python -m piracylens.data.convert_raw "$t"; done
```

Before writing, the converter checks each CSV's size and header. It writes to a temporary file and renames it only on success. Output in `data/processed/` (zstd-compressed):

| Table | One row per | Rows |
| --- | --- | --- |
| `requests.parquet` | takedown request | 57,273,319 |
| `domains.parquet` | request x domain | 963,823,791 |
| `urls_no_action.parquet` | URL with no action taken | 456,643,466 |

Reference timings (22 threads): requests 19 s, domains 2.2 min, urls 1.1 min. After a successful conversion the CSVs can be deleted; keep `README.txt`, which documents the columns.

### 3. Build owner name keys

```bash
python -m piracylens.data.build_owner_keys
```

Writes `data/interim/owner_name_keys.parquet`, which maps each raw copyright owner name to a normalized key. Normalization covers HTML entities, invisible Unicode characters, Cyrillic transliteration, legal-form suffixes and repeated tokens. On this snapshot: 1,702,749 names map to 1,544,395 keys; about 18 s on the reference machine.

### 4. Generate scope candidates

Scope is defined by a rule and applied in two stages, so the one manual decision is made before any owner-level output exists. Every threshold, reporter list and name prefix is in `configs/scope_candidates.yaml`, committed before any candidate list was produced.

```bash
python -m piracylens.data.scope_candidates agencies
```

Writes `data/interim/scope_agencies.csv`: one row per reporter group above the volume threshold, with its request count and seed share (the share of its requests that belong to the seed owners listed in the parameter file). Reporter names are normalized and merged with `configs/reporter_aliases.yaml`. The film/TV agencies are then picked by hand from this table and recorded under `film_tv_agencies.selected`. On this snapshot: 58 groups covering 71.2% of requests since 2021-01-01; about 19 s on the reference machine.

```bash
python -m piracylens.data.scope_candidates candidates
```

Writes `data/interim/scope_candidates.csv`: one row per owner key caught by any of four sources (request volume, the protection-network reporters, the selected agencies, the name-prefix scan), plus every seed owner. Columns hold volumes and evidence only: source flags, matched prefixes, name variants, and the top reporter group with its share. There is no label or outcome column, so the scope review cannot be steered by results. The command refuses to run while `selected` is empty or names a group outside the agency table. On this snapshot: 1,302 candidates; about 27 s on the reference machine.

Both modes stop before writing anything if a seed owner key, a reporter alias key or a protection-network reporter is missing from the data.

## Development

```bash
pytest -q
ruff check .
ruff format --check .
```

CI runs the same checks on every push to `main` and on pull requests. The tests do not need the dataset.

## Project layout

```
configs/            alias groups and pre-registered parameters (YAML)
src/piracylens/     package source
  data/             conversion, owner normalization, alias loading
tests/              unit tests
data/               local data, git-ignored (raw/, interim/, processed/)
```

## License

Not yet chosen.
