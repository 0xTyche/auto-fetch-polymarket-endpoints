# auto-fetch-polymarket-endpoints

可以获取 polymarket 的所有 endpoint 并存储在 out 文件夹下，并且整理成适合 vibe coding 的文档。

Fetch Polymarket-related HTTP endpoints and write them to files you can keep in a project.

## What it does

This script collects endpoints from:
- Polymarket docs index: `https://docs.polymarket.com/llms.txt`
- Polymarket website JS bundles (what the site actually calls): `https://polymarket.com`

## Requirements

- Python 3 (no third-party dependencies)
- Network access to Polymarket domains

## Run

```bash
python3 fetch_polymarket_endpoints.py
```

Common options:

```bash
# add extra pages to scan (repeatable)
python3 fetch_polymarket_endpoints.py --seed-url "https://docs.polymarket.com/quickstart.md"

# change output directory
python3 fetch_polymarket_endpoints.py --out-dir out

# disable cache + diff
python3 fetch_polymarket_endpoints.py --no-cache
```

## Output

Generated in `out/` (by default):
- `polymarket_endpoints.md`: grouped by host, with a source URL for each entry
- `polymarket_endpoints.json`: structured data (same content)
- `polymarket_endpoints_diff.md`: added/removed endpoints vs last run (if cache enabled)

Cache (by default): `.cache/seen_endpoint_keys.json`