"""Randomly sample rows from every parquet file under a folder (recursively)
so that the sample is roughly a target size.

Only parquet footers are read to plan the sample (row counts), then every row
group is read once and only its picked rows are kept, so memory stays at about
one row group. The target size is measured against the on-disk size of the
input: a 500 MB target from a 20 GB folder keeps 2.5% of all rows, chosen
uniformly at random across all files.

Usage:
    python filter_randomly.py INPUT_DIR OUTPUT.parquet --size-mb 500 [--seed 0]
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", help="folder to search for *.parquet files (searched recursively)")
    parser.add_argument("output", help="output parquet file")
    parser.add_argument("--size-mb", type=float, default=500, help="target size in MB (default: 500)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    files = sorted(Path(args.input_dir).rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files found under {args.input_dir}")

    # 1. plan from metadata only: row counts come from the parquet footers, sizes from the filesystem
    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    total_bytes = sum(f.stat().st_size for f in files)
    n = min(total_rows, round(total_rows * args.size_mb * 1024 * 1024 / total_bytes))
    print(f"{len(files)} files, {total_rows} rows, {total_bytes / 2**20:.0f} MB on disk -> keeping {n} rows")

    # 2. pick n distinct global row indices (row 0 of the first file ... last row of the last file)
    rng = np.random.default_rng(args.seed)
    picked = np.sort(rng.choice(total_rows, size=n, replace=False))

    # 3. walk the files row group by row group, reading only the groups that contain picked rows
    writer = None
    offset = 0  # global index of the first row of the current row group
    for i, f in enumerate(files):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            group_rows = pf.metadata.row_group(g).num_rows
            lo, hi = np.searchsorted(picked, [offset, offset + group_rows])
            if hi > lo:
                table = pf.read_row_group(g).take(picked[lo:hi] - offset)
                if writer is None:
                    writer = pq.ParquetWriter(args.output, table.schema)
                writer.write_table(table)
            offset += group_rows
        print(f"[{i + 1}/{len(files)}] {f}")

    if writer is not None:
        writer.close()
    print(f"wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
