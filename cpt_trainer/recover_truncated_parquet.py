"""Recover the `text` column of a parquet file whose footer was never written (copied while the writer was
still running, or the writer was never closed). pyarrow refuses such a file ("Parquet magic bytes not
found in footer"), but every finished row group is still in it.

    python -m cpt_trainer.recover_truncated_parquet validate GOOD.parquet          # self-test on a readable file
    python -m cpt_trainer.recover_truncated_parquet recover BROKEN.parquet OUT.parquet

Walks the pages from byte 4, parsing each thrift-compact PageHeader; decodes DICTIONARY pages and PLAIN
data pages of BYTE_ARRAY columns and keeps the ones whose strings look like document text (long,
not URL / id like). Row order and the other columns are not recovered; OUT has `id` (sha1) and `text`.
Used on cpt_data/finewebedu_replay_small.parquet (2026-10-18): 66.6 MB, 8,647 row groups, 9,379 texts.
"""
import struct, sys, hashlib, collections
import pyarrow as pa, pyarrow.parquet as pq

class Thrift:
    def __init__(self, buf, pos):
        self.b, self.p = buf, pos
    def byte(self):
        v = self.b[self.p]; self.p += 1; return v
    def varint(self):
        r = s = 0
        while True:
            x = self.byte(); r |= (x & 0x7F) << s; s += 7
            if not x & 0x80: return r
    def zig(self):
        n = self.varint(); return (n >> 1) ^ -(n & 1)
    def skip(self, t):
        if t in (1, 2): return
        if t == 3: self.p += 1
        elif t in (4, 5, 6): self.varint()
        elif t == 7: self.p += 8
        elif t == 8: n = self.varint(); self.p += n
        elif t in (9, 10):
            h = self.byte(); n, et = h >> 4, h & 15
            if n == 15: n = self.varint()
            for _ in range(n): self.skip(et)
        elif t == 11:
            n = self.varint()
            if n:
                h = self.byte()
                for _ in range(n): self.skip(h >> 4); self.skip(h & 15)
        elif t == 12: self.struct()
        else: raise ValueError(f"bad thrift type {t}")
    def struct(self, want=None):
        """Parse a struct; return {field_id: value} for i32/bool fields in `want` (others skipped)."""
        out, fid = {}, 0
        while True:
            h = self.byte()
            if h == 0: return out
            t, d = h & 15, h >> 4
            fid = fid + d if d else self.zig()  # i16 zigzag
            if want is not None and fid in want and t in (1, 2, 3, 4, 5, 6):
                out[fid] = True if t == 1 else False if t == 2 else self.byte() if t == 3 else self.zig()
            elif want is not None and fid in want and t == 12:
                out[fid] = self.struct(want[fid])
            else:
                self.skip(t)

PAGE_WANT = {1: None, 2: None, 3: None, 5: {1: None, 2: None, 3: None}, 7: {1: None, 2: None}, 8: {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None}}

def read_header(buf, pos):
    th = Thrift(buf, pos)
    h = th.struct(PAGE_WANT)
    return h, th.p

def decode_plain_byte_arrays(data, max_vals):
    vals, p, n = [], 0, len(data)
    while p < n and len(vals) < max_vals:
        if p + 4 > n: return None
        L = struct.unpack_from("<I", data, p)[0]; p += 4
        if p + L > n: return None
        vals.append(data[p:p + L]); p += L
    return vals if p == n else None

def looks_like_text(vals):
    if len(vals) == 0: return False
    try:
        s = [v.decode("utf-8") for v in vals[:50]]
    except UnicodeDecodeError:
        return False
    mean = sum(len(v) for v in vals) / len(vals)
    urlish = sum(x.startswith(("http://", "https://", "s3://", "<urn:", "CC-MAIN")) for x in s) / len(s)
    return mean > 150 and urlish < 0.2 and all(len(x) > 0 for x in s)

def recover(path, max_pages=None, verbose=False):
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[:4] == b"PAR1"
    pos, pages, codec = 4, 0, None
    texts, kinds = [], collections.Counter()
    dict_lens = collections.Counter()
    while pos < len(buf) - 8:
        try:
            h, body = read_header(buf, pos)
        except (IndexError, ValueError) as e:
            print(f"  header parse failed at {pos} ({len(buf) - pos} bytes left): {e}"); break
        ptype, usz, csz = h.get(1), h.get(2), h.get(3)
        if ptype not in (0, 1, 2, 3) or usz is None or csz is None or body + csz > len(buf) or usz < 0 or csz < 0:
            print(f"  implausible header at {pos}: {h}"); break
        raw = buf[body:body + csz]
        def decomp(raw, usz):
            nonlocal codec
            if codec is None:
                for c in ("snappy", "zstd", "gzip", "lz4", "brotli"):
                    try:
                        out = pa.decompress(raw, decompressed_size=usz, codec=c, asbytes=True)
                        if len(out) == usz: codec = c; print(f"  codec detected: {c}"); return out
                    except Exception: pass
                if len(raw) == usz: codec = "none"; print("  codec detected: uncompressed"); return bytes(raw)
                return None
            if codec == "none": return bytes(raw) if len(raw) == usz else None
            try: return pa.decompress(raw, decompressed_size=usz, codec=codec, asbytes=True)
            except Exception: return None
        if ptype == 2:  # dictionary page
            d = decomp(raw, usz)
            nv = (h.get(7) or {}).get(1, 0)
            vals = decode_plain_byte_arrays(d, nv) if d is not None else None
            if vals is not None and len(vals) == nv and looks_like_text(vals):
                texts.extend(vals); kinds["text_dict"] += 1; dict_lens[nv] += 1
            else:
                kinds["other_dict"] += 1
        elif ptype == 0:  # data page v1
            dp = h.get(5) or {}
            enc, nv = dp.get(2), dp.get(1, 0)
            if enc == 0:  # PLAIN values -> maybe text that spilled past the dictionary limit
                d = decomp(raw, usz)
                if d is not None and len(d) >= 4:
                    L = struct.unpack_from("<I", d, 0)[0]  # definition levels
                    vals = decode_plain_byte_arrays(d[4 + L:], nv) if 4 + L <= len(d) else None
                    if vals is not None and looks_like_text(vals):
                        texts.extend(vals); kinds["text_plain_page"] += 1
                    else:
                        kinds["other_plain_page"] += 1
            else:
                kinds[f"data_enc{enc}"] += 1
        elif ptype == 3:
            dp = h.get(8) or {}
            enc, nv, dl, rl = dp.get(4), dp.get(1, 0), dp.get(5, 0), dp.get(6, 0)
            if enc == 0:
                vraw = raw[dl + rl:]
                d = decomp(vraw, usz - dl - rl) if dp.get(7, True) else bytes(vraw)
                vals = decode_plain_byte_arrays(d, nv) if d is not None else None
                if vals is not None and looks_like_text(vals):
                    texts.extend(vals); kinds["text_plain_page_v2"] += 1
                else:
                    kinds["other_v2"] += 1
            else:
                kinds[f"v2_enc{enc}"] += 1
        else:
            kinds["index"] += 1
        if verbose and pages < 20:
            print(f"  page {pages} @ {pos}: type {ptype} usz {usz} csz {csz} hdr {h}")
        pos = body + csz; pages += 1
        if max_pages and pages >= max_pages: break
    print(f"  {pages} pages parsed, stopped at byte {pos} of {len(buf)} ({len(buf) - pos} bytes unparsed); kinds {dict(kinds)}")
    print(f"  text-dictionary sizes (rows per row group): {sorted(dict_lens.items())[:8]} ... {len(dict_lens)} distinct")
    return texts

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "validate":
        p = sys.argv[2]
        pf = pq.ParquetFile(p)
        truth = []
        for g in range(5):
            truth += pf.read_row_group(g, columns=["text"])["text"].to_pylist()
        print(f"validating on {p}: first 5 row groups = {len(truth)} rows")
        texts = recover(p, max_pages=5 * 40, verbose=True)
        rec = [t.decode("utf-8") for t in texts]
        print(f"  recovered {len(rec)} texts; first-5-row-group truth covered: {len(set(truth) & set(rec))}/{len(set(truth))}; "
              f"recovered not in truth (later row groups, expected): {len(set(rec) - set(truth))}")
    else:
        p, outp = sys.argv[2], sys.argv[3]
        texts = recover(p)
        seen, rows = set(), []
        for t in texts:
            s = t.decode("utf-8")
            k = hashlib.sha1(t).hexdigest()
            if k in seen: continue
            seen.add(k); rows.append((k, s))
        print(f"  {len(texts)} texts recovered, {len(rows)} unique, {sum(len(s) for _, s in rows) / 1e6:.1f} M chars, "
              f"mean {sum(len(s) for _, s in rows) // max(1, len(rows))} chars")
        lens = sorted(len(s) for _, s in rows)
        print(f"  char length quantiles: min {lens[0]} p10 {lens[len(lens)//10]} p50 {lens[len(lens)//2]} p90 {lens[9*len(lens)//10]} max {lens[-1]}")
        tbl = pa.table({"id": [k for k, _ in rows], "text": [s for _, s in rows]})
        pq.write_table(tbl, outp, compression="zstd")
        print(f"  wrote {outp}")
