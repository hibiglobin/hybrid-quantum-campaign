#!/usr/bin/env python3
"""Verify complete SHA-256 manifest; run after extracting ZIP."""
from pathlib import Path
import hashlib,sys
R=Path(__file__).resolve().parent
manifest=R/'MANIFEST_SHA256.txt'
files=set()
for line in manifest.read_text().splitlines():
 if not line.strip():continue
 expected,rel=line.split('  ',1)
 if rel in files:raise SystemExit(f'Duplicate manifest entry: {rel}')
 files.add(rel)
 path=R/rel
 if not path.is_file():raise SystemExit(f'MISSING: {rel}')
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 if digest!=expected:raise SystemExit(f'SHA256 MISMATCH: {rel}')
actual={str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and p.name!='MANIFEST_SHA256.txt' and '__pycache__' not in p.parts and not p.name.endswith('.pyc')}
if actual!=files:raise SystemExit(f'Manifest coverage mismatch: missing={actual-files}, stale={files-actual}')
print(f'PASS: SHA-256 verified for all {len(files)} package files')
