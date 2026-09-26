"""Verify the exact bytes used in the benchmark before loading the model."""
import hashlib
import json
from pathlib import Path


def verify_checkpoint(directory):
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'benchmarks/model-manifest.json').read_text())
    for entry in manifest['files']:
        path = Path(directory) / entry['path']
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise ValueError(f'Checkpoint file missing or wrong size: {path}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != entry['sha256']:
            raise ValueError(f'Checkpoint SHA-256 mismatch: {path}')
    return manifest
