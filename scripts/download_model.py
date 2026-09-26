"""Download exactly one immutable SANA checkpoint, excluding duplicate fp16 files."""
import argparse
from huggingface_hub import snapshot_download
from nanodiffusion.models.sana import MODEL_ID, REVISION

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default='models/Sana_600M_512px_diffusers')
args = parser.parse_args()
print(snapshot_download(MODEL_ID, revision=REVISION, local_dir=args.output,
                        ignore_patterns=['*.fp16*', '.gitattributes']))
