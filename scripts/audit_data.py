"""Audit actual source pixels and source-linked augmentation metadata."""
from pathlib import Path
import argparse,csv,json,hashlib
from collections import Counter,defaultdict
from PIL import Image
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
rows=list(csv.DictReader((R/'project/results/split_manifest.csv').open()))
for row in rows:
 with Image.open(a.dataset/row['path']) as im:
  im=im.convert('RGB');actual=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
 assert actual==row['pixel_hash'],row['path']
assert len({r['pixel_hash'] for r in rows})==3893
aug=list(csv.DictReader((R/'data_preparation/manifest.csv').open()))
assert len(aug)==17517
assert len({r['output_pixel_hash'] for r in aug})==17517
splits=defaultdict(set)
for r in aug:
 splits[r['source_id']].add(r['split'])
 if r['augmentation']!='original':assert r['split']=='train'
assert len(splits)==3893 and all(len(s)==1 for s in splits.values())
source={r['pixel_hash']:r['split'] for r in rows}
assert all(next(iter(v))==source[k] for k,v in splits.items())
res={'source_pixels_verified':len(rows),'unique_source_hashes':3893,'augmentation_records_checked':len(aug),'augmentation_counts':dict(Counter(r['augmentation'] for r in aug)),'split_counts':dict(Counter(r['split'] for r in aug)),'augmentation_bytes_checked':False,'note':'Actual original pixels verified. Stored derivative metadata checked; image-generation audit is not a model-effect experiment.'}
(R/'verification/data_audit.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
