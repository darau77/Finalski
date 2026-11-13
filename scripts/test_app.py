"""Exercise real bundled-image classification through Streamlit AppTest."""
from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'project'))
from streamlit.testing.v1 import AppTest
at=AppTest.from_file(str(R/'project/app.py')).run(timeout=90)
assert not at.exception
at.checkbox[0].check().run();checks=[]
registry=json.loads((R/'project/models/registry.json').read_text())
for name in registry:
 at.selectbox[0].set_value(name).run(timeout=90)
 at.button[0].click().run(timeout=90)
 assert not at.exception,(name,str(at.exception))
 assert len(at.success)==1,name
 checks.append({'model':name,'result':at.success[0].value})
(R/'verification/app_checks.json').write_text(json.dumps({'method':'Streamlit AppTest, bundled example and real Classify button for each model','checks':checks},indent=2));print('PASS',len(checks),'classification paths')
