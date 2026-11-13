"""Check new prediction evidence, frozen hashes and the real demo controls."""
from run import *
from streamlit.testing.v1 import AppTest
R=Path(__file__).resolve().parent;D=R/'results'
manifest=pd.read_csv(ROOT/'project/results/split_manifest.csv');test=manifest[manifest.split=='test'].sort_values('path');classes=sorted(manifest.label.unique());y=np.array([classes.index(x) for x in test.label]);table=pd.read_csv(D/'comparison.csv');checks=[]
for row in table.itertuples():
 p=pd.read_csv(D/(row.name+'_predictions.csv'));pr=np.load(D/(row.name+'_probabilities.npy'))
 assert p.path.tolist()==test.path.tolist() and p.true_label.tolist()==test.label.tolist()
 assert pr.shape==(584,7) and np.isfinite(pr).all() and np.allclose(pr.sum(1),1,atol=1e-5)
 assert p.predicted_label.tolist()==[classes[i] for i in pr.argmax(1)]
 m=metrics(y,pr)
 assert abs(m['accuracy']-row.accuracy)<1e-12 and abs(m['macro_f1']-row.macro_f1)<1e-12
 checks.append({'name':row.name,'predictions_and_metrics':'pass'})
freeze=json.loads((D/'selection_frozen_before_test.json').read_text());assert freeze['test_loaded'] is False
assert sha(D/'mobilenet_head.joblib')==freeze['mobile_head_sha256'];assert sha(D/'mobilenet_backbone.pt')==freeze['mobile_backbone_sha256']
for c in json.loads((D/'cnn_validation.json').read_text())['trials']:assert sha(D/(c['name']+'.pt'))==c['checkpoint_sha256']
assert max(freeze['candidates'],key=lambda c:c['validation']['macro_f1'])['name']==freeze['winner']['name']
at=AppTest.from_file(str(R/'app.py')).run(timeout=90);assert not at.exception
for name in table.name:
 at.selectbox[0].set_value(name).run(timeout=90);at.button[0].click().run(timeout=90);assert not at.exception,(name,at.exception);assert len(at.success)==1
save(R/'verification.json',{'prediction_checks':checks,'frozen_hashes_verified':True,'selection_rule_verified':True,'demo_classification_paths_passed':len(table),'live_browser_verified':False})
print('PASS',len(table),'prediction sets and demo paths')
