"""Verify saved artifacts and optionally rerun inference without training.

Usage: python verify_final.py --dataset /path/to/BD3_original_dataset
"""
from pathlib import Path
import argparse, json, hashlib, sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path)
    p.add_argument('--output', type=Path, default=ROOT.parent/'verification/checkpoints.json')
    a = p.parse_args()
    manifest = pd.read_csv(ROOT/'results/split_manifest.csv')
    assert len(manifest) == 3893 and manifest.path.is_unique
    assert manifest.pixel_hash.is_unique, 'Duplicate image content'
    assert manifest.groupby('split').size().to_dict() == {'test':584,'train':2725,'validation':584}
    truth = manifest[manifest.split == 'test'].set_index('path').label.sort_index()
    table = pd.read_csv(ROOT/'results/model_comparison.csv').set_index('Model')
    registry = json.loads((ROOT/'models/registry.json').read_text())
    classical = pd.read_csv(ROOT/'results/test_predictions.csv')
    evidence = {'split_counts': manifest.groupby('split').size().to_dict(), 'models':{}}
    for name, info in registry.items():
        kind = info['type']
        preds = (classical[classical.model == name] if kind == 'classical' else
                 pd.read_csv(ROOT/info.get('prediction_file', f'results/{kind}_test_predictions.csv')))
        assert preds.path.is_unique
        preds = preds.set_index('path').sort_index()
        assert preds.index.equals(truth.index)
        assert np.array_equal(preds.true_label, truth)
        acc = accuracy_score(truth,preds.predicted_label)
        f1 = f1_score(truth,preds.predicted_label,average='macro')
        assert np.isclose(acc,table.loc[name,'Accuracy'])
        assert np.isclose(f1,table.loc[name,'F1 macro'])
        assert np.isclose(acc,info['test_accuracy'])
        entry = {'accuracy':acc,'macro_f1':f1,'test_membership_verified':True,
                 'checkpoint_sha256':hashlib.sha256((ROOT/info['file']).read_bytes()).hexdigest()}
        if a.dataset:
            from PIL import Image
            if kind == 'classical':
                import joblib
                from features import extract_features
                model = joblib.load(ROOT/info['file'])
                X = []
                for path in truth.index:
                    with Image.open(a.dataset/path) as im: X.append(extract_features(im))
                actual = model.predict(np.asarray(X))
            else:
                import torch, importlib
                module = importlib.import_module(kind+'_model')
                torch.set_num_threads(4)
                model = module.build_model().eval()
                model.load_state_dict(torch.load(ROOT/info['file'],map_location='cpu',weights_only=True))
                xs=[]
                for path in truth.index:
                    with Image.open(a.dataset/path) as im: xs.append(module.prepare_image(im))
                with torch.inference_mode():
                    prob = torch.cat([model(b).softmax(1) for b in torch.stack(xs).split(64)]).numpy()
                actual=np.asarray(info['classes'])[prob.argmax(1)]
                recorded=preds[['prob_'+c for c in info['classes']]].to_numpy()
                assert np.allclose(prob,recorded,atol=2e-5,rtol=2e-4)
                entry['max_probability_error']=float(np.abs(prob-recorded).max())
            assert np.array_equal(actual,preds.predicted_label),name+' checkpoint predictions differ'
            entry['checkpoint_inference_verified']=True
        evidence['models'][name]=entry
        print(name, f'{acc:.4%}', 'PASS',flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(evidence,indent=2)+'\n')
    print('All checks passed.')

if __name__ == '__main__': main()
