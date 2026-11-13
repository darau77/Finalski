"""Run ImageNet EfficientNet-B0 and DenseNet201 on the unchanged BD3 split.
These architectures have published 99% binary-crack results; this is a fresh
seven-class transfer experiment, not a reproduction of those paper checkpoints.
"""
from run import *
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights,densenet201,DenseNet201_Weights
import socket,re

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--weights-dir',type=Path);args=p.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise ValueError('Use a new output directory')
 torch.set_num_threads(4);socket.setdefaulttimeout(15);seed()
 manifest=ROOT/'project/results/split_manifest.csv';df=pd.read_csv(manifest).sort_values('path');assert df.pixel_hash.is_unique and len(df)==3893
 classes=sorted(df.label.unique());subsets={k:df[df.split==k].reset_index(drop=True) for k in ['train','validation','test']};ys={k:np.array([classes.index(v) for v in d.label]) for k,d in subsets.items()}
 configs=[('efficientnet_b0',efficientnet_b0,EfficientNet_B0_Weights.IMAGENET1K_V1),('densenet201',densenet201,DenseNet201_Weights.IMAGENET1K_V1)]
 save(out/'protocol.json',{'seed':42,'manifest_sha256':sha(manifest),'models':[n for n,_,_ in configs],'training':'Frozen ImageNet backbone, official transforms; original plus flipped train embeddings; StandardScaler and LogisticRegression','C_grid':[.01,.1,1.,10.],'selection':'validation macro F1 only','test':'Once after all available model heads selected; previously reused holdout','published_scores_are_not_BD3_results':True})
 available=[];status=[]
 for name,builder,weights in configs:
  filename=weights.url.rsplit('/',1)[-1];local=(args.weights_dir/filename) if args.weights_dir else out/'weights'/filename
  try:
   if not local.is_file():
    local.parent.mkdir(parents=True,exist_ok=True);torch.hub.download_url_to_file(weights.url,str(local),hash_prefix=filename.rsplit('-',1)[-1].split('.')[0],progress=False)
   assert sha(local).startswith(filename.rsplit('-',1)[-1].split('.')[0])
   model=builder(weights=None);state=torch.load(local,map_location='cpu',weights_only=True)
   if name=='densenet201':
    pattern=re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state):
     match=pattern.match(key)
     if match:state[match.group(1)+match.group(2)]=state.pop(key)
   model.load_state_dict(state);model.classifier=nn.Identity();tf=weights.transforms()
   train=mobile_features(model,subsets['train'],args.data,tf);flipped=mobile_features(model,subsets['train'],args.data,tf,True);val=mobile_features(model,subsets['validation'],args.data,tf);best=-1;trials=[]
   for C in [.01,.1,1.,10.]:
    head=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000,random_state=42));head.fit(np.concatenate([train,flipped]),np.tile(ys['train'],2));m=metrics(ys['validation'],head.predict_proba(val));trials.append({'C':C,**m})
    if m['macro_f1']>best:best=m['macro_f1'];chosen={'C':C,**m};joblib.dump(head,out/(name+'_head.joblib'))
   save(out/(name+'_validation.json'),trials);torch.save(model.state_dict(),out/(name+'_backbone.pt'));available.append((name,model,tf,chosen));status.append({'model':name,'status':'head_selected','selection':chosen,'source_weights_sha256':sha(local)})
  except (OSError,RuntimeError) as e:status.append({'model':name,'status':'blocked_or_failed','error':str(e),'weights_url':weights.url});print(name,str(e),flush=True)
 save(out/'selection_frozen_before_test.json',{'models':status,'test_loaded':False})
 results=[]
 for name,model,tf,chosen in available:
  features=mobile_features(model,subsets['test'],args.data,tf);probs=joblib.load(out/(name+'_head.joblib')).predict_proba(features);m=metrics(ys['test'],probs);results.append({'name':name,'validation_macro_f1':chosen['macro_f1'],**m});np.save(out/(name+'_test_probabilities.npy'),probs)
 save(out/'status.json',{'models':status,'measured_test_results':results})
 if results:pd.DataFrame(results).to_csv(out/'comparison.csv',index=False)
if __name__=='__main__':main()
