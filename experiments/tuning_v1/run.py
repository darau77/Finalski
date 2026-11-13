"""Validation-only CNN tuning and cached ImageNet transfer benchmark.
Run from repository root; never overwrites an existing output directory.
"""
from pathlib import Path
import sys,argparse,json,hashlib,time,random,csv,copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
from torchvision.transforms.functional import rotate
from PIL import Image,ImageOps
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
import joblib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'project'))
from cnn_model import build_model,prepare_image
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2))
def seed():torch.manual_seed(42);np.random.seed(42);random.seed(42)
def batches(x,y,shuffle=False,batch=64):return DataLoader(TensorDataset(x,y),batch_size=batch,shuffle=shuffle)
def infer(model,x,batch=64):
 model.eval()
 with torch.inference_mode():return torch.cat([model(b).softmax(1).cpu() for b in x.split(batch)]).numpy()
def metrics(y,pr):return {'accuracy':float(accuracy_score(y,pr.argmax(1))),'macro_f1':float(f1_score(y,pr.argmax(1),average='macro'))}
def load_images(df,base,transform):
 ims=[]
 for name in df.path:
  with Image.open(base/name) as im:
   im=ImageOps.exif_transpose(im).convert('RGBA');im=Image.alpha_composite(Image.new('RGBA',im.size,'white'),im).convert('RGB');ims.append(transform(im))
 return torch.stack(ims)
def mobile_features(model,df,base,transform,flip=False):
 pieces=[];model.eval()
 with torch.inference_mode():
  for start in range(0,len(df),32):
   x=load_images(df.iloc[start:start+32],base,transform)
   if flip:x=x.flip(-1)
   pieces.append(model(x).cpu().numpy())
 return np.concatenate(pieces)
def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--mobile-weights',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise ValueError('Choose a new output directory')
 torch.set_num_threads(4);seed()
 manifest=ROOT/'project/results/split_manifest.csv';df=pd.read_csv(manifest).sort_values('path');classes=sorted(df.label.unique());assert len(df)==3893 and df.pixel_hash.is_unique
 subsets={k:df[df.split==k].reset_index(drop=True) for k in ['train','validation','test']}
 ys={k:np.array([classes.index(v) for v in d.label]) for k,d in subsets.items()}
 configs=[{'name':'cnn_low_lr','lr':0.0001,'weight_decay':0.0001,'label_smoothing':0.0},{'name':'cnn_smoothing','lr':0.0001,'weight_decay':0.001,'label_smoothing':0.05},{'name':'cnn_small_step','lr':0.00003,'weight_decay':0.0001,'label_smoothing':0.0}]
 protocol={'seed':42,'threads':4,'device':'cpu','cnn_initialization':'existing validation-selected CNN C2','cnn_initial_hash':sha(ROOT/'project/models/cnn.pt'),'cnn_trials':configs,'cnn_epochs_per_trial':15,'cnn_batch':64,'cnn_optimizer':'AdamW','cnn_schedule':'cosine to 1e-6','cnn_augmentation':'per-image flip and batch rotation ±10 degrees','mobile_weights_sha256':sha(args.mobile_weights),'mobile_expected_hash_prefix':'047dcff4','mobile_architecture':'torchvision MobileNetV3 Small ImageNet1K V1, classifier removed; 576 pooled features','mobile_transform':'official weights.transforms(): resize 256, center crop224, ImageNet normalize; one additional horizontal flip per training image','mobile_C_grid':[0.01,0.1,1.0,10.0],'mobile_head':'StandardScaler + multinomial LogisticRegression max_iter3000; original and flipped training embeddings','selection':'Highest validation macro F1; earliest epoch/trial on ties; compare against unchanged C2 validation score','test_policy':'All candidates and selected checkpoints frozen before any test images loaded; evaluate each CNN trial winner and selected MobileNet head once. Reused test set, no further tuning from test.','split_counts':df.split.value_counts().to_dict(),'manifest_sha256':sha(manifest),'created_at':pd.Timestamp.now(tz='UTC').isoformat()}
 assert protocol['mobile_weights_sha256'].startswith('047dcff4'), 'Unexpected pretrained weights'
 save(out/'protocol.json',protocol)
 x={k:load_images(subsets[k],args.data,prepare_image) for k in ['train','validation']}
 original=torch.load(ROOT/'project/models/cnn.pt',map_location='cpu',weights_only=True)
 base=build_model();base.load_state_dict(original);basem=metrics(ys['validation'],infer(base,x['validation']));print('Baseline validation',basem,flush=True)
 all_candidates=[]
 for cfg in configs:
  seed();model=build_model();model.load_state_dict(original);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=15,eta_min=1e-6);lossfn=nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing']);best=-1;hist=[]
  for ep in range(1,16):
   start=time.perf_counter();model.train();total=0
   for bx,by in batches(x['train'],torch.tensor(ys['train']),True):
    flip=torch.rand(len(bx))<.5;bx[flip]=bx[flip].flip(-1);bx=rotate(bx,float(torch.empty(1).uniform_(-10,10)))
    opt.zero_grad();loss=lossfn(model(bx),by);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();total+=float(loss.detach())*len(by)
   m=metrics(ys['validation'],infer(model,x['validation']));row={'epoch':ep,'lr':opt.param_groups[0]['lr'],'train_loss':total/len(x['train']),**m,'seconds':time.perf_counter()-start};hist.append(row)
   if m['macro_f1']>best:best=m['macro_f1'];chosen=row;torch.save(model.state_dict(),out/(cfg['name']+'.pt'))
   sched.step();pd.DataFrame(hist).to_csv(out/(cfg['name']+'_history.csv'),index=False);print(cfg['name'],ep,round(m['macro_f1'],4),round(row['seconds'],1),flush=True)
  all_candidates.append({'name':cfg['name'],'validation':chosen,'checkpoint_sha256':sha(out/(cfg['name']+'.pt'))})
 save(out/'cnn_validation.json',{'baseline':basem,'trials':all_candidates})
 del x
 print('Extracting frozen ImageNet MobileNetV3 features',flush=True)
 mobile=mobilenet_v3_small(weights=None);mobile.load_state_dict(torch.load(args.mobile_weights,map_location='cpu',weights_only=True));mobile.classifier=nn.Identity();tf=MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
 tr=mobile_features(mobile,subsets['train'],args.data,tf);trf=mobile_features(mobile,subsets['train'],args.data,tf,True);val=mobile_features(mobile,subsets['validation'],args.data,tf)
 xx=np.concatenate([tr,trf]);yy=np.tile(ys['train'],2);best=-1;heads=[]
 for C in protocol['mobile_C_grid']:
  head=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000,random_state=42));head.fit(xx,yy);m=metrics(ys['validation'],head.predict_proba(val));heads.append({'C':C,**m});print('MobileNet C',C,m,flush=True)
  if m['macro_f1']>best:best=m['macro_f1'];chosen_mobile={'name':'mobilenet_v3_frozen','C':C,'validation':m};joblib.dump(head,out/'mobilenet_head.joblib')
 torch.save(mobile.state_dict(),out/'mobilenet_backbone.pt');save(out/'mobile_validation.json',heads)
 candidates=[{'name':'cnn_c2_baseline','validation':basem}]+[{'name':v['name'],'validation':{k:v['validation'][k] for k in ['accuracy','macro_f1']}} for v in all_candidates]+[chosen_mobile]
 winner=max(candidates,key=lambda d:d['validation']['macro_f1'])
 save(out/'selection_frozen_before_test.json',{'candidates':candidates,'winner':winner,'test_loaded':False,'time':pd.Timestamp.now(tz='UTC').isoformat(),'mobile_head_sha256':sha(out/'mobilenet_head.joblib'),'mobile_backbone_sha256':sha(out/'mobilenet_backbone.pt')})
 print('Selection frozen. Loading test images now.',flush=True)
 testx=load_images(subsets['test'],args.data,prepare_image);results=[]
 for candidate in candidates:
  name=candidate['name']
  if name=='mobilenet_v3_frozen':pr=joblib.load(out/'mobilenet_head.joblib').predict_proba(mobile_features(mobile,subsets['test'],args.data,tf))
  else:
   model=build_model();model.load_state_dict(original if name=='cnn_c2_baseline' else torch.load(out/(name+'.pt'),map_location='cpu',weights_only=True));pr=infer(model,testx)
  m=metrics(ys['test'],pr);results.append({'name':name,'validation_macro_f1':candidate['validation']['macro_f1'],**m,'correct':int((ys['test']==pr.argmax(1)).sum()),'n':len(pr)})
  np.save(out/(name+'_probabilities.npy'),pr);pd.DataFrame({'path':subsets['test'].path,'true_label':subsets['test'].label,'predicted_label':[classes[i] for i in pr.argmax(1)]}).to_csv(out/(name+'_predictions.csv'),index=False)
  save(out/(name+'_class_report.json'),classification_report(ys['test'],pr.argmax(1),target_names=classes,output_dict=True,zero_division=0));print('TEST',name,m,flush=True)
 pd.DataFrame(results).to_csv(out/'comparison.csv',index=False);save(out/'completed.json',{'winner_by_validation':winner['name'],'results':results,'completed_at':pd.Timestamp.now(tz='UTC').isoformat()})
if __name__=='__main__':main()
