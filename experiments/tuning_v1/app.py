"""Demo of the measured tuning and ImageNet transfer-learning experiment."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import streamlit as st
import torch,joblib
from torch import nn
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
from PIL import Image,ImageOps
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT/'project'))
from cnn_model import build_model,prepare_image
st.set_page_config(page_title='Tuning and transfer comparison',layout='wide')
st.title('CNN tuning and pretrained-model comparison')
results=HERE/'results'
if not (results/'completed.json').exists():st.info('Experiments are still running.');st.stop()
meta=json.loads((results/'completed.json').read_text());table=pd.read_csv(results/'comparison.csv')
st.caption('Same 584-image test subset. Selection used validation macro F1. Published 99% scores refer to separate binary-crack experiments.')
st.dataframe(table,hide_index=True)
choices=list(table.name);selected=st.selectbox('Model',choices,index=choices.index(meta['winner_by_validation']))
@st.cache_resource
def load(name):
 torch.set_num_threads(4)
 if name=='mobilenet_v3_frozen':
  m=mobilenet_v3_small(weights=None);m.classifier=nn.Identity();m.load_state_dict(torch.load(results/'mobilenet_backbone.pt',map_location='cpu',weights_only=True));return m.eval(),joblib.load(results/'mobilenet_head.joblib')
 m=build_model();path=ROOT/'project/models/cnn.pt' if name=='cnn_c2_baseline' else results/(name+'.pt');m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));return m.eval(),None
example=st.selectbox('Bundled image',sorted(p.name for p in (ROOT/'project/test_images').glob('*.png')))
upload=st.file_uploader('Or upload a photo',type=['jpg','jpeg','png','webp'])
with Image.open(upload if upload is not None else ROOT/'project/test_images'/example) as im:
 im=ImageOps.exif_transpose(im).convert('RGBA');im=Image.alpha_composite(Image.new('RGBA',im.size,'white'),im).convert('RGB')
st.image(im,width=300)
if st.button('Classify'):
 m,head=load(selected)
 tf=MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms() if head is not None else prepare_image
 with torch.inference_mode():out=m(tf(im).unsqueeze(0))
 prob=head.predict_proba(out.numpy())[0] if head is not None else out.softmax(1)[0].numpy()
 classes=['algae','major_crack','minor_crack','peeling','plain','spalling','stain'];st.success('Prediction: '+classes[int(np.argmax(prob))]);st.dataframe(pd.DataFrame({'class':classes,'probability':prob}),hide_index=True)
st.caption('MobileNet uses a frozen ImageNet backbone and a fitted seven-class head. Probabilities are uncalibrated. This is not a safety assessment.')
