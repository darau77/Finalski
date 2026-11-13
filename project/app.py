"""Local demonstration of five primary model families and supporting experiments."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from features import extract_features

from inference import ROOT, REGISTRY, PRIMARY, CLASSES, load_model as load_saved, predict_images
registry=REGISTRY
st.set_page_config(page_title='Building Surface Classifier',page_icon='🧱',layout='wide')
st.title('Building Surface Classifier')
st.write('Five main models, seven building-surface categories, one shared test subset.')
show_extra=st.checkbox('Show supporting experiments and historical configurations')
choices=PRIMARY+([x for x in registry if x not in PRIMARY] if show_extra else [])
name=st.selectbox('Model',choices,format_func=lambda x:registry[x].get('display_name',x))
info=registry[name]
st.caption(f"Recorded test accuracy: {info['test_accuracy']:.2%} · 584 shared test images")
st.caption(info.get('initialization','No external pretrained image weights'))
@st.cache_resource
def load_model(name):return load_saved(name)
def classify(image,name):
 model=load_model(name);probabilities=predict_images(model,name,[image])[0]
 classes=list(model.classes_) if registry[name]['type']=='classical' else CLASSES
 return pd.DataFrame({'Class':[str(c).replace('_',' ').title() for c in classes],'Model probability':probabilities}).sort_values('Model probability',ascending=False)

source = st.radio('Image source', ['Bundled example','Upload a photo'], horizontal=True)
if source == 'Bundled example':
    example = st.selectbox('Example image', sorted(p.name for p in (ROOT/'test_images').glob('*.png')))
    file = ROOT/'test_images'/example
else:
    file = st.file_uploader('Choose a wall photograph',type=['jpg','jpeg','png','webp'])
if file is not None:
    try:
        with Image.open(file) as opened:
            rgba = ImageOps.exif_transpose(opened).convert('RGBA')
            image = Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
        left,right = st.columns(2)
        with left: st.image(image, caption='Input image',width=350)
        with right:
            if st.button('Classify',type='primary'):
                scores=classify(image,name)
                st.success('Prediction: '+scores.iloc[0]['Class'])
                st.dataframe(scores,hide_index=True)
                st.caption('These are model scores, not calibrated guarantees of correctness.')
    except (OSError,ValueError,Image.DecompressionBombError):
        st.error('Please select a valid image.')
with st.expander('Recorded comparison and evaluation protocol'):
    st.dataframe(pd.read_csv(ROOT/'results/model_comparison.csv').round(4),hide_index=True)
    st.write('Classical baselines use five-fold CV on 3,309 development images. Neural models use 2,725 training and 584 validation images. All use the same 584-image test membership.')
    st.write('MobileNetV3 is the default: validation macro F1 selected its logistic head and the overall winner in the tuning comparison. The tuned CNN uses label smoothing. C1 weights are unavailable; C1 is not offered for inference.')
st.caption('Classroom demonstration. Inputs are processed locally and not saved. Every image receives a known label, including unrelated images. This is not a structural-safety assessment.')
