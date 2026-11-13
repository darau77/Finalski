"""Local demonstration of five primary model families and supporting experiments."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from features import extract_features

ROOT = Path(__file__).resolve().parent
registry = json.loads((ROOT / 'models/registry.json').read_text())
PRIMARY = ['CNN trained from scratch', 'ViT trained from scratch', 'Random Forest (tuned)', 'Logistic Regression', 'KNN']
st.set_page_config(page_title='Building Surface Classifier', page_icon='🧱', layout='wide')
st.title('Building Surface Classifier')
st.write('Compare five model families on seven building-surface categories.')
show_extra = st.checkbox('Show supporting experiments (CNN C3, untuned forest and historical tree)')
choices = PRIMARY + ([x for x in registry if x not in PRIMARY] if show_extra else [])
name = st.selectbox('Model', choices, format_func=lambda x: registry[x].get('display_name', x))
info = registry[name]
st.caption(f"Recorded test accuracy: {info['test_accuracy']:.2%} · 584 shared test images · trained without pretrained weights")

@st.cache_resource
def load_model(name):
    info = registry[name]
    if info['type'] in ('cnn', 'vit'):
        import importlib, torch
        torch.set_num_threads(4)
        model = importlib.import_module(info['type']+'_model').build_model()
        model.load_state_dict(torch.load(ROOT/info['file'], map_location='cpu', weights_only=True))
        return model.eval()
    return joblib.load(ROOT/info['file'])

def classify(image, name):
    info = registry[name]; model = load_model(name)
    if info['type'] in ('cnn','vit'):
        import importlib, torch
        prepare = importlib.import_module(info['type']+'_model').prepare_image
        with torch.inference_mode(): probabilities = model(prepare(image).unsqueeze(0)).softmax(1)[0].numpy()
        classes = info['classes']
    else:
        probabilities = model.predict_proba([extract_features(image)])[0]
        classes = list(model.classes_)
    return pd.DataFrame({'Class':[str(c).replace('_',' ').title() for c in classes], 'Model probability': probabilities}).sort_values('Model probability',ascending=False)

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
    st.write('CNN C2 is the default and the best validation-selected CNN configuration. C1 is historical: its checkpoint is unavailable, so it is not offered for inference.')
st.caption('Classroom demonstration. Inputs are processed locally and not saved. Every image receives a known label, including unrelated images. This is not a structural-safety assessment.')
