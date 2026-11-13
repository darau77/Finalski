"""Shared local inference for all saved final-project configurations."""
from pathlib import Path
import json,importlib
import joblib,numpy as np,torch
from PIL import ImageOps,Image
ROOT=Path(__file__).resolve().parent
REGISTRY=json.loads((ROOT/'models/registry.json').read_text())
PRIMARY=['MobileNetV3 transfer','CNN tuned smoothing','ViT trained from scratch','Random Forest (tuned)','Logistic Regression']
CLASSES=['algae','major_crack','minor_crack','peeling','plain','spalling','stain']
def clean_image(image):
 image=ImageOps.exif_transpose(image).convert('RGBA')
 return Image.alpha_composite(Image.new('RGBA',image.size,'white'),image).convert('RGB')
def load_model(name):
 info=REGISTRY[name];torch.set_num_threads(4)
 if info['type']=='classical':return joblib.load(ROOT/info['file'])
 if info['type']=='mobile':
  from torchvision.models import mobilenet_v3_small
  m=mobilenet_v3_small(weights=None);m.classifier=torch.nn.Identity();m.load_state_dict(torch.load(ROOT/info['file'],map_location='cpu',weights_only=True))
  return m.eval(),joblib.load(ROOT/info['head_file'])
 module=importlib.import_module(info['type']+'_model');m=module.build_model();m.load_state_dict(torch.load(ROOT/info['file'],map_location='cpu',weights_only=True));return m.eval()
def predict_images(model,name,images):
 info=REGISTRY[name];images=[clean_image(im) for im in images]
 if info['type']=='classical':
  from features import extract_features
  return model.predict_proba(np.asarray([extract_features(im) for im in images]))
 if info['type']=='mobile':
  from torchvision.models import MobileNet_V3_Small_Weights
  tf=MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms();m,head=model
  with torch.inference_mode():features=m(torch.stack([tf(im) for im in images])).numpy()
  return head.predict_proba(features)
 tf=importlib.import_module(info['type']+'_model').prepare_image
 with torch.inference_mode():return model(torch.stack([tf(im) for im in images])).softmax(1).numpy()
