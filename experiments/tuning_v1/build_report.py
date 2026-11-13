from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
R=Path(__file__).resolve().parent;D=R/'results';df=pd.read_csv(D/'comparison.csv');winner=json.loads((D/'completed.json').read_text())['winner_by_validation']
labels={'cnn_c2_baseline':'CNN C2 baseline','cnn_low_lr':'CNN: lower LR','cnn_smoothing':'CNN: label smoothing','cnn_small_step':'CNN: smaller step','mobilenet_v3_frozen':'ImageNet MobileNetV3 + LR'}
base=float(df[df.name=='cnn_c2_baseline'].accuracy.iloc[0]);best=df[df.name==winner].iloc[0]
md='# Tuning and pretrained comparison: measured results\n\n'
md+='Selection used validation macro F1 before test-image loading. All five measured configurations share the same 584 test images.\n\n| Configuration | Validation F1 | Test accuracy | Test macro F1 | Correct / 584 |\n|---|---:|---:|---:|---:|\n'
for row in df.itertuples():md+=f'| {labels[row.name]} | {row.validation_macro_f1:.4f} | {row.accuracy*100:.2f}% | {row.macro_f1:.4f} | {row.correct} |\n'
md+=f'\nValidation-selected configuration: **{labels[winner]}**. It scores **{best.accuracy*100:.2f}% test accuracy** and **{best.macro_f1:.4f} macro F1**, a change of **{(best.accuracy-base)*100:+.2f} percentage points** from CNN C2. This is an observed single-run comparison, not a significance claim.\n\n'
md+='Three CNN trials each ran 15 additional epochs from the same C2 checkpoint. MobileNet uses a frozen ImageNet feature extractor plus a validation-tuned logistic head, with original and horizontally flipped training images. It is not a fully fine-tuned backbone.\n\nEfficientNet-B0 and DenseNet201 could not be executed: their official pretrained weights failed to download because DNS resolution is unavailable. Their literature scores (99.11% and 99.32% on binary crack detection) are external references only. See [PUBLISHED_COMPARISON.md](PUBLISHED_COMPARISON.md) for sources and task differences.\n\nThe old final report remains a snapshot of the earlier eight configurations. This addendum supplies the new experiment; do not present its results as if they were already included in that report. All original artifacts are preserved.\n'
(R/'TUNING_RESULTS.md').write_text(md)
fig,ax=plt.subplots(figsize=(9,4.6));bars=ax.barh([labels[n] for n in df.name],df.accuracy*100,color=['#087f8c' if n==winner else '#597387' for n in df.name]);ax.set_xlim(0,100);ax.set_xlabel('Accuracy (%) on the same 584 images');ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
for b,val in zip(bars,df.accuracy*100):ax.text(val+1,b.get_y()+b.get_height()/2,f'{val:.2f}%',va='center',fontsize=9)
fig.tight_layout();fig.savefig(D/'comparison.png',dpi=180);plt.close(fig)
styles=getSampleStyleSheet();styles['Normal'].fontSize=10;styles['Normal'].leading=14;styles['Normal'].spaceAfter=8
story=[]
def p(text,style='Normal'):story.append(Paragraph(text,styles[style]))
def table(headers,rows,widths):
 data=[[Paragraph(str(x),styles['Normal']) for x in row] for row in [headers]+rows]
 t=Table(data,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e0eaef')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#172d40')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#cccccc')),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t)
p('CNN tuning and pretrained-model comparison','Title');p('ML Final experimental addendum | 13 November 2026')
p('Measured comparison','Heading1');p('All configurations use the unchanged, deduplicated seven-class BD3 split: 2,725 training, 584 validation and 584 test images. Validation macro F1 controls epoch, head regularization and model selection. Test images are loaded only after selections are frozen.')
table(['Configuration','Val F1','Test accuracy','Test F1','Correct'],[[labels[x.name],f'{x.validation_macro_f1:.4f}',f'{x.accuracy*100:.2f}%',f'{x.macro_f1:.4f}',str(x.correct)] for x in df.itertuples()],[190,65,85,65,50])
p(f'Validation selects <b>{labels[winner]}</b>: {best.accuracy*100:.2f}% test accuracy, macro F1 {best.macro_f1:.4f}. Change from the original CNN C2: {(best.accuracy-base)*100:+.2f} percentage points.')
story.append(Image(str(D/'comparison.png'),width=455,height=233));p('Figure 1. Measured test accuracy; teal identifies the validation-selected configuration. The axis starts at zero.')
story.append(PageBreak());p('What was tuned and how it was checked','Heading1')
p('CNN continuation trials','Heading2');p('Every trial starts from the same preserved CNN C2 checkpoint, then trains for 15 epochs with batch size 64, AdamW, cosine decay to 0.000001 and gradient clipping at 1.0. Training-only random flips and rotations remain active. Each trial retains the epoch with highest validation macro F1; the unchanged C2 is also a candidate.')
table(['Trial','Initial LR','Weight decay','Label smoothing'],[['Lower LR','0.0001','0.0001','0'],['Smoothing','0.0001','0.001','0.05'],['Smaller step','0.00003','0.0001','0']],[150,95,105,105])
p('Transfer-learning baseline','Heading2');p('MobileNetV3 Small loads the locally cached official ImageNet1K V1 weights, checked against their published filename hash prefix. Its 1,000-class classifier is removed. The frozen backbone extracts 576 pooled features using the official resize, center-crop and ImageNet normalization. A StandardScaler and multinomial logistic head train on original and horizontally flipped training images. C is searched over 0.01, 0.1, 1 and 10 using validation macro F1. This is frozen-feature transfer, not end-to-end backbone fine-tuning.')
p('Evidence and limitations','Heading2');p('The results folder retains the protocol, histories, selected checkpoints, selection-freeze record, per-image probabilities, predictions and class reports. The same test membership is used for all measured rows. This is a single seed and a reused holdout. Repeated validation searches can overfit validation; neither a tiny change nor one improved score establishes statistical significance or performance on new buildings. The earlier label-provenance and possible same-building-overlap limitations still apply.')
p('The previous final report and slides are preserved as the earlier study snapshot. This addendum contains the new results. The experiment demo is experiments/tuning_v1/app.py.')
story.append(PageBreak());p('Published 99% results: separate benchmarks','Heading1')
p('A published score is specific to a trained system, dataset, task and evaluation split. It is not a transferable property of an architecture or an ImageNet initialization.')
table(['Published system','Reported accuracy','Context'],[['EfficientNet-B0 [1]','99.11%; 97.37% on different data','Binary crack/noncrack'],['DenseNet201 [1]','99.32%; 97.49% on different data','Binary crack/noncrack'],['VGG transfer [2]','99.83% CCIC; 99.72% BCD; 97.07% SDNET','Binary crack datasets'],['Pretrained ViT on BD3 [3]','98.79% F1, not accuracy','Authors\' augmented data and 60/20/20 split']],[130,180,145])
p('EfficientNet-B0 and DenseNet201 were selected for an additional local benchmark, but official weight downloads failed because the runtime could not resolve download.pytorch.org. No local test scores are assigned to them. A reproducible runner accepts downloaded weights or downloads them when networking works. The published BD3 README table and abstract disagree on original-data ViT F1; the table says 0.9323 and the abstract 0.9342. Its augmented-data F1 is 0.9879 in both.')
p('References','Heading2')
p('[1] Su, C., &amp; Wang, W. (2020). Concrete cracks detection using convolutional neural network based on transfer learning. Mathematical Problems in Engineering, Article 7240129. Table 2. https://doi.org/10.1155/2020/7240129')
p('[2] Yang, Q., Shi, W., Chen, J., &amp; Lin, W. (2020). Deep convolution neural network-based transfer learning method for civil infrastructure crack detection. Automation in Construction, 116, 103199. https://doi.org/10.1016/j.autcon.2020.103199')
p('[3] Kottari, P., &amp; Arjunan, P. (2024). BD3 benchmark repository. https://github.com/Praveenkottari/BD3-Dataset')
p('[4] Torchvision. MobileNetV3 Small pretrained weights and transforms. https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html')
def footer(c,d):c.setFont('Helvetica',9);c.drawRightString(555,25,f'ML Final tuning | {d.page}')
SimpleDocTemplate(str(R/'Tuning_Comparison.pdf'),pagesize=A4,rightMargin=70,leftMargin=70,topMargin=40,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
print(md)
