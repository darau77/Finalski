"""Train fixed endterm CNN configuration 3 without changing the original project."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import PIL
from PIL import Image
import sklearn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score, recall_score)
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision.transforms.functional import rotate
from cnn_model import build_model, prepare_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True,
                        help='Original project containing results/split_manifest.csv')
    parser.add_argument('--data', type=Path, required=True, help='BD3_original_dataset directory')
    parser.add_argument('--output', type=Path, default=Path('config3_results'))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a new empty output directory to preserve existing evidence.')
    manifest = args.project / 'results/split_manifest.csv'
    data = pd.read_csv(manifest).sort_values('path').reset_index(drop=True)
    classes = sorted(data.label.unique())
    assert len(classes) == 7
    assert len(data) == 3893 and data.path.is_unique and data.pixel_hash.is_unique
    assert data.split.value_counts().to_dict() == {'train': 2725, 'test': 584, 'validation': 584}
    for first, second in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not set(data[data.split == first].pixel_hash) & set(data[data.split == second].pixel_hash)
    protocol = {
        'configuration': 'CNN Config 3 constant learning rate',
        'frozen_before_training_utc': datetime.now(timezone.utc).isoformat(),
        'epochs': 40, 'batch_size': 64, 'seed': 42, 'device': 'cpu', 'threads': 4,
        'optimizer': 'Adam', 'learning_rate': 0.0005, 'scheduler': None,
        'weight_decay': 0.0, 'loss': 'CrossEntropyLoss',
        'architecture': 'Unchanged cnn_model.py; 361095 randomly initialized parameters',
        'augmentation': 'Per-image horizontal flip p=0.5 and one random rotation angle in [-10,10] degrees per training batch, matching revised notebook',
        'preprocessing': 'EXIF orientation, white alpha composite, RGB, resize 96x96, scale to [0,1]',
        'checkpoint_selection': 'Highest validation macro F1 across all 40 epochs; earliest epoch in a tie',
        'test_protocol': 'Load test pixels only after training and freezing the validation-selected checkpoint; evaluate once; no tuning from test',
        'comparison': 'Same prescribed architecture, seed, batch, initial learning rate and epoch budget as archival revised CNN; removes cosine scheduler. Historical execution environment may differ.',
        'test_reuse_note': 'This fixed test set was used for prior project models and is not a fresh blind benchmark.',
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'model_source_sha256': hashlib.sha256(Path(__file__).with_name('cnn_model.py').read_bytes()).hexdigest(),
        'split_counts': data.split.value_counts().to_dict(),
        'checks': '3893 unique paths and pixel hashes; no manifest hash overlap between splits',
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
        'machine': platform.machine(), 'processor': platform.processor(), 'torch': torch.__version__,
        'torchvision': torchvision.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
        'scikit_learn': sklearn.__version__, 'Pillow': PIL.__version__, 'device': 'cpu', 'torch_threads': 4}
    (output / 'environment.json').write_text(json.dumps(environment, indent=2))

    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    loaders = {}
    for split in ['train', 'validation']:
        subset = data[data.split == split]
        images = []
        for path in subset.path:
            with Image.open(args.data / path) as image:
                images.append(prepare_image(image))
        X = torch.stack(images)
        y = torch.tensor([classes.index(label) for label in subset.label])
        loaders[split] = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=(split == 'train'))
    print('Loaded train and validation only. Test pixels not loaded.', flush=True)
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_function = nn.CrossEntropyLoss()
    history, best_f1 = [], -1
    best = None
    for epoch in range(1, 41):
        start = time.perf_counter()
        model.train()
        train_loss, train_correct = 0.0, 0
        for batch, labels in loaders['train']:
            flip = torch.rand(len(batch)) < 0.5
            batch[flip] = batch[flip].flip(-1)
            batch = rotate(batch, float(torch.empty(1).uniform_(-10, 10)))
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
            train_correct += (logits.argmax(1) == labels).sum().item()
        model.eval()
        predictions, true_labels = [], []
        validation_loss = 0.0
        with torch.inference_mode():
            for batch, labels in loaders['validation']:
                logits = model(batch)
                validation_loss += loss_function(logits, labels).item() * len(batch)
                predictions.extend(logits.argmax(1).tolist())
                true_labels.extend(labels.tolist())
        row = {'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr'],
            'train_loss': train_loss / len(loaders['train'].dataset),
            'train_accuracy': train_correct / len(loaders['train'].dataset),
            'validation_loss': validation_loss / len(loaders['validation'].dataset),
            'validation_accuracy': accuracy_score(true_labels, predictions),
            'validation_macro_f1': f1_score(true_labels, predictions, average='macro', zero_division=0),
            'seconds': round(time.perf_counter() - start, 2)}
        history.append(row)
        if row['validation_macro_f1'] > best_f1:
            best_f1, best = row['validation_macro_f1'], row.copy()
            torch.save(model.state_dict(), output / 'cnn_config3.pt')
            (output / 'best_validation.json').write_text(json.dumps(best, indent=2))
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        print(f"Epoch {epoch:02d}/40 train={row['train_accuracy']:.4f} val={row['validation_accuracy']:.4f} F1={row['validation_macro_f1']:.4f} loss={row['validation_loss']:.4f} sec={row['seconds']:.2f}", flush=True)

    frozen = {'frozen_after_training_utc': datetime.now(timezone.utc).isoformat(),
        'selected_epoch': best['epoch'], 'best_validation_macro_f1': best_f1,
        'checkpoint_sha256': hashlib.sha256((output / 'cnn_config3.pt').read_bytes()).hexdigest(),
        'test_pixels_loaded': False}
    (output / 'checkpoint_frozen_before_test.json').write_text(json.dumps(frozen, indent=2))
    print('Training complete; selected checkpoint frozen. Loading test pixels now.', flush=True)
    test_data = data[data.split == 'test']
    images = []
    for path in test_data.path:
        with Image.open(args.data / path) as image:
            images.append(prepare_image(image))
    X_test = torch.stack(images)
    y_test = np.array([classes.index(label) for label in test_data.label])
    model.load_state_dict(torch.load(output / 'cnn_config3.pt', map_location='cpu', weights_only=True))
    model.eval()
    start = time.perf_counter()
    scores, total_loss = [], 0.0
    with torch.inference_mode():
        for index in range(0, len(X_test), 64):
            batch = X_test[index:index+64]
            logits = model(batch)
            scores.append(logits.softmax(1).numpy())
            total_loss += loss_function(logits, torch.tensor(y_test[index:index+64])).item() * len(batch)
    predict_seconds = time.perf_counter() - start
    probabilities = np.concatenate(scores)
    prediction = probabilities.argmax(1)
    matrix = confusion_matrix(y_test, prediction, labels=range(7))
    report = classification_report(y_test, prediction, labels=range(7), target_names=classes, output_dict=True, zero_division=0)
    metrics = {'model': 'CNN Config 3 constant learning rate', 'pretrained': False,
        'parameter_count': sum(p.numel() for p in model.parameters()), 'epochs_planned': 40,
        'best_epoch': best['epoch'], 'classes': classes,
        'validation_accuracy': best['validation_accuracy'], 'validation_macro_f1': best_f1,
        'validation_loss': best['validation_loss'], 'test_loss': total_loss / len(y_test),
        'test_accuracy': accuracy_score(y_test, prediction),
        'test_precision_macro': precision_score(y_test, prediction, average='macro', zero_division=0),
        'test_recall_macro': recall_score(y_test, prediction, average='macro', zero_division=0),
        'test_macro_f1': f1_score(y_test, prediction, average='macro', zero_division=0),
        'test_balanced_accuracy': balanced_accuracy_score(y_test, prediction),
        'correct_count': int((prediction == y_test).sum()),
        'fit_seconds': float(pd.DataFrame(history).seconds.sum()), 'predict_seconds': predict_seconds,
        'classification_report': report, 'confusion_matrix': matrix.tolist(),
        'training_settings': protocol, 'environment': environment,
        'timing_note': 'Sum of epoch times includes training and validation; excludes image loading, saving checkpoint/history. Test timing includes forward pass, softmax and loss calculation; excludes image loading. Historical times are not a controlled speed comparison.'}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    np.save(output / 'test_probabilities.npy', probabilities)
    test_results = test_data[['path', 'label']].copy().rename(columns={'label': 'true_label'})
    test_results['predicted_label'] = [classes[index] for index in prediction]
    for index, label in enumerate(classes):
        test_results['prob_' + label] = probabilities[:, index]
    test_results.to_csv(output / 'test_predictions.csv', index=False)
    print(json.dumps({k: metrics[k] for k in ['best_epoch','validation_accuracy','validation_macro_f1','validation_loss','test_loss','test_accuracy','test_macro_f1','correct_count','fit_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
