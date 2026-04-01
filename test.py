import warnings
import glob
warnings.filterwarnings('ignore')
import json
import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from datasets.SPRSound_dataloader import SPRSoundDataset, collate_fn_valid, get_class_names
from utils.evaluation_metrics import calculate_metrics
from feature_extractor.EfficientNet import CustomEfficientNetWithLoad, load_efficientnet_model
from feature_extractor.HaB_ResNet import CustomResNet
from feature_extractor.HaB_DenseNet import CustomDenseNet
from feature_extractor.HaB_ConvNeXt import CustomConvNeXt
from feature_extractor.HaB_MobileViTV2_timm import (
    mobilevitv2_050, mobilevitv2_075, mobilevitv2_100,
    mobilevitv2_125, mobilevitv2_150, mobilevitv2_175, mobilevitv2_200
)
from torchvision.models import resnet50, densenet121, convnext_base
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
import pandas as pd
from datetime import datetime
from tabulate import tabulate
from datasets.ICBHI2017_dataset import (
    ICBHI2017Dataset,
    collate_fn_icbhi_valid,
    get_class_names as get_icbhi_class_names
)

def get_mobilevitv2_model(variant, num_classes, pretrained=True):
    variant_map = {
        'MobileViTV2-050': mobilevitv2_050,
        'MobileViTV2-075': mobilevitv2_075,
        'MobileViTV2-100': mobilevitv2_100,
        'MobileViTV2-125': mobilevitv2_125,
        'MobileViTV2-150': mobilevitv2_150,
        'MobileViTV2-175': mobilevitv2_175,
        'MobileViTV2-200': mobilevitv2_200,
    }
    if variant not in variant_map:
        raise ValueError(f"Unknown MobileViTV2 variant: {variant}")
    model_fn = variant_map[variant]
    return model_fn(num_classes=num_classes, pretrained=pretrained)

def load_icbhi_test_datasets(data_dir, split_file, task_type, feature_type):
    test_datasets = {}
    try:
        test_dataset = ICBHI2017Dataset(
            data_dir=data_dir,
            split_file=split_file,
            task_type=task_type,
            mode='test',
            feature_type=feature_type,
            use_cache=True,
            use_oversampling=False,
            oversampler_type=None
        )
        if len(test_dataset) > 0:
            test_datasets['icbhi_test'] = test_dataset
            print(f"Loaded ICBHI test dataset: {len(test_dataset)} samples")
            labels = test_dataset.labels
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"   Test set class distribution:")
            for label, count in zip(unique_labels, counts):
                percentage = (count / len(labels)) * 100
                class_names = get_icbhi_class_names(task_type)
                class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
                print(f"     {class_name} (Class {label}): {count} samples ({percentage:.1f}%)")
        else:
            print("Warning: ICBHI test dataset is empty")
    except Exception as e:
        print(f"Error loading ICBHI test dataset: {e}")
        import traceback
        print(traceback.format_exc())
    return test_datasets

def determine_dataset_type_from_path(exp_dir):
    exp_dir_name = os.path.basename(exp_dir)
    if "ICBHI2017" in exp_dir_name or "icbhi" in exp_dir_name.lower():
        return "ICBHI2017"
    elif "SPRSound22_23" in exp_dir_name or "spr" in exp_dir_name.lower():
        return "SPRSound22_23"
    else:
        print(f"Warning: Cannot determine dataset type from directory name: {exp_dir_name}")
        print("Defaulting to SPRSound22_23. Use --dataset_type to specify explicitly.")
        return "SPRSound22_23"

def scan_available_models(exp_dir):
    possible_model_dirs = [
        os.path.join(exp_dir, 'models'),
        exp_dir,
        os.path.join(exp_dir, 'checkpoints'),
        os.path.join(exp_dir, 'weights'),
    ]
    all_pth_files = []
    used_dir = None
    for models_dir in possible_model_dirs:
        if os.path.exists(models_dir):
            pth_files = glob.glob(os.path.join(models_dir, "*.pth"))
            if pth_files:
                all_pth_files.extend(pth_files)
                used_dir = models_dir
                break
    if not all_pth_files:
        print(f"No .pth files found in common locations. Searching recursively in: {exp_dir}")
        all_pth_files = glob.glob(os.path.join(exp_dir, "**", "*.pth"), recursive=True)
        used_dir = exp_dir
    if not all_pth_files:
        print(f"No .pth files found in experiment directory: {exp_dir}")
        print("Please check the directory structure and model file locations.")
        return [], None, []
    print(f"Found {len(all_pth_files)} .pth files in: {used_dir}")
    for file in all_pth_files:
        print(f"  - {os.path.relpath(file, exp_dir)}")
    fold_models = {}
    final_model = None
    other_models = []
    for file_path in all_pth_files:
        filename = os.path.basename(file_path)
        fold_patterns = [
            r'fold_(\d+)_best\.pth',
            r'fold(\d+)_best\.pth',
            r'fold_(\d+)\.pth',
            r'fold(\d+)\.pth',
            r'model_fold_(\d+)\.pth',
            r'model_fold(\d+)\.pth'
        ]
        fold_found = False
        for pattern in fold_patterns:
            import re
            match = re.match(pattern, filename)
            if match:
                fold_num = int(match.group(1))
                fold_models[fold_num] = file_path
                fold_found = True
                break
        if fold_found:
            continue
        final_patterns = [
            'final_model.pth',
            'final.pth',
            'best_model.pth',
            'best.pth',
            'model_final.pth'
        ]
        if filename in final_patterns:
            final_model = file_path
        else:
            other_models.append(file_path)
    print(f"\nModel classification:")
    print(f"  Fold models: {len(fold_models)} found")
    for fold_num in sorted(fold_models.keys()):
        print(f"    Fold {fold_num}: {os.path.basename(fold_models[fold_num])}")
    if final_model:
        print(f"  Final model: {os.path.basename(final_model)}")
    else:
        print(f"  Final model: Not found")
    if other_models:
        print(f"  Other models: {len(other_models)} found")
        for model in other_models:
            print(f"    - {os.path.basename(model)}")
    return fold_models, final_model, other_models

def evaluate_and_print_metrics(preds, labels, task_type, dataset_name, dataset_type="SPRSound22_23"):
    (se, sp, avg_se, avg_sp, avg_score,
     harmonic_score, overall_score) = calculate_metrics(labels, preds, task_type)
    print(f"\n{'=' * 60}")
    print(f"{dataset_name.upper()} TEST RESULTS ({dataset_type})")
    print(f"{'=' * 60}")
    print(f"Performance Metrics:")
    print(f"   Sensitivity: {avg_se:.4f}")
    print(f"   Specificity: {avg_sp:.4f}")
    print(f"   Average Score: {avg_score:.4f}")
    print(f"   Harmonic Score: {harmonic_score:.4f}")
    print(f"   Overall Score: {overall_score:.4f}")
    if dataset_type == "ICBHI2017":
        class_names = get_icbhi_class_names(task_type)
        print(f"Task Info: ICBHI2017 Task-{task_type}")
        if task_type == 11:
            print(f"   Binary Classification: {class_names}")
        else:
            print(f"   4-Class Classification: {class_names}")
    else:
        class_names = get_class_names(task_type)
        print(f"Task Info: SPRSound22_23 Task-{task_type}")
        print(f"   Classes: {class_names}")
    print(f"{'=' * 60}")
    return {
        'sensitivity': avg_se,
        'specificity': avg_sp,
        'average_score': avg_score,
        'harmonic_score': harmonic_score,
        'overall_score': overall_score
    }

def plot_pr_curve(y_true, y_pred_prob, task_type, title, save_path, dataset_type="SPRSound22_23"):
    plt.figure(figsize=(8, 6))
    if dataset_type == "ICBHI2017":
        class_names = get_icbhi_class_names(task_type)
    else:
        class_names = get_class_names(task_type)
    n_classes = len(class_names)
    colors = [
                 '#58614C',
                 '#FFE0C1',
                 '#FFC080',
                 '#FEA040',
                 '#FF7F00',
                 '#FF6100',
                 '#F28080'
             ][:n_classes]
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'font.size': 18,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.0,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5
    })
    plt.gca().set_facecolor('#F5F8FA')
    plt.grid(True, color='white', linestyle='-', linewidth=1.5)
    plt.gca().set_axisbelow(True)
    if task_type == 11:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob[:, 1])
        ap = average_precision_score(y_true, y_pred_prob[:, 1])
        plt.plot(recall, precision, color=colors[0], lw=5, alpha=0.2)
        plt.plot(recall, precision, color=colors[0], lw=2,
                 label=f'AP = {ap:.3f}')
    else:
        for i in range(n_classes):
            if not any(y_true == i):
                continue
            precision, recall, _ = precision_recall_curve(y_true == i, y_pred_prob[:, i])
            ap = average_precision_score(y_true == i, y_pred_prob[:, i])
            plt.plot(recall, precision, color=colors[i], lw=5, alpha=0.2)
            plt.plot(recall, precision, color=colors[i], lw=2,
                     label=f'{class_names[i]} (AP={ap:.3f})')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title(title, fontsize=22, pad=10)
    plt.legend(
        loc='lower left',
        fontsize=18,
        frameon=True,
        edgecolor='none',
        facecolor='white',
        ncol=1
    )
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_roc_curve(y_true, y_pred_prob, task_type, title, save_path, dataset_type="SPRSound22_23"):
    plt.figure(figsize=(8, 6))
    if dataset_type == "ICBHI2017":
        class_names = get_icbhi_class_names(task_type)
    else:
        class_names = get_class_names(task_type)
    n_classes = len(class_names)
    colors = [
                 '#58614C',
                 '#FFE0C1',
                 '#FFC080',
                 '#FEA040',
                 '#FF7F00',
                 '#FF6100',
                 '#F28080'
             ][:n_classes]
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'font.size': 18,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.0,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5
    })
    plt.gca().set_facecolor('#F5F8FA')
    plt.grid(True, color='white', linestyle='-', linewidth=1.5)
    plt.gca().set_axisbelow(True)
    if task_type == 11:
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[0], lw=5, alpha=0.3)
        plt.plot(fpr, tpr, color=colors[0], lw=2,
                 label=f'AUC = {roc_auc:.3f}')
    else:
        for i in range(n_classes):
            if not any(y_true == i):
                continue
            fpr, tpr, _ = roc_curve(y_true == i, y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=5, alpha=0.3)
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                     label=f'{class_names[i]} (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='#8B0000', alpha=0.8, lw=1)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title(title, fontsize=22, pad=10)
    plt.legend(
        loc='lower right',
        fontsize=18,
        frameon=True,
        edgecolor='none',
        facecolor='white',
        ncol=1
    )
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, task_type, title, save_path, dataset_type="SPRSound22_23"):
    plt.figure(figsize=(12, 10))
    if dataset_type == "ICBHI2017":
        class_names = get_icbhi_class_names(task_type)
    else:
        class_names = get_class_names(task_type)
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    row_sums = cm.sum(axis=1)
    cm_percent = np.zeros_like(cm, dtype=float)
    for i in range(len(row_sums)):
        if row_sums[i] != 0:
            cm_percent[i] = (cm[i] / row_sums[i]) * 100
    labels = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_percent[i, j]
            row.append(f'{count}\n({percentage:.1f}%)')
        labels.append(row)
    df_cm = pd.DataFrame(labels,
                         index=class_names,
                         columns=class_names)
    cmap = sns.light_palette("#66c2a5", as_cmap=True)
    sns.heatmap(cm_percent,
                annot=df_cm,
                fmt='',
                cmap=cmap,
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={'size': 18})
    plt.title(title, fontsize=24, pad=20)
    plt.xlabel('Predicted Label', fontsize=22)
    plt.ylabel('True Label', fontsize=22)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_model(model, task_type, test_loader, device):
    model.eval()
    all_preds_gpu = []
    all_labels_gpu = []
    all_probs_gpu = []
    total_batches = len(test_loader)
    print(f"Processing {total_batches} batches...")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            if batch_idx % 20 == 0:
                print(f"   Batch {batch_idx}/{total_batches} ({100 * batch_idx / total_batches:.1f}%)")
            inputs = inputs.float().to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast'):
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_preds_gpu.append(predicted)
            all_labels_gpu.append(labels)
            all_probs_gpu.append(probs)
            del outputs
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
    print("Converting results from GPU to CPU...")
    all_preds = torch.cat(all_preds_gpu, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels_gpu, dim=0).cpu().numpy()
    all_probs = torch.cat(all_probs_gpu, dim=0).cpu().numpy()
    del all_preds_gpu, all_labels_gpu, all_probs_gpu
    torch.cuda.empty_cache()
    print(f"Testing completed. Processed {len(all_preds)} samples.")
    return all_preds, all_labels, all_probs

def visualize_results(preds, labels, probs, task_type, dataset_name, vis_dir, dataset_type="SPRSound22_23"):
    print(f"Creating visualizations for {dataset_name} ({dataset_type})...")
    plot_roc_curve(
        labels, probs, task_type,
        f'ROC Curve - {dataset_name.capitalize()} Test Set ({dataset_type})',
        os.path.join(vis_dir, f'roc_{dataset_name}.png'),
        dataset_type=dataset_type
    )
    plot_pr_curve(
        labels, probs, task_type,
        f'Precision-Recall Curve - {dataset_name.capitalize()} Test Set ({dataset_type})',
        os.path.join(vis_dir, f'pr_{dataset_name}.png'),
        dataset_type=dataset_type
    )
    plot_confusion_matrix(
        labels, preds, task_type,
        f'Confusion Matrix - {dataset_name.capitalize()} Test Set ({dataset_type})',
        os.path.join(vis_dir, f'cm_{dataset_name}.png'),
        dataset_type=dataset_type
    )
    if dataset_type == "ICBHI2017":
        class_names = get_icbhi_class_names(task_type)
    else:
        class_names = get_class_names(task_type)
    metrics = {
        'PR_AUC': [average_precision_score(labels == i, probs[:, i])
                   for i in range(len(class_names))]
    }
    with open(os.path.join(vis_dir, f'metrics_{dataset_name}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Visualizations saved for {dataset_name}")

def create_model(args, num_classes):
    print(f"Creating model: {args.feature_extractor}")
    print(f"   HaB enabled: {args.HaB}")
    print(f"   Number of classes: {num_classes}")
    if args.HaB:
        if args.feature_extractor == 'EfficientNet-B4':
            model = CustomEfficientNetWithLoad.from_pretrained('efficientnet-b4', num_classes=num_classes)
        elif args.feature_extractor == 'ResNet50':
            model = CustomResNet(num_classes=num_classes)
        elif args.feature_extractor == 'DenseNet121':
            model = CustomDenseNet(num_classes=num_classes)
        elif args.feature_extractor == 'ConvNeXt_base':
            model = CustomConvNeXt(num_classes=num_classes)
        elif args.feature_extractor.startswith('MobileViTV2'):
            print(f"   Creating {args.feature_extractor} + HaB model")
            model = get_mobilevitv2_model(args.feature_extractor, num_classes, pretrained=True)
        else:
            raise ValueError(f"Unknown feature extractor with HaB: {args.feature_extractor}")
        return model
    else:
        if args.feature_extractor == 'EfficientNet-B4':
            model = load_efficientnet_model(num_classes=num_classes)
        elif args.feature_extractor == 'ResNet50':
            model = resnet50(pretrained=True)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif args.feature_extractor == 'DenseNet121':
            model = densenet121(pretrained=True)
            model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif args.feature_extractor == 'ConvNeXt_base':
            model = convnext_base(pretrained=True)
            model.features[0][0] = nn.Conv2d(1, 128, kernel_size=4, stride=4)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        elif args.feature_extractor.startswith('MobileViTV2'):
            print(f"   Creating {args.feature_extractor} (without HaB) model")
            import timm
            variant_name = args.feature_extractor.replace('-', '_').lower()
            model = timm.create_model(variant_name, pretrained=True, in_chans=1, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown feature extractor: {args.feature_extractor}")
        return model

def load_model_weights(model, weights_path):
    from torch.serialization import add_safe_globals
    import numpy.core.multiarray as multiarray
    import numpy as np
    add_safe_globals([
        multiarray._reconstruct,
        np.ndarray,
        multiarray.scalar,
        multiarray.dtype
    ])
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k: v for k, v in state_dict.items() if 'lipschitz_const' not in k}
        model.load_state_dict(new_state_dict, strict=False)
        epoch = checkpoint.get('epoch', 0)
        return model, epoch
    except Exception as e:
        print(f"First attempt failed, trying alternative loading method...")
        try:
            checkpoint = torch.load(weights_path, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {k: v for k, v in state_dict.items() if 'lipschitz_const' not in k}
            model.load_state_dict(new_state_dict, strict=False)
            epoch = checkpoint.get('epoch', 0)
            return model, epoch
        except Exception as e2:
            print(f"Both loading attempts failed.")
            print(f"First error: {e}")
            print(f"Second error: {e2}")
            raise e2

def create_ensemble_prediction(all_fold_probs, labels, task_type, vis_dir, test_set_name, dataset_type="SPRSound22_23"):
    print(f"\n{'=' * 60}")
    print(f"CREATING ENSEMBLE PREDICTION FOR {test_set_name.upper()}")
    print(f"Dataset: {dataset_type}")
    print(f"{'=' * 60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(all_fold_probs) > 0 and len(all_fold_probs[0]) > 1000:
        fold_probs_gpu = [torch.tensor(probs, device=device) for probs in all_fold_probs]
        ensemble_probs_gpu = torch.mean(torch.stack(fold_probs_gpu), dim=0)
        ensemble_probs = ensemble_probs_gpu.cpu().numpy()
        del fold_probs_gpu, ensemble_probs_gpu
        torch.cuda.empty_cache()
    else:
        ensemble_probs = np.mean(all_fold_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_metrics = evaluate_and_print_metrics(
        ensemble_preds, labels, task_type, f"Ensemble {test_set_name}", dataset_type
    )
    plot_confusion_matrix(
        labels, ensemble_preds, task_type,
        f'Confusion Matrix - Ensemble Model - {test_set_name} ({dataset_type})',
        os.path.join(vis_dir, f'confusion_matrix_ensemble_{test_set_name}.png'),
        dataset_type=dataset_type
    )
    plot_roc_curve(
        labels, ensemble_probs, task_type,
        f'ROC Curve - Ensemble Model - {test_set_name} ({dataset_type})',
        os.path.join(vis_dir, f'roc_curve_ensemble_{test_set_name}.png'),
        dataset_type=dataset_type
    )
    plot_pr_curve(
        labels, ensemble_probs, task_type,
        f'Precision-Recall Curve - Ensemble Model - {test_set_name} ({dataset_type})',
        os.path.join(vis_dir, f'pr_curve_ensemble_{test_set_name}.png'),
        dataset_type=dataset_type
    )
    print(f"Ensemble prediction completed for {test_set_name}")
    return ensemble_preds, ensemble_probs, ensemble_metrics

def compile_results_table(all_test_sets_results, test_set_names, available_folds, has_final_model):
    headers = ["Dataset", "Model", "Sensitivity", "Specificity", "Average Score",
               "Harmonic Score", "Overall Score"]
    table_data = []
    display_order = ['test_2022_intra', 'test_2022_inter', 'test_2022_combined', 'test_2023', 'icbhi_test']
    ordered_test_sets = [name for name in display_order if name in test_set_names]
    for name in test_set_names:
        if name not in ordered_test_sets:
            ordered_test_sets.append(name)
    for test_set_name in ordered_test_sets:
        results = all_test_sets_results.get(test_set_name, {})
        for fold in sorted(available_folds):
            fold_results = results.get(f'fold_{fold}', None)
            if fold_results:
                fold_metrics = fold_results.get('metrics', {})
                table_data.append([
                    test_set_name,
                    f"Fold {fold}",
                    f"{fold_metrics.get('sensitivity', 0):.4f}",
                    f"{fold_metrics.get('specificity', 0):.4f}",
                    f"{fold_metrics.get('average_score', 0):.4f}",
                    f"{fold_metrics.get('harmonic_score', 0):.4f}",
                    f"{fold_metrics.get('overall_score', 0):.4f}"
                ])
        if len(available_folds) > 1:
            avg_metrics = results.get('avg_metrics', None)
            if avg_metrics:
                table_data.append([
                    test_set_name,
                    "Average Folds",
                    f"{avg_metrics.get('sensitivity', 0):.4f}",
                    f"{avg_metrics.get('specificity', 0):.4f}",
                    f"{avg_metrics.get('average_score', 0):.4f}",
                    f"{avg_metrics.get('harmonic_score', 0):.4f}",
                    f"{avg_metrics.get('overall_score', 0):.4f}"
                ])
        if has_final_model:
            final_model_results = results.get('final_model', None)
            if final_model_results:
                final_metrics = final_model_results.get('metrics', {})
                table_data.append([
                    test_set_name,
                    "Final Model",
                    f"{final_metrics.get('sensitivity', 0):.4f}",
                    f"{final_metrics.get('specificity', 0):.4f}",
                    f"{final_metrics.get('average_score', 0):.4f}",
                    f"{final_metrics.get('harmonic_score', 0):.4f}",
                    f"{final_metrics.get('overall_score', 0):.4f}"
                ])
        if len(available_folds) > 1:
            ensemble_metrics = results.get('ensemble_metrics', None)
            if ensemble_metrics:
                table_data.append([
                    test_set_name,
                    "Ensemble",
                    f"{ensemble_metrics.get('sensitivity', 0):.4f}",
                    f"{ensemble_metrics.get('specificity', 0):.4f}",
                    f"{ensemble_metrics.get('average_score', 0):.4f}",
                    f"{ensemble_metrics.get('harmonic_score', 0):.4f}",
                    f"{ensemble_metrics.get('overall_score', 0):.4f}"
                ])
        for key in results.keys():
            if key.startswith('other_'):
                other_model_results = results[key]
                if 'metrics' in other_model_results:
                    model_name = key.replace('other_', '')
                    other_metrics = other_model_results.get('metrics', {})
                    table_data.append([
                        test_set_name,
                        model_name,
                        f"{other_metrics.get('sensitivity', 0):.4f}",
                        f"{other_metrics.get('specificity', 0):.4f}",
                        f"{other_metrics.get('average_score', 0):.4f}",
                        f"{other_metrics.get('harmonic_score', 0):.4f}",
                        f"{other_metrics.get('overall_score', 0):.4f}"
                    ])
        table_data.append([''] * len(headers))
    if table_data:
        table_data.pop()
    return headers, table_data

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    return obj

def main():
    parser = argparse.ArgumentParser(description='Test Available Models - GPU Optimized for SPRSound22_23 and ICBHI2017')
    parser.add_argument("--task_type", required=True, type=int, choices=[11, 12, 21, 22])
    parser.add_argument("--exp_dir", required=True, type=str,
                        help='Experiment directory containing trained models')
    parser.add_argument("--dataset_type", type=str, default=None,
                        choices=["SPRSound22_23", "ICBHI2017"],
                        help="Dataset type to use. If not specified, will try to infer from exp_dir")
    parser.add_argument("--test_2022_wav_path",
                        default='/home/p2412918/Lung_sound_detection/datasets/test/test_wav',
                        type=str, help='Path to 2022 test WAV files')
    parser.add_argument("--test_2022_json_path",
                        default='/home/p2412918/Lung_sound_detection/datasets/test/test_json',
                        type=str, help='Path to 2022 test JSON files')
    parser.add_argument("--test_2023_wav_path",
                        default='/home/p2412918/Lung_sound_detection/SPRSound22_23/restore_valid_classification_wav',
                        type=str, help='Path to 2023 test WAV files')
    parser.add_argument("--test_2023_json_path",
                        default='/home/p2412918/Lung_sound_detection/SPRSound22_23/valid_classification_json',
                        type=str, help='Path to 2023 test JSON files')
    parser.add_argument("--icbhi_data_dir",
                        default='/home/p2412918/Benchmark_Codebase_Construction/datasets/datasets/ICBHI_final_database',
                        type=str, help='Path to ICBHI2017 data directory')
    parser.add_argument("--icbhi_split_file",
                        default='/home/p2412918/Benchmark_Codebase_Construction/datasets/datasets/ICBHI_final_database/meta_information/ICBHI_challenge_train_test.txt',
                        type=str, help='Path to ICBHI2017 train/test split file')
    parser.add_argument("--feature_type", default='log-mel', type=str,
                        choices=['MFCC', 'log-mel', 'mel', 'STFT'])
    parser.add_argument("--feature_extractor", default='EfficientNet-B4', type=str,
                        choices=['EfficientNet-B4', 'ConvNeXt_base', 'ResNet50', 'DenseNet121',
                                 'MobileViTV2-050', 'MobileViTV2-075', 'MobileViTV2-100',
                                 'MobileViTV2-125', 'MobileViTV2-150', 'MobileViTV2-175', 'MobileViTV2-200'])
    parser.add_argument("--HaB", default=True, type=bool,
                        help='Whether to use HaB models')
    parser.add_argument("--batch_size", default=128, type=int,
                        help='Batch size for testing (larger for better GPU utilization)')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=8, type=int,
                        help='Number of data loading workers')
    parser.add_argument("--pin_memory", default=True, type=bool,
                        help='Pin memory for faster GPU transfer')
    parser.add_argument("--prefetch_factor", default=4, type=int,
                        help='Number of sample loaded in advance by each worker')
    args = parser.parse_args()
    if args.dataset_type is None:
        args.dataset_type = determine_dataset_type_from_path(args.exp_dir)
        print(f"Inferred dataset type: {args.dataset_type}")
    if args.dataset_type == "ICBHI2017":
        if args.task_type not in [11, 12]:
            print(f"Error: ICBHI2017 only supports task_type 11 (binary) or 12 (4-class), "
                  f"but got {args.task_type}")
            return
    global dataset_type
    dataset_type = args.dataset_type
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"CUDA optimizations enabled. Available GPUs: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    set_seed(args.seed)
    print(f"\nScanning for available models in: {args.exp_dir}")
    fold_models, final_model, other_models = scan_available_models(args.exp_dir)
    if not fold_models and not final_model and not other_models:
        print("\n" + "=" * 60)
        print("ERROR: No model files (.pth) found!")
        print("=" * 60)
        print("Please check:")
        print("1. The experiment directory path is correct")
        print("2. The training process completed successfully")
        print("3. Model files were saved properly")
        print("4. You have the correct permissions to access the files")
        print("=" * 60)
        return
    print(f"\n" + "=" * 60)
    print("MODELS FOUND - STARTING TESTING")
    print("=" * 60)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.dataset_type == "ICBHI2017":
        base_dir = 'icbhi_test_results'
    else:
        base_dir = 'merged_test_SPRSound_visualization_results'
    os.makedirs(base_dir, exist_ok=True)
    exp_name = f'task_{args.task_type}_{args.feature_extractor}_{args.feature_type}_{args.dataset_type}'
    if args.HaB:
        exp_name += '_wHaB'
    vis_dir = os.path.join(base_dir, f'{exp_name}_{timestamp}')
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Created visualization directory: {vis_dir}")
    config = vars(args)
    config['timestamp'] = timestamp
    config['vis_dir'] = vis_dir
    config['available_models'] = {
        'fold_models': {k: os.path.basename(v) for k, v in fold_models.items()},
        'final_model': os.path.basename(final_model) if final_model else None,
        'other_models': [os.path.basename(f) for f in other_models]
    }
    with open(os.path.join(vis_dir, 'test_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\nLoading {args.dataset_type} test datasets...")
    test_datasets = {}
    if args.dataset_type == "SPRSound22_23":
        test_2022_intra = None
        test_2022_inter = None
        try:
            test_2022_intra = SPRSoundDataset(
                args.test_2022_wav_path, args.test_2022_json_path, args.task_type,
                mode='test', valid_type='intra', feature_type=args.feature_type
            )
            test_datasets['test_2022_intra'] = test_2022_intra
            print(f"Loaded test_2022_intra dataset: {len(test_datasets['test_2022_intra'])} samples")
            test_2022_inter = SPRSoundDataset(
                args.test_2022_wav_path, args.test_2022_json_path, args.task_type,
                mode='test', valid_type='inter', feature_type=args.feature_type
            )
            test_datasets['test_2022_inter'] = test_2022_inter
            print(f"Loaded test_2022_inter dataset: {len(test_datasets['test_2022_inter'])} samples")
        except Exception as e:
            print(f"Error loading 2022 datasets: {e}")
        if test_2022_intra is not None and test_2022_inter is not None:
            try:
                from torch.utils.data import ConcatDataset
                test_2022_combined = ConcatDataset([test_2022_intra, test_2022_inter])
                test_datasets['test_2022_combined'] = test_2022_combined
                print(f"Created test_2022_combined dataset: {len(test_2022_combined)} samples")
                print(f"   - Intra samples: {len(test_2022_intra)}")
                print(f"   - Inter samples: {len(test_2022_inter)}")
            except Exception as e:
                print(f"Error creating combined 2022 dataset: {e}")
        try:
            test_datasets['test_2023'] = SPRSoundDataset(
                args.test_2023_wav_path, args.test_2023_json_path, args.task_type,
                mode='valid', year=2023, feature_type=args.feature_type
            )
            print(f"Loaded test_2023 dataset: {len(test_datasets['test_2023'])} samples")
        except Exception as e:
            print(f"Error loading 2023 dataset: {e}")
        test_collate_fn = collate_fn_valid
    elif args.dataset_type == "ICBHI2017":
        print(f"Loading ICBHI2017 test dataset for task {args.task_type}")
        icbhi_test_datasets = load_icbhi_test_datasets(
            args.icbhi_data_dir,
            args.icbhi_split_file,
            args.task_type,
            args.feature_type
        )
        test_datasets.update(icbhi_test_datasets)
        test_collate_fn = collate_fn_icbhi_valid
    if not test_datasets:
        print(f"Error: No test datasets found for {args.dataset_type}. Please check the data paths.")
        return
    first_dataset = next(iter(test_datasets.values()))
    if args.dataset_type == "SPRSound22_23":
        num_classes = len(set(first_dataset.get_task_dict().values()))
    else:
        num_classes = len(set(first_dataset.get_task_dict().values()))
    print(f"Number of classes: {num_classes}")
    all_test_sets_results = {test_name: {} for test_name in test_datasets.keys()}
    fold_probs_by_dataset = {test_name: [] for test_name in test_datasets.keys()}
    if fold_models:
        print(f"\nTesting {len(fold_models)} fold models...")
        for fold_num in sorted(fold_models.keys()):
            fold_model_path = fold_models[fold_num]
            print(f"\n{'=' * 50}")
            print(f"TESTING FOLD {fold_num} MODEL")
            print(f"{'=' * 50}")
            print(f"Model path: {fold_model_path}")
            model = create_model(args, num_classes)
            model, epoch = load_model_weights(model, fold_model_path)
            model.to(device)
            model.eval()
            if hasattr(model, 'half') and torch.cuda.is_available():
                try:
                    model.half()
                    print(f"   Using half precision for fold {fold_num}")
                except:
                    print(f"   Half precision not supported for fold {fold_num}, using float32")
            for test_set_name, test_dataset in test_datasets.items():
                print(f"\nEvaluating fold {fold_num} on {test_set_name} ({args.dataset_type})")
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=test_collate_fn,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    prefetch_factor=args.prefetch_factor,
                    persistent_workers=True
                )
                preds, labels, probs = test_model(model, args.task_type, test_loader, device)
                metrics = evaluate_and_print_metrics(preds, labels, args.task_type,
                                                     f"Fold {fold_num} {test_set_name}", args.dataset_type)
                all_test_sets_results[test_set_name][f'fold_{fold_num}'] = {
                    'metrics': metrics,
                    'preds': preds.tolist(),
                    'labels': labels.tolist(),
                    'saved_epoch': epoch
                }
                fold_probs_by_dataset[test_set_name].append(probs)
                visualize_results(preds, labels, probs, args.task_type,
                                  f"fold_{fold_num}_{test_set_name}", vis_dir, args.dataset_type)
            del model
            torch.cuda.empty_cache()
    if final_model:
        print(f"\n{'=' * 50}")
        print(f"TESTING FINAL MODEL")
        print(f"{'=' * 50}")
        print(f"Model path: {final_model}")
        model = create_model(args, num_classes)
        model, epoch = load_model_weights(model, final_model)
        model.to(device)
        model.eval()
        if hasattr(model, 'half') and torch.cuda.is_available():
            try:
                model.half()
                print(f"   Using half precision for final model")
            except:
                print(f"   Half precision not supported for final model, using float32")
        for test_set_name, test_dataset in test_datasets.items():
            print(f"\nEvaluating final model on {test_set_name} ({args.dataset_type})")
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=test_collate_fn,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=True
            )
            preds, labels, probs = test_model(model, args.task_type, test_loader, device)
            metrics = evaluate_and_print_metrics(preds, labels, args.task_type,
                                                 f"Final Model {test_set_name}", args.dataset_type)
            all_test_sets_results[test_set_name]['final_model'] = {
                'metrics': metrics,
                'preds': preds.tolist(),
                'labels': labels.tolist(),
                'saved_epoch': epoch
            }
            visualize_results(preds, labels, probs, args.task_type,
                              f"final_{test_set_name}", vis_dir, args.dataset_type)
        del model
        torch.cuda.empty_cache()
    if other_models:
        print(f"\nTesting {len(other_models)} other models...")
        for i, other_model_path in enumerate(other_models):
            model_name = os.path.basename(other_model_path).replace('.pth', '')
            print(f"\n{'=' * 50}")
            print(f"TESTING OTHER MODEL: {model_name.upper()}")
            print(f"{'=' * 50}")
            print(f"Model path: {other_model_path}")
            model = create_model(args, num_classes)
            model, epoch = load_model_weights(model, other_model_path)
            model.to(device)
            model.eval()
            if hasattr(model, 'half') and torch.cuda.is_available():
                try:
                    model.half()
                    print(f"   Using half precision for {model_name}")
                except:
                    print(f"   Half precision not supported for {model_name}, using float32")
            for test_set_name, test_dataset in test_datasets.items():
                print(f"\nEvaluating {model_name} on {test_set_name} ({args.dataset_type})")
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=test_collate_fn,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    prefetch_factor=args.prefetch_factor,
                    persistent_workers=True
                )
                preds, labels, probs = test_model(model, args.task_type, test_loader, device)
                metrics = evaluate_and_print_metrics(preds, labels, args.task_type,
                                                     f"{model_name} {test_set_name}", args.dataset_type)
                all_test_sets_results[test_set_name][f'other_{model_name}'] = {
                    'metrics': metrics,
                    'preds': preds.tolist(),
                    'labels': labels.tolist(),
                    'saved_epoch': epoch
                }
                visualize_results(preds, labels, probs, args.task_type,
                                  f"{model_name}_{test_set_name}", vis_dir, args.dataset_type)
            del model
            torch.cuda.empty_cache()
    if len(fold_models) > 1:
        print(f"\nCreating ensemble predictions...")
        for test_set_name, fold_probs in fold_probs_by_dataset.items():
            if len(fold_probs) > 1:
                print(f"\n{'=' * 50}")
                print(f"CREATING ENSEMBLE FOR {test_set_name.upper()}")
                print(f"{'=' * 50}")
                labels = np.array(all_test_sets_results[test_set_name][f'fold_{min(fold_models.keys())}']['labels'])
                ensemble_preds, ensemble_probs, ensemble_metrics = create_ensemble_prediction(
                    fold_probs, labels, args.task_type, vis_dir, test_set_name, args.dataset_type
                )
                all_test_sets_results[test_set_name]['ensemble_metrics'] = ensemble_metrics
    available_folds = list(fold_models.keys()) if fold_models else []
    if len(available_folds) > 1:
        print(f"\nComputing average metrics across folds...")
        for test_set_name, results in all_test_sets_results.items():
            fold_metrics = []
            for fold_num in available_folds:
                fold_key = f'fold_{fold_num}'
                if fold_key in results and 'metrics' in results[fold_key]:
                    fold_metrics.append(results[fold_key]['metrics'])
            if fold_metrics:
                avg_metrics = {
                    'sensitivity': np.mean([m['sensitivity'] for m in fold_metrics]),
                    'specificity': np.mean([m['specificity'] for m in fold_metrics]),
                    'average_score': np.mean([m['average_score'] for m in fold_metrics]),
                    'harmonic_score': np.mean([m['harmonic_score'] for m in fold_metrics]),
                    'overall_score': np.mean([m['overall_score'] for m in fold_metrics])
                }
                results['avg_metrics'] = avg_metrics
                print(f"\n{'=' * 60}")
                print(f"AVERAGE METRICS ACROSS ALL FOLDS FOR {test_set_name.upper()}")
                print(f"Dataset: {args.dataset_type}")
                print(f"{'=' * 60}")
                print(f"Average Performance:")
                print(f"   Sensitivity: {avg_metrics['sensitivity']:.4f}")
                print(f"   Specificity: {avg_metrics['specificity']:.4f}")
                print(f"   Average Score: {avg_metrics['average_score']:.4f}")
                print(f"   Harmonic Score: {avg_metrics['harmonic_score']:.4f}")
                print(f"   Overall Score: {avg_metrics['overall_score']:.4f}")
    print("\n\n")
    print("=" * 100)
    print(f"SUMMARY OF ALL TEST RESULTS ({args.dataset_type})")
    print("=" * 100)
    test_set_names = list(test_datasets.keys())
    headers, table_data = compile_results_table(all_test_sets_results, test_set_names,
                                                available_folds, final_model is not None)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    with open(os.path.join(vis_dir, 'results_summary.txt'), 'w') as f:
        f.write(f"SUMMARY OF ALL TEST RESULTS ({args.dataset_type})\n")
        f.write("=" * 100 + "\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
    serializable_results = {}
    for key, value in all_test_sets_results.items():
        if isinstance(value, dict):
            serializable_results[key] = {k: convert_to_serializable(v) for k, v in value.items()}
        else:
            serializable_results[key] = convert_to_serializable(value)
    results_filename = f'{args.dataset_type}_task_type_{args.task_type}_complete_test_results.json'
    with open(os.path.join(vis_dir, results_filename), 'w') as f:
        json.dump(serializable_results, f, indent=4)
    print("\n" + "=" * 100)
    print("TESTING COMPLETED SUCCESSFULLY!")
    print("=" * 100)
    print(f"Results saved in: {vis_dir}")
    print(f"Dataset tested: {args.dataset_type}")
    print(f"Task type: {args.task_type}")
    print(f"Model architecture: {args.feature_extractor}")
    print(f"Tested models summary:")
    print(f"   - Fold models: {len(fold_models)}")
    print(f"   - Final model: {'Yes' if final_model else 'No'}")
    print(f"   - Other models: {len(other_models)}")
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"Final GPU memory usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print("=" * 100)

if __name__ == "__main__":
    main()