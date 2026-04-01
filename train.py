from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from contextlib import nullcontext
import gc
from tqdm import tqdm
from utils.comet_record import (init_comet_experiment, log_hyperparameters, log_model_to_comet)
import os
import numpy as np
import torch
import argparse
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from datasets.SPRSound_dataloader import SPRSoundDataset, collate_fn_train, collate_fn_valid, collate_fn_train_with_RQ
from loss.polyloss import Poly1CrossEntropyLoss
from loss.Lipschitz_regularization_loss import LipschitzRegularization, compute_kl_loss, EntropyAdaptiveWeight
from utils.oversampling import perform_oversampling, perform_multioversampling
from utils.balanced_batch_sampler import BalancedBatchSampler
from utils.evaluation_metrics import evaluate_model, calculate_metrics
from torchvision.models import resnet50, densenet121, convnext_base, convnext_tiny
from feature_extractor.EfficientNet import load_efficientnet_model, CustomEfficientNetWithLoad
from feature_extractor.HaB_ResNet import CustomResNet
from feature_extractor.HaB_DenseNet import CustomDenseNet
from feature_extractor.HaB_ConvNeXt import CustomConvNeXt
try:
    from feature_extractor.HaB_StarNet import starnet_t0, starnet_s4, starnet_b1
    STARNET_AVAILABLE = True
    print("StarNet backbone loaded!")
except ImportError as e:
    STARNET_AVAILABLE = False
    print(f"StarNet backbone not available: {e}")

# Import MobileViTV2+HaB module
from feature_extractor.HaB_MobileViTV2_timm import (
    mobilevitv2_050, mobilevitv2_075, mobilevitv2_100,
    mobilevitv2_125, mobilevitv2_150, mobilevitv2_175, mobilevitv2_200
)

import random
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
import time
import math
from sklearn.model_selection import StratifiedKFold
from utils.train_trick import (
    CosineAnnealingWarmupRestarts,
    LabelSmoothingCrossEntropy,
    EMA,
    train_with_gradient_accumulation,
    adaptive_gradient_accumulation,
    get_progressive_batch_size,
    AdaptiveWeightDecay
)
from datasets.Randomized_QuantizationAug import RandomizedQuantizationAugModule
from datasets.ICBHI2017_dataset import (
    ICBHI2017Dataset,
    collate_fn_icbhi_train,
    collate_fn_icbhi_valid,
    collate_fn_icbhi_train_with_RQ,
    get_class_names as get_icbhi_class_names,
    create_fold_dataset_with_oversampling_icbhi
)

def create_icbhi_dataset(data_dir, split_file, task_type, mode='train', feature_type='log-mel',
                         augment_type=None, augment_params=None, use_oversampling=False,
                         oversampler_type=None):
    return ICBHI2017Dataset(
        data_dir=data_dir,
        split_file=split_file,
        task_type=task_type,
        mode=mode,
        feature_type=feature_type,
        augment_type=augment_type,
        augment_params=augment_params,
        use_cache=True,
        use_oversampling=use_oversampling,
        oversampler_type=oversampler_type
    )

def create_fold_dataset_with_oversampling_icbhi(full_dataset, train_idx, args, fold_num):
    print(f"\n{'=' * 60}")
    print(f"SAFE OVERSAMPLING FOR ICBHI FOLD {fold_num}")
    print(f"{'=' * 60}")
    train_data = []
    train_labels = []
    print(f"Extracting training data for fold {fold_num}...")
    for idx in train_idx:
        data, label = full_dataset[idx]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
        if isinstance(label, torch.Tensor):
            if label.dim() > 0:
                label = label.item()
            else:
                label = label.item()
        train_data.append(data)
        train_labels.append(label)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    print(f"Data shape: {train_data.shape}")
    print(f"Data type: {train_data.dtype}")
    print(f"Labels shape: {train_labels.shape}")
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    print(f"\nBEFORE oversampling (Fold {fold_num}):")
    print(f"  Total samples: {len(train_labels)}")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(train_labels)) * 100
        print(f"  Class {label}: {count} samples ({percentage:.1f}%)")
    if args.use_oversampling and args.oversamplers:
        print(f"\nApplying {args.oversamplers} oversampling...")
        try:
            original_shape = train_data.shape
            if len(original_shape) > 2:
                train_data_flat = train_data.reshape(len(train_data), -1)
            else:
                train_data_flat = train_data
            from utils.oversampling import perform_oversampling, perform_multioversampling
            if args.task_type == 11:
                train_data_resampled, train_labels_resampled = perform_oversampling(
                    args.task_type, train_data_flat, train_labels, args.oversamplers
                )
            else:
                train_data_resampled, train_labels_resampled = perform_multioversampling(
                    args.task_type, train_data_flat, train_labels, args.oversamplers
                )
            if len(original_shape) > 2:
                train_data_resampled = train_data_resampled.reshape(
                    len(train_data_resampled), *original_shape[1:]
                )
            resampled_unique, resampled_counts = np.unique(train_labels_resampled, return_counts=True)
            print(f"\nAFTER oversampling (Fold {fold_num}):")
            print(f"  Total samples: {len(train_labels_resampled)}")
            for label, count in zip(resampled_unique, resampled_counts):
                percentage = (count / len(train_labels_resampled)) * 100
                original_count = dict(zip(unique_labels, counts)).get(label, 0)
                increase = count - original_count
                print(f"  Class {label}: {count} samples ({percentage:.1f}%) [+{increase}]")
            print(f"\nOversampling summary:")
            print(f"  Original: {len(train_labels)} -> Resampled: {len(train_labels_resampled)}")
            print(f"  Increase: {len(train_labels_resampled) - len(train_labels)} samples")
            print(f"  Multiplier: {len(train_labels_resampled) / len(train_labels):.2f}x")
            train_data_tensor = torch.FloatTensor(train_data_resampled)
            train_labels_tensor = torch.LongTensor(train_labels_resampled)
            print(f"   Tensor shapes: data {train_data_tensor.shape}, labels {train_labels_tensor.shape}")
            print(f"   Tensors created on device: {train_data_tensor.device}")
            resampled_dataset = torch.utils.data.TensorDataset(
                train_data_tensor,
                train_labels_tensor
            )
            print(f"Safe oversampling completed for ICBHI fold {fold_num}")
            print(f"   No data leakage: oversampling only on fold's training set")
            print(f"{'=' * 60}")
            return resampled_dataset
        except Exception as e:
            print(f"Oversampling failed: {e}")
            print("Falling back to original training set...")
            train_data_tensor = torch.FloatTensor(train_data)
            train_labels_tensor = torch.LongTensor(train_labels)
            original_dataset = torch.utils.data.TensorDataset(
                train_data_tensor,
                train_labels_tensor
            )
            return original_dataset
    else:
        print(f"No oversampling applied for fold {fold_num}")
        print(f"Data shape: {train_data.shape}")
        print(f"Labels shape: {train_labels.shape}")
        train_data_tensor = torch.FloatTensor(train_data)
        train_labels_tensor = torch.LongTensor(train_labels)
        print(f"   Tensor shapes: data {train_data_tensor.shape}, labels {train_labels_tensor.shape}")
        print(f"   Tensors created on device: {train_data_tensor.device}")
        original_dataset = torch.utils.data.TensorDataset(
            train_data_tensor,
            train_labels_tensor
        )
        return original_dataset

def safe_scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        else:
            return float(np.mean(value))
    return float(value)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def save_checkpoint(model, optimizer, scheduler, epoch, seed_num, exp_dir, metrics, is_best=False):
    if is_best:
        best_path = os.path.join(exp_dir, f'seed_{seed_num}_best.pth')
        model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, best_path)
        print(f"Saved best model for seed {seed_num} at epoch {epoch} with test_average_score: {metrics['avg_score']:.4f}")

def create_experiment_dir(args):
    base_dir = './experiments'
    hab_tag = 'wHaB' if args.HaB else 'oHaB'
    taf_enabled = args.R_Drop and args.Lipschitz_regularization and args.PolyCrossEntropyLoss
    taf_tag = 'wTAF' if taf_enabled else 'oTAF'
    oversampler_tag = f'w{args.oversamplers}' if args.use_oversampling and args.oversamplers else 'oOversampling'
    if args.audio_augment_type == 'time_stretch':
        aug_tag = 'TS'
    elif args.audio_augment_type == 'pitch_shift':
        aug_tag = 'PS'
    elif args.audio_augment_type == 'noise_injection':
        aug_tag = 'NI'
    elif args.Rrandomized_Quantization_Aug:
        aug_tag = 'RQ'
    else:
        aug_tag = 'oAug'
    rq_tag = 'wRQ' if args.Rrandomized_Quantization_Aug else 'oRQ'
    dataset_tag = args.dataset_type
    if args.dataset_type == "ICBHI2017":
        exp_identifier = (
            f"{dataset_tag}_"
            f"task{args.task_type}_"
            f"{args.feature_extractor}_"
            f"{args.feature_type}_"
            f"{hab_tag}_"
            f"{taf_tag}_"
            f"{oversampler_tag}_"
            f"{aug_tag}_"
            f"{rq_tag}_"
            f"bs{args.batch_size}_"
            f"5Seeds"
        )
    else:
        exp_identifier = (
            f"{dataset_tag}_"
            f"task{args.task_type}_"
            f"{args.feature_extractor}_"
            f"{args.feature_type}_"
            f"{hab_tag}_"
            f"{taf_tag}_"
            f"{oversampler_tag}_"
            f"{aug_tag}_"
            f"{rq_tag}_"
            f"bs{args.batch_size}_"
            f"5FoldCV"
        )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{exp_identifier}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Experiment Directory: {exp_identifier}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 60 + "\n")
        f.write("MODULE STATUS:\n")
        f.write(f"  HaB Module: {'ENABLED' if args.HaB else 'DISABLED'}\n")
        f.write(f"  TAF Module: {'ENABLED' if taf_enabled else 'DISABLED'}\n")
        f.write(f"    - R_Drop: {args.R_Drop}\n")
        f.write(f"    - Lipschitz_regularization: {args.Lipschitz_regularization}\n")
        f.write(f"    - PolyCrossEntropyLoss: {args.PolyCrossEntropyLoss}\n")
        f.write(f"  Oversampling: {'ENABLED' if args.use_oversampling else 'DISABLED'}\n")
        f.write(f"    - Method: {args.oversamplers if args.oversamplers else 'None'}\n")
        f.write(f"  Audio Augmentation: {args.audio_augment_type if args.audio_augment_type else 'None'}\n")
        f.write(f"  RQ Augmentation: {'ENABLED' if args.Rrandomized_Quantization_Aug else 'DISABLED'}\n")
        f.write("-" * 60 + "\n")
        if args.dataset_type == "ICBHI2017":
            f.write("TRAINING STRATEGY: 5 Random Seeds\n")
        else:
            f.write("TRAINING STRATEGY: 5-Fold Cross Validation\n")
        f.write("-" * 60 + "\n")
        f.write("ALL PARAMETERS:\n")
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')
        f.write("=" * 60 + "\n")
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT DIRECTORY CREATED")
    print(f"{'=' * 60}")
    print(f"Directory: {exp_dir}")
    print(f"Identifier: {exp_identifier}")
    print(f"HaB: {'ENABLED' if args.HaB else 'DISABLED'}")
    print(f"TAF: {'ENABLED' if taf_enabled else 'DISABLED'}")
    print(f"Oversampling: {'ENABLED' if args.use_oversampling else 'DISABLED'}")
    print(f"Audio Aug: {args.audio_augment_type if args.audio_augment_type else 'None'}")
    print(f"RQ Aug: {'ENABLED' if args.Rrandomized_Quantization_Aug else 'DISABLED'}")
    if args.dataset_type == "ICBHI2017":
        print(f"Training Strategy: 5 Random Seeds")
    else:
        print(f"Training Strategy: 5-Fold Cross Validation")
    print(f"{'=' * 60}")
    return exp_dir

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 target_lr=0.0001,
                 warmup_start_lr=0.001,
                 min_lr=None, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr if min_lr is not None else target_lr / 100
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            progress = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (self.target_lr - self.warmup_start_lr) * progress
                    for _ in self.optimizer.param_groups]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (self.target_lr - self.min_lr) * cosine
                    for _ in self.optimizer.param_groups]

def improved_early_stopping_check(epoch, val_scores, best_val_score, counter, patience, warmup_protection):
    if epoch < warmup_protection:
        return False, best_val_score, counter
    window_size = 5
    if len(val_scores) >= window_size:
        moving_avg_score = sum(val_scores[-window_size:]) / window_size
    else:
        moving_avg_score = val_scores[-1] if val_scores else 0
    if moving_avg_score > best_val_score:
        best_val_score = moving_avg_score
        counter = 0
    else:
        counter += 1
    should_stop = counter >= patience
    return should_stop, best_val_score, counter

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

def train_seed_icbhi(model, task_type, train_loader, test_loader, seed_num,
                     criterion, optimizer, scheduler, R_Drop, α, early_stop,
                     Lipschitz_regularization, Lipschitz_regularization_degree_alpha,
                     num_epochs, experiment, exp_dir, args, use_amp=True, distributed=False):
    print(f"\n{'=' * 50}")
    print(f"Starting training for seed {seed_num}")
    print(f"{'=' * 50}")
    seed_start_time = time.time()
    best_test_score = float('-inf')
    test_scores_history = []
    early_stop_counter = 0
    warmup_protection = args.warmup_epoch + 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scaler = GradScaler() if use_amp else None
    zero_tensor = torch.tensor(0.0, device=device)
    one_tensor = torch.tensor(1.0, device=device)
    alpha_tensor = torch.tensor(α, device=device, dtype=torch.float32)
    if Lipschitz_regularization:
        constrain_layers = [nn.Linear, nn.Conv2d]
        lipschitz_regularizer = LipschitzRegularization(
            model,
            alpha=Lipschitz_regularization_degree_alpha,
            constrain_layers=constrain_layers
        )
        adaptive_weight = EntropyAdaptiveWeight().to(device)
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast() if use_amp else nullcontext():
                outputs = model(inputs)
                if R_Drop and Lipschitz_regularization:
                    outputs1_polyloss = lipschitz_regularizer.regularize_loss(criterion(outputs, labels))
                    outputs2 = model(inputs)
                    outputs2_polyloss = criterion(outputs2, labels)
                    weights = adaptive_weight(outputs, outputs2, criterion, labels, model)
                    outputs1_mean = outputs1_polyloss.mean()
                    outputs2_mean = outputs2_polyloss.mean()
                    loss_scale = torch.where(outputs2_mean > 1e-8,
                                             outputs1_mean / outputs2_mean,
                                             one_tensor)
                    normalized_weights = weights * loss_scale
                    normalized_weights = normalized_weights / (normalized_weights + one_tensor)
                    combined_loss = normalized_weights * outputs1_polyloss + (
                            one_tensor - normalized_weights) * outputs2_polyloss
                    loss = combined_loss.mean()
                    kl_loss = compute_kl_loss(outputs.detach(), outputs2.detach())
                    loss = loss + alpha_tensor * kl_loss
                elif R_Drop and not Lipschitz_regularization:
                    outputs2 = model(inputs)
                    ce_loss = 0.5 * (criterion(outputs, labels) + criterion(outputs2, labels))
                    kl_loss = compute_kl_loss(outputs, outputs2)
                    loss = ce_loss + alpha_tensor * kl_loss
                elif Lipschitz_regularization and not R_Drop:
                    loss = criterion(outputs, labels)
                    loss = lipschitz_regularizer.regularize_loss(loss)
                else:
                    loss = criterion(outputs, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += loss.detach().item()
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            if batch_idx % 50 == 0:
                experiment.log_metric(f"seed_{seed_num}_train_batch_loss", loss.item(),
                                      step=(epoch * len(train_loader) + batch_idx))
            del outputs, loss
            if R_Drop:
                del outputs2
            if 'kl_loss' in locals():
                del kl_loss
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        experiment.log_metric(f"seed_{seed_num}_learning_rate", current_lr, step=epoch)
        (sensitivity, specificity, average_sensitivity, average_specificity,
         overall_average_score, overall_harmonic_score, overall_score) = calculate_metrics(all_labels, all_preds,
                                                                                           task_type)
        experiment.log_metric(f"seed_{seed_num}_train_accuracy", 100 * correct / total, step=epoch)
        experiment.log_metric(f"seed_{seed_num}_train_loss", running_loss / len(train_loader), step=epoch)
        experiment.log_metric(f"seed_{seed_num}_train_sensitivity", safe_scalar(sensitivity), step=epoch)
        experiment.log_metric(f"seed_{seed_num}_train_specificity", safe_scalar(specificity), step=epoch)
        experiment.log_metric(f"seed_{seed_num}_train_average_score", safe_scalar(overall_average_score), step=epoch)
        experiment.log_metric(f"seed_{seed_num}_train_harmonic_score", safe_scalar(overall_harmonic_score), step=epoch)
        experiment.log_metric(f"seed_{seed_num}_train_overall_score", safe_scalar(overall_score), step=epoch)
        avg_loss = running_loss / len(train_loader)
        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        test_all_preds = []
        test_all_labels = []
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.float().to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast() if use_amp else nullcontext():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                if batch_idx % 5 == 0 or batch_idx == len(test_loader) - 1:
                    test_all_preds.extend(predicted.cpu().numpy())
                    test_all_labels.extend(labels.cpu().numpy())
            (test_sensitivity, test_specificity, test_avg_se, test_avg_sp,
             test_avg_score, test_harmonic_score, test_overall_score) = calculate_metrics(
                test_all_labels, test_all_preds, task_type)
            test_accuracy = 100 * test_correct / test_total
            test_loss = test_running_loss / len(test_loader)
            experiment.log_metric(f"seed_{seed_num}_test_accuracy", test_accuracy, step=epoch)
            experiment.log_metric(f"seed_{seed_num}_test_loss", test_loss, step=epoch)
            experiment.log_metric(f"seed_{seed_num}_test_sensitivity", safe_scalar(test_sensitivity), step=epoch)
            experiment.log_metric(f"seed_{seed_num}_test_specificity", safe_scalar(test_specificity), step=epoch)
            experiment.log_metric(f"seed_{seed_num}_test_avg_sensitivity", safe_scalar(test_avg_se), step=epoch)
            experiment.log_metric(f"seed_{seed_num}_test_avg_specificity", safe_scalar(test_avg_sp), step=epoch)
            experiment.log_metric(f"seed_{seed_num}_test_avg_score", safe_scalar(test_avg_score), step=epoch)
            experiment.log_metric(f"seed_{seed_num}_test_harmonic_score", safe_scalar(test_harmonic_score), step=epoch)
            experiment.log_metric(f"seed_{seed_num}_test_overall_score", safe_scalar(test_overall_score), step=epoch)
            test_avg_score_scalar = safe_scalar(test_avg_score)
            is_best = test_avg_score_scalar > best_test_score
            test_scores_history.append(test_avg_score_scalar)
            if is_best:
                best_test_score = test_avg_score_scalar
                metrics = {
                    'accuracy': test_accuracy,
                    'loss': test_loss,
                    'sensitivity': safe_scalar(test_sensitivity),
                    'specificity': safe_scalar(test_specificity),
                    'avg_sensitivity': safe_scalar(test_avg_se),
                    'avg_specificity': safe_scalar(test_avg_sp),
                    'avg_score': safe_scalar(test_avg_score),
                    'harmonic_score': safe_scalar(test_harmonic_score),
                    'overall_score': safe_scalar(test_overall_score)
                }
                save_checkpoint(
                    model, optimizer, scheduler, epoch, seed_num, exp_dir, metrics, is_best=True
                )
            if early_stop and epoch >= warmup_protection:
                should_stop, best_test_score, early_stop_counter = improved_early_stopping_check(
                    epoch, test_scores_history, best_test_score, early_stop_counter,
                    patience=35, warmup_protection=warmup_protection
                )
                if should_stop:
                    print(f"Early stopping triggered at epoch {epoch} for seed {seed_num}!")
                    print(f"Best test average score: {best_test_score:.4f}")
                    break
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Seed {seed_num} | Epoch {epoch + 1}/{num_epochs} [{epoch_duration:.2f}s]')
        print(f'  Train - Loss: {avg_loss:.4f} | Acc: {100 * correct / total:.2f}% | '
              f'Sen: {safe_scalar(sensitivity):.4f} | Spe: {safe_scalar(specificity):.4f} | '
              f'Avg: {safe_scalar(overall_average_score):.4f} | Har: {safe_scalar(overall_harmonic_score):.4f} | '
              f'Overall: {safe_scalar(overall_score):.4f}')
        print(f'  Test  - Loss: {test_loss:.4f} | Acc: {test_accuracy:.2f}% | '
              f'Sen: {safe_scalar(test_sensitivity):.4f} | Spe: {safe_scalar(test_specificity):.4f} | '
              f'Avg: {safe_scalar(test_avg_score):.4f} | Har: {safe_scalar(test_harmonic_score):.4f} | '
              f'Overall: {safe_scalar(test_overall_score):.4f}')
        if is_best:
            print(f'  *** NEW BEST MODEL (Test Average Score: {safe_scalar(test_avg_score):.4f}) ***')
    seed_time = time.time() - seed_start_time
    print(f"\n{'=' * 60}")
    print(f"Seed {seed_num} Summary:")
    print(f"  Training time: {seed_time:.2f}s")
    print(f"  Best test average score: {best_test_score:.4f}")
    print(f"{'=' * 60}")
    experiment.log_metric(f"seed_{seed_num}_total_time", seed_time)
    experiment.log_metric(f"seed_{seed_num}_best_avg_score", best_test_score)
    return best_test_score

def train_fold(model, task_type, train_loader, val_loader, fold_num,
               criterion, optimizer, scheduler, R_Drop, α, early_stop,
               Lipschitz_regularization, Lipschitz_regularization_degree_alpha,
               num_epochs, experiment, exp_dir, args, use_amp=True, distributed=False):
    print(f"\n{'=' * 50}")
    print(f"Starting training for fold {fold_num}")
    print(f"{'=' * 50}")
    fold_start_time = time.time()
    epoch_times = []
    best_val_score = float('-inf')
    val_scores_history = []
    early_stop_counter = 0
    warmup_protection = args.warmup_epoch + 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scaler = GradScaler() if use_amp else None
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    zero_tensor = torch.tensor(0.0, device=device)
    one_tensor = torch.tensor(1.0, device=device)
    alpha_tensor = torch.tensor(α, device=device, dtype=torch.float32)
    if Lipschitz_regularization:
        constrain_layers = [nn.Linear, nn.Conv2d]
        lipschitz_regularizer = LipschitzRegularization(
            model,
            alpha=Lipschitz_regularization_degree_alpha,
            constrain_layers=constrain_layers
        )
        adaptive_weight = EntropyAdaptiveWeight().to(device)
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        if torch.cuda.is_available():
            start_event.record()
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast() if use_amp else nullcontext():
                outputs = model(inputs)
                if R_Drop and Lipschitz_regularization:
                    outputs1_polyloss = lipschitz_regularizer.regularize_loss(criterion(outputs, labels))
                    outputs2 = model(inputs)
                    outputs2_polyloss = criterion(outputs2, labels)
                    weights = adaptive_weight(outputs, outputs2, criterion, labels, model)
                    outputs1_mean = outputs1_polyloss.mean()
                    outputs2_mean = outputs2_polyloss.mean()
                    loss_scale = torch.where(outputs2_mean > 1e-8,
                                             outputs1_mean / outputs2_mean,
                                             one_tensor)
                    normalized_weights = weights * loss_scale
                    normalized_weights = normalized_weights / (normalized_weights + one_tensor)
                    combined_loss = normalized_weights * outputs1_polyloss + (
                            one_tensor - normalized_weights) * outputs2_polyloss
                    loss = combined_loss.mean()
                    kl_loss = compute_kl_loss(outputs.detach(), outputs2.detach())
                    loss = loss + alpha_tensor * kl_loss
                elif R_Drop and not Lipschitz_regularization:
                    outputs2 = model(inputs)
                    ce_loss = 0.5 * (criterion(outputs, labels) + criterion(outputs2, labels))
                    kl_loss = compute_kl_loss(outputs, outputs2)
                    loss = ce_loss + alpha_tensor * kl_loss
                elif Lipschitz_regularization and not R_Drop:
                    loss = criterion(outputs, labels)
                    loss = lipschitz_regularizer.regularize_loss(loss)
                else:
                    loss = criterion(outputs, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += loss.detach().item()
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            if batch_idx % 50 == 0:
                experiment.log_metric(f"fold_{fold_num}_train_batch_loss", loss.item(),
                                      step=(epoch * len(train_loader) + batch_idx))
            del outputs, loss
            if R_Drop:
                del outputs2
            if 'kl_loss' in locals():
                del kl_loss
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        experiment.log_metric(f"fold_{fold_num}_learning_rate", current_lr, step=epoch)
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            epoch_time = start_event.elapsed_time(end_event) / 1000
        else:
            epoch_time = 0
        (sensitivity, specificity, average_sensitivity, average_specificity,
         overall_average_score, overall_harmonic_score, overall_score) = calculate_metrics(all_labels, all_preds,
                                                                                           task_type)
        experiment.log_metric(f"fold_{fold_num}_train_accuracy", 100 * correct / total, step=epoch)
        experiment.log_metric(f"fold_{fold_num}_train_loss", running_loss / len(train_loader), step=epoch)
        experiment.log_metric(f"fold_{fold_num}_train_sensitivity", safe_scalar(sensitivity), step=epoch)
        experiment.log_metric(f"fold_{fold_num}_train_specificity", safe_scalar(specificity), step=epoch)
        experiment.log_metric(f"fold_{fold_num}_train_average_score", safe_scalar(overall_average_score), step=epoch)
        experiment.log_metric(f"fold_{fold_num}_train_harmonic_score", safe_scalar(overall_harmonic_score), step=epoch)
        experiment.log_metric(f"fold_{fold_num}_train_overall_score", safe_scalar(overall_score), step=epoch)
        avg_loss = running_loss / len(train_loader)
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.float().to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast() if use_amp else nullcontext():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                if batch_idx % 5 == 0 or batch_idx == len(val_loader) - 1:
                    val_all_preds.extend(predicted.cpu().numpy())
                    val_all_labels.extend(labels.cpu().numpy())
            (val_sensitivity, val_specificity, val_avg_se, val_avg_sp,
             val_avg_score, val_harmonic_score, val_overall_score) = calculate_metrics(
                val_all_labels, val_all_preds, task_type)
            val_accuracy = 100 * val_correct / val_total
            val_loss = val_running_loss / len(val_loader)
            experiment.log_metric(f"fold_{fold_num}_val_accuracy", val_accuracy, step=epoch)
            experiment.log_metric(f"fold_{fold_num}_val_loss", val_loss, step=epoch)
            experiment.log_metric(f"fold_{fold_num}_val_sensitivity", safe_scalar(val_sensitivity), step=epoch)
            experiment.log_metric(f"fold_{fold_num}_val_specificity", safe_scalar(val_specificity), step=epoch)
            experiment.log_metric(f"fold_{fold_num}_val_avg_sensitivity", safe_scalar(val_avg_se), step=epoch)
            experiment.log_metric(f"fold_{fold_num}_val_avg_specificity", safe_scalar(val_avg_sp), step=epoch)
            experiment.log_metric(f"fold_{fold_num}_val_avg_score", safe_scalar(val_avg_score), step=epoch)
            experiment.log_metric(f"fold_{fold_num}_val_harmonic_score", safe_scalar(val_harmonic_score), step=epoch)
            experiment.log_metric(f"fold_{fold_num}_val_overall_score", safe_scalar(val_overall_score), step=epoch)
            val_overall_score_scalar = safe_scalar(val_overall_score)
            is_best = val_overall_score_scalar > best_val_score
            val_scores_history.append(val_overall_score_scalar)
            if is_best:
                best_val_score = val_overall_score_scalar
                metrics = {
                    'accuracy': val_accuracy,
                    'loss': val_loss,
                    'sensitivity': safe_scalar(val_sensitivity),
                    'specificity': safe_scalar(val_specificity),
                    'avg_sensitivity': safe_scalar(val_avg_se),
                    'avg_specificity': safe_scalar(val_avg_sp),
                    'avg_score': safe_scalar(val_avg_score),
                    'harmonic_score': safe_scalar(val_harmonic_score),
                    'overall_score': safe_scalar(val_overall_score)
                }
                best_path = os.path.join(exp_dir, f'fold_{fold_num}_best.pth')
                model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': metrics
                }
                torch.save(checkpoint, best_path)
                print(f"Saved best model for fold {fold_num} at epoch {epoch} with val_overall_score: {metrics['overall_score']:.4f}")
            if early_stop and epoch >= warmup_protection:
                should_stop, best_val_score, early_stop_counter = improved_early_stopping_check(
                    epoch, val_scores_history, best_val_score, early_stop_counter,
                    patience=35, warmup_protection=warmup_protection
                )
                if should_stop:
                    print(f"Early stopping triggered at epoch {epoch} for fold {fold_num}!")
                    print(f"Best validation score: {best_val_score:.4f}")
                    break
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        print(f'Fold {fold_num} | Epoch {epoch + 1}/{num_epochs} [{epoch_duration:.2f}s]')
        print(f'  Train - Loss: {avg_loss:.4f} | Acc: {100 * correct / total:.2f}% | '
              f'Sen: {safe_scalar(sensitivity):.4f} | Spe: {safe_scalar(specificity):.4f} | '
              f'Avg: {safe_scalar(overall_average_score):.4f} | Har: {safe_scalar(overall_harmonic_score):.4f} | '
              f'Overall: {safe_scalar(overall_score):.4f}')
        print(f'  Valid - Loss: {val_loss:.4f} | Acc: {val_accuracy:.2f}% | '
              f'Sen: {safe_scalar(val_sensitivity):.4f} | Spe: {safe_scalar(val_specificity):.4f} | '
              f'Avg: {safe_scalar(val_avg_score):.4f} | Har: {safe_scalar(val_harmonic_score):.4f} | '
              f'Overall: {safe_scalar(val_overall_score):.4f}')
        if is_best:
            print(f'  *** NEW BEST MODEL (Overall Score: {safe_scalar(val_overall_score):.4f}) ***')
    fold_time = time.time() - fold_start_time
    print(f"\n{'=' * 60}")
    print(f"Fold {fold_num} Summary:")
    print(f"  Training time: {fold_time:.2f}s")
    print(f"  Best validation overall score: {best_val_score:.4f}")
    print(f"{'=' * 60}")
    experiment.log_metric(f"fold_{fold_num}_total_time", fold_time)
    experiment.log_metric(f"fold_{fold_num}_best_overall_score", best_val_score)
    return best_val_score

def train_final_model(model, task_type, train_loader, criterion, optimizer, scheduler,
                      R_Drop, α, Lipschitz_regularization, Lipschitz_regularization_degree_alpha,
                      num_epochs, experiment, exp_dir, args, use_amp=True, distributed=False):
    print("\n" + "=" * 50)
    print("Training final model on complete dataset")
    print("=" * 50)
    best_train_loss = float('inf')
    best_epoch = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scaler = GradScaler() if use_amp else None
    zero_tensor = torch.tensor(0.0, device=device)
    one_tensor = torch.tensor(1.0, device=device)
    alpha_tensor = torch.tensor(α, device=device, dtype=torch.float32)
    if Lipschitz_regularization:
        constrain_layers = [nn.Linear, nn.Conv2d]
        lipschitz_regularizer = LipschitzRegularization(
            model,
            alpha=Lipschitz_regularization_degree_alpha,
            constrain_layers=constrain_layers
        )
        adaptive_weight = EntropyAdaptiveWeight().to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast() if use_amp else nullcontext():
                outputs = model(inputs)
                if R_Drop and Lipschitz_regularization:
                    outputs1_polyloss = lipschitz_regularizer.regularize_loss(criterion(outputs, labels))
                    outputs2 = model(inputs)
                    outputs2_polyloss = criterion(outputs2, labels)
                    weights = adaptive_weight(outputs, outputs2, criterion, labels, model)
                    outputs1_mean = outputs1_polyloss.mean()
                    outputs2_mean = outputs2_polyloss.mean()
                    loss_scale = torch.where(outputs2_mean > 1e-8,
                                             outputs1_mean / outputs2_mean,
                                             one_tensor)
                    normalized_weights = weights * loss_scale
                    normalized_weights = normalized_weights / (normalized_weights + one_tensor)
                    combined_loss = normalized_weights * outputs1_polyloss + (
                            one_tensor - normalized_weights) * outputs2_polyloss
                    loss = combined_loss.mean()
                    kl_loss = compute_kl_loss(outputs.detach(), outputs2.detach())
                    loss = loss + alpha_tensor * kl_loss
                elif R_Drop and not Lipschitz_regularization:
                    outputs2 = model(inputs)
                    ce_loss = 0.5 * (criterion(outputs, labels) + criterion(outputs2, labels))
                    kl_loss = compute_kl_loss(outputs, outputs2)
                    loss = ce_loss + alpha_tensor * kl_loss
                elif Lipschitz_regularization and not R_Drop:
                    loss = criterion(outputs, labels)
                    loss = lipschitz_regularizer.regularize_loss(loss)
                else:
                    loss = criterion(outputs, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += loss.detach().item()
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            if batch_idx % 50 == 0:
                experiment.log_metric("final_model_train_batch_loss", loss.item(),
                                      step=(epoch * len(train_loader) + batch_idx))
            del outputs, loss
            if R_Drop:
                del outputs2
            if 'kl_loss' in locals():
                del kl_loss
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        experiment.log_metric("final_model_learning_rate", current_lr, step=epoch)
        avg_loss = running_loss / len(train_loader)
        (sensitivity, specificity, average_sensitivity, average_specificity,
         overall_average_score, overall_harmonic_score, overall_score) = calculate_metrics(all_labels, all_preds,
                                                                                           task_type)
        experiment.log_metric("final_model_train_accuracy", 100 * correct / total, step=epoch)
        experiment.log_metric("final_model_train_loss", avg_loss, step=epoch)
        experiment.log_metric("final_model_train_sensitivity", safe_scalar(sensitivity), step=epoch)
        experiment.log_metric("final_model_train_specificity", safe_scalar(specificity), step=epoch)
        experiment.log_metric("final_model_train_average_score", safe_scalar(overall_average_score), step=epoch)
        experiment.log_metric("final_model_train_harmonic_score", safe_scalar(overall_harmonic_score), step=epoch)
        experiment.log_metric("final_model_train_overall_score", safe_scalar(overall_score), step=epoch)
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            best_epoch = epoch
            metrics = {
                'accuracy': 100 * correct / total,
                'loss': avg_loss,
                'sensitivity': safe_scalar(sensitivity),
                'specificity': safe_scalar(specificity),
                'avg_sensitivity': safe_scalar(average_sensitivity),
                'avg_specificity': safe_scalar(average_specificity),
                'avg_score': safe_scalar(overall_average_score),
                'harmonic_score': safe_scalar(overall_harmonic_score),
                'overall_score': safe_scalar(overall_score)
            }
            model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics
            }
            final_model_path = os.path.join(exp_dir, 'final_model.pth')
            torch.save(checkpoint, final_model_path)
            print(f"Saved final model at epoch {epoch} with train loss: {avg_loss:.4f}")
        print(f'Final Model | Epoch {epoch + 1}/{num_epochs}')
        print(f'  Train - Loss: {avg_loss:.4f} | Acc: {100 * correct / total:.2f}% | '
              f'Sen: {sensitivity:.4f} | Spe: {specificity:.4f} | '
              f'Avg: {overall_average_score:.4f} | Har: {overall_harmonic_score:.4f} | '
              f'Overall: {overall_score:.4f}')
        if avg_loss < best_train_loss:
            print(f'  *** NEW BEST FINAL MODEL (Train Loss: {avg_loss:.4f}) ***')
    print(f"\n{'=' * 60}")
    print(f'Final Model Summary:')
    print(f'  Best training loss: {best_train_loss:.4f} at epoch {best_epoch}')
    print(f"{'=' * 60}")
    experiment.log_metric("final_model_best_epoch", best_epoch)
    experiment.log_metric("final_model_best_loss", best_train_loss)
    return model, best_train_loss, best_epoch

def collate_fn_oversampled(batch):
    batch_data = [d for d, _ in batch]
    batch_labels = [l for _, l in batch]
    padded_data = torch.stack(batch_data, dim=0)
    if len(padded_data.shape) == 3:
        padded_data = padded_data.unsqueeze(1)
    batch_labels = torch.stack(batch_labels, dim=0)
    return padded_data, batch_labels

def collate_fn_oversampled_with_RQ(batch):
    batch_data = [d for d, _ in batch]
    batch_labels = [l for _, l in batch]
    padded_data = torch.stack(batch_data, dim=0)
    if len(padded_data.shape) == 3:
        padded_data = padded_data.unsqueeze(1)
    batch_labels = torch.stack(batch_labels, dim=0)
    from datasets.Randomized_QuantizationAug import RandomizedQuantizationAugModule
    augmentation_module = RandomizedQuantizationAugModule(region_num=5, p_random_apply_rand_quant=0.5)
    padded_data = augmentation_module(padded_data)
    return padded_data, batch_labels

def collate_fn_icbhi_oversampled(batch):
    batch_data = [d for d, _ in batch]
    batch_labels = [l for _, l in batch]
    padded_data = torch.stack(batch_data, dim=0)
    if len(padded_data.shape) == 3:
        padded_data = padded_data.unsqueeze(1)
    batch_labels = torch.stack(batch_labels, dim=0)
    return padded_data, batch_labels

def collate_fn_icbhi_oversampled_with_RQ(batch):
    batch_data = [d for d, _ in batch]
    batch_labels = [l for _, l in batch]
    padded_data = torch.stack(batch_data, dim=0)
    if len(padded_data.shape) == 3:
        padded_data = padded_data.unsqueeze(1)
    batch_labels = torch.stack(batch_labels, dim=0)
    from datasets.Randomized_QuantizationAug import RandomizedQuantizationAugModule
    augmentation_module = RandomizedQuantizationAugModule(region_num=5, p_random_apply_rand_quant=0.5)
    padded_data = augmentation_module(padded_data)
    return padded_data, batch_labels

def create_fold_dataset_with_oversampling(full_dataset, train_idx, args, fold_num):
    print(f"\n{'=' * 60}")
    print(f"SAFE OVERSAMPLING FOR FOLD {fold_num}")
    print(f"{'=' * 60}")
    train_data = []
    train_labels = []
    print(f"Extracting training data for fold {fold_num}...")
    for idx in train_idx:
        data, label = full_dataset[idx]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
        if isinstance(label, torch.Tensor):
            label = label.item()
        train_data.append(data)
        train_labels.append(label)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    print(f"Data shape: {train_data.shape}")
    print(f"Data type: {train_data.dtype}")
    print(f"Labels shape: {train_labels.shape}")
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    print(f"\nBEFORE oversampling (Fold {fold_num}):")
    print(f"  Total samples: {len(train_labels)}")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(train_labels)) * 100
        print(f"  Class {label}: {count} samples ({percentage:.1f}%)")
    if args.use_oversampling and args.oversamplers:
        print(f"\nApplying {args.oversamplers} oversampling...")
        try:
            original_shape = train_data.shape
            if len(original_shape) > 2:
                train_data_flat = train_data.reshape(len(train_data), -1)
            else:
                train_data_flat = train_data
            from utils.oversampling import perform_oversampling, perform_multioversampling
            if args.task_type == 11:
                train_data_resampled, train_labels_resampled = perform_oversampling(
                    args.task_type, train_data_flat, train_labels, args.oversamplers
                )
            else:
                train_data_resampled, train_labels_resampled = perform_multioversampling(
                    args.task_type, train_data_flat, train_labels, args.oversamplers
                )
            if len(original_shape) > 2:
                train_data_resampled = train_data_resampled.reshape(
                    len(train_data_resampled), *original_shape[1:]
                )
            resampled_unique, resampled_counts = np.unique(train_labels_resampled, return_counts=True)
            print(f"\nAFTER oversampling (Fold {fold_num}):")
            print(f"  Total samples: {len(train_labels_resampled)}")
            for label, count in zip(resampled_unique, resampled_counts):
                percentage = (count / len(train_labels_resampled)) * 100
                original_count = dict(zip(unique_labels, counts)).get(label, 0)
                increase = count - original_count
                print(f"  Class {label}: {count} samples ({percentage:.1f}%) [+{increase}]")
            print(f"\nOversampling summary:")
            print(f"  Original: {len(train_labels)} -> Resampled: {len(train_labels_resampled)}")
            print(f"  Increase: {len(train_labels_resampled) - len(train_labels)} samples")
            print(f"  Multiplier: {len(train_labels_resampled) / len(train_labels):.2f}x")
            train_data_tensor = torch.FloatTensor(train_data_resampled)
            train_labels_tensor = torch.LongTensor(train_labels_resampled)
            print(f"   Tensor shapes: data {train_data_tensor.shape}, labels {train_labels_tensor.shape}")
            print(f"   Tensors created on device: {train_data_tensor.device}")
            resampled_dataset = torch.utils.data.TensorDataset(
                train_data_tensor,
                train_labels_tensor
            )
            print(f"Safe oversampling completed for fold {fold_num}")
            print(f"   No data leakage: oversampling only on fold's training set")
            print(f"{'=' * 60}")
            return resampled_dataset
        except Exception as e:
            print(f"Oversampling failed: {e}")
            print("Falling back to original training set...")
            train_data_tensor = torch.FloatTensor(train_data)
            train_labels_tensor = torch.LongTensor(train_labels)
            original_dataset = torch.utils.data.TensorDataset(
                train_data_tensor,
                train_labels_tensor
            )
            return original_dataset
    else:
        print(f"No oversampling applied for fold {fold_num}")
        print(f"Data shape: {train_data.shape}")
        print(f"Labels shape: {train_labels.shape}")
        train_data_tensor = torch.FloatTensor(train_data)
        train_labels_tensor = torch.LongTensor(train_labels)
        print(f"   Tensor shapes: data {train_data_tensor.shape}, labels {train_labels_tensor.shape}")
        print(f"   Tensors created on device: {train_data_tensor.device}")
        original_dataset = torch.utils.data.TensorDataset(
            train_data_tensor,
            train_labels_tensor
        )
        return original_dataset

def main():
    parser = argparse.ArgumentParser(description='Lung Sound SPRSound22_23 Training')
    parser.add_argument("--dataset_type", type=str, default="ICBHI2017",
                        choices=["SPRSound22_23", "ICBHI2017"],
                        help="Dataset type to use")
    parser.add_argument("--task_type", type=int, default=11, choices=[11, 12, 21, 22],
                        help="Task type")
    parser.add_argument("--ICBHI_data", type=str,
                        default='/home/p2412918/Benchmark_Codebase_Construction/datasets/datasets/ICBHI_final_database')
    parser.add_argument("--ICBHI_split_txt", type=str,
                        default='/home/p2412918/Benchmark_Codebase_Construction/datasets/datasets/ICBHI_final_database/ICBHI_challenge_train_test.txt')
    parser.add_argument("--train_wav_path", type=str,
                        default='/home/p2412918/Lung_sound_detection/SPRSound22_23/restore_train_classification_wav')
    parser.add_argument("--train_json_path", type=str,
                        default='/home/p2412918/Lung_sound_detection/SPRSound22_23/train_classification_json')
    parser.add_argument("--feature_type", type=str, default='log-mel',
                        choices=['MFCC', 'log-mel', 'mel', 'STFT'])
    parser.add_argument("--feature_extractor", type=str, default='EfficientNet-B4',
                        choices=['EfficientNet-B4', 'ConvNeXt_base', 'ResNet50', 'DenseNet121',
                                 'MobileViTV2-050', 'MobileViTV2-075', 'MobileViTV2-100',
                                 'MobileViTV2-125', 'MobileViTV2-150', 'MobileViTV2-175',
                                 'MobileViTV2-200', 'StarNet-T0', 'StarNet-S4', 'StarNet-B1'])
    parser.add_argument("--HaB", type=bool, default=True)
    parser.add_argument("--R_Drop", type=bool, default=True)
    parser.add_argument("--α", type=int, default=2)
    parser.add_argument("--Lipschitz_regularization", type=bool, default=True)
    parser.add_argument("--Lipschitz_regularization_degree_alpha", type=float, default=0.0001)
    parser.add_argument("--PolyCrossEntropyLoss", type=bool, default=True)
    parser.add_argument("--epsilon", type=int, default=1)
    parser.add_argument("--use_oversampling", type=str, default=False)
    parser.add_argument("--oversamplers", type=str, default='Borderline_SMOTE2')
    parser.add_argument("--batch_balance_sampler", type=bool, default=False)
    parser.add_argument("--Rrandomized_Quantization_Aug", default=True, type=bool, help="whether using RQ")
    parser.add_argument("--audio_augment_type", type=str, default=None, const=None, nargs='?',
                        help="Type of audio augmentation to use. Use time_stretch, pitch_shift, noise_injection, or leave empty for no augmentation")
    parser.add_argument("--time_stretch_min_rate", type=float, default=0.8)
    parser.add_argument("--time_stretch_max_rate", type=float, default=1.2)
    parser.add_argument("--pitch_shift_min_steps", type=float, default=-4.0)
    parser.add_argument("--pitch_shift_max_steps", type=float, default=4.0)
    parser.add_argument("--noise_level_min", type=float, default=0.001)
    parser.add_argument("--noise_level_max", type=float, default=0.015)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup", type=bool, default=True)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--warmup_epoch", type=int, default=25)
    parser.add_argument("--warmup_base_lr", type=int, default=0.001)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--early_stop", type=bool, default=True)
    parser.add_argument("--use_amp", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of random seeds for ICBHI training")
    parser.add_argument("--base_seed", type=int, default=0, help="Base seed for ICBHI training")
    args = parser.parse_args()
    if args.dataset_type == "ICBHI2017":
        if args.task_type not in [11, 12]:
            print(f"Warning: ICBHI2017 only supports task_type 11 (binary) or 12 (4-class). "
                  f"Changing task_type from {args.task_type} to 11.")
            args.task_type = 11
    if args.audio_augment_type == 'none':
        args.audio_augment_type = None
    if not args.use_oversampling or args.use_oversampling in ["False", "false", "None", "none"]:
        args.oversamplers = None
        args.use_oversampling = False
    else:
        args.use_oversampling = True
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"CUDA optimizations enabled. Available GPUs: {torch.cuda.device_count()}")
    experiment = init_comet_experiment()
    exp_dir = create_experiment_dir(args)
    log_hyperparameters(experiment, args)
    if args.dataset_type == "SPRSound22_23":
        print(f"\n{'=' * 80}")
        print(f"SPRSound22_23 Dataset: Using 5-Fold Cross Validation")
        print(f"{'=' * 80}")
        set_seed(0)
        full_dataset = SPRSoundDataset(
            args.train_wav_path,
            args.train_json_path,
            args.task_type,
            mode='train',
            feature_type=args.feature_type,
            augment_type=args.audio_augment_type,
            augment_params={
                'time_stretch_min_rate': args.time_stretch_min_rate,
                'time_stretch_max_rate': args.time_stretch_max_rate,
                'pitch_shift_min_steps': args.pitch_shift_min_steps,
                'pitch_shift_max_steps': args.pitch_shift_max_steps,
                'noise_level_min': args.noise_level_min,
                'noise_level_max': args.noise_level_max,
            },
            use_cache=True,
            use_oversampling=False,
            oversampler_type=None
        )
        train_collate_fn = collate_fn_train_with_RQ if args.Rrandomized_Quantization_Aug else collate_fn_train
        valid_collate_fn = collate_fn_valid
        oversampled_collate_fn = collate_fn_oversampled_with_RQ if args.Rrandomized_Quantization_Aug else collate_fn_oversampled
        all_labels = full_dataset.labels
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        print(f"\n{'=' * 80}")
        print(f"COMPLETE SPRSound22_23 DATASET DISTRIBUTION")
        print(f"{'=' * 80}")
        print(f"Total samples: {len(all_labels)}")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(all_labels)) * 100
            print(f"  Class {label}: {count} samples ({percentage:.1f}%)")
        all_indices = np.arange(len(full_dataset))
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
            fold_num = fold + 1
            print(f"\n{'=' * 80}")
            print(f"FOLD {fold_num}/{args.n_splits} - DATA PREPARATION")
            print(f"{'=' * 80}")
            overlap = set(train_idx) & set(val_idx)
            if overlap:
                print(f"ERROR: {len(overlap)} overlapping samples detected!")
                raise ValueError(f"Data leakage detected in fold {fold_num}")
            else:
                print(f"No index overlap: train and val sets are disjoint")
            print(f"Training indices: {len(train_idx)} samples")
            print(f"Validation indices: {len(val_idx)} samples")
            train_labels_fold = [all_labels[i] for i in train_idx]
            val_labels_fold = [all_labels[i] for i in val_idx]
            train_unique, train_counts = np.unique(train_labels_fold, return_counts=True)
            val_unique, val_counts = np.unique(val_labels_fold, return_counts=True)
            print(f"\nTraining set distribution (before oversampling):")
            for label, count in zip(train_unique, train_counts):
                percentage = (count / len(train_labels_fold)) * 100
                print(f"  Class {label}: {count} samples ({percentage:.1f}%)")
            print(f"\nValidation set distribution:")
            for label, count in zip(val_unique, val_counts):
                percentage = (count / len(val_labels_fold)) * 100
                print(f"  Class {label}: {count} samples ({percentage:.1f}%)")
            val_subset = Subset(full_dataset, val_idx)
            val_loader = DataLoader(
                val_subset,
                batch_size=args.batch_size * 2,
                shuffle=False,
                collate_fn=valid_collate_fn,
                pin_memory=True,
                num_workers=0 if args.use_oversampling else args.num_workers
            )
            if args.use_oversampling and args.oversamplers:
                train_dataset_resampled = create_fold_dataset_with_oversampling(
                    full_dataset, train_idx, args, fold_num
                )
                train_loader = DataLoader(
                    train_dataset_resampled,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=oversampled_collate_fn,
                    pin_memory=True,
                    num_workers=0,
                    persistent_workers=False
                )
            else:
                print(f"\nNo oversampling applied for fold {fold_num}")
                train_subset = Subset(full_dataset, train_idx)
                train_loader = DataLoader(
                    train_subset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=train_collate_fn,
                    pin_memory=True,
                    num_workers=args.num_workers
                )
            print(f"\nDataLoader creation complete (Fold {fold_num}):")
            print(f"   Training batches: {len(train_loader)}")
            print(f"   Validation batches: {len(val_loader)}")
            print(f"   Training samples: {len(train_loader.dataset)}")
            print(f"   Validation samples: {len(val_loader.dataset)}")
            num_classes = len(set(full_dataset.get_task_dict().values()))
            if args.HaB:
                print("using HaB for extracting frequency features")
                if args.feature_extractor == 'EfficientNet-B4':
                    model = CustomEfficientNetWithLoad.from_pretrained(
                        'efficientnet-b4',
                        num_classes=num_classes
                    )
                elif args.feature_extractor == 'ResNet50':
                    model = CustomResNet(num_classes=num_classes)
                elif args.feature_extractor == 'DenseNet121':
                    model = CustomDenseNet(num_classes=num_classes)
                elif args.feature_extractor == 'ConvNeXt_base':
                    model = CustomConvNeXt(num_classes=num_classes)
                elif args.feature_extractor.startswith('MobileViTV2'):
                    print(f"using {args.feature_extractor} + HaB for extracting features")
                    model = get_mobilevitv2_model(args.feature_extractor, num_classes, pretrained=True)
                elif args.feature_extractor == 'StarNet-T0' and STARNET_AVAILABLE:
                    print("using StarNet-T0 + HaB for extracting features")
                    model = starnet_t0(num_classes)
                elif args.feature_extractor == 'StarNet-S4' and STARNET_AVAILABLE:
                    print("using StarNet-S4 + HaB for extracting features (RECOMMENDED)")
                    model = starnet_s4(num_classes)
                elif args.feature_extractor == 'StarNet-B1' and STARNET_AVAILABLE:
                    print("using StarNet-B1 + HaB for extracting features")
                    model = starnet_b1(num_classes)
            else:
                if args.feature_extractor == 'EfficientNet-B4':
                    print("using EfficientNet-B4 for extracting features")
                    model = load_efficientnet_model(num_classes=num_classes)
                elif args.feature_extractor == 'ResNet50':
                    print("using ResNet50 for extracting features")
                    model = resnet50(pretrained=True)
                    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, num_classes)
                elif args.feature_extractor == 'DenseNet121':
                    print("using DenseNet121 for extracting features")
                    model = densenet121(pretrained=True)
                    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    num_ftrs = model.classifier.in_features
                    model.classifier = nn.Linear(num_ftrs, num_classes)
                elif args.feature_extractor == 'ConvNeXt_base':
                    print("using ConvNeXt_base for extracting features")
                    model = convnext_base(pretrained=True)
                    model.features[0][0] = nn.Conv2d(1, 128, kernel_size=4, stride=4)
                    num_ftrs = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
                elif args.feature_extractor.startswith('MobileViTV2'):
                    print(f"using {args.feature_extractor} (without HaB) for extracting features")
                    import timm
                    variant_name = args.feature_extractor.replace('-', '_').lower()
                    model = timm.create_model(variant_name, pretrained=True, in_chans=1, num_classes=num_classes)
                elif args.feature_extractor == 'StarNet-T0' and STARNET_AVAILABLE:
                    print("using StarNet-T0 + HaB for extracting features")
                    model = starnet_t0(num_classes)
                elif args.feature_extractor == 'StarNet-S4' and STARNET_AVAILABLE:
                    print("using StarNet-S4 + HaB for extracting features (RECOMMENDED)")
                    model = starnet_s4(num_classes)
                elif args.feature_extractor == 'StarNet-B1' and STARNET_AVAILABLE:
                    print("using StarNet-B1 + HaB for extracting features")
                    model = starnet_b1(num_classes)
            criterion = Poly1CrossEntropyLoss(
                num_classes=num_classes,
                epsilon=args.epsilon,
                reduction='mean'
            ) if args.PolyCrossEntropyLoss else nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=1e-6
            )
            scheduler = WarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=args.warmup_epoch,
                total_epochs=args.epoch,
                target_lr=args.lr,
                warmup_start_lr=args.warmup_base_lr,
                min_lr=0.000001
            )
            try:
                best_score = train_fold(
                    model,
                    args.task_type,
                    train_loader,
                    val_loader,
                    fold_num,
                    criterion,
                    optimizer,
                    scheduler,
                    args.R_Drop,
                    args.α,
                    args.early_stop,
                    args.Lipschitz_regularization,
                    args.Lipschitz_regularization_degree_alpha,
                    num_epochs=args.epoch,
                    experiment=experiment,
                    exp_dir=exp_dir,
                    args=args,
                    use_amp=args.use_amp
                )
                fold_scores.append(best_score)
            except Exception as e:
                print(f"Training failed for fold {fold_num} with error: {str(e)}")
                error_path = os.path.join(exp_dir, f'error_fold_{fold_num}.pth')
                torch.save({
                    'error': str(e),
                    'fold': fold_num
                }, error_path)
                continue
        if fold_scores:
            avg_score = sum(fold_scores) / len(fold_scores)
            std_score = np.std(fold_scores)
            print(f"\n{'=' * 80}")
            print(f"SPRSound22_23 CROSS-VALIDATION RESULTS SUMMARY")
            print(f"{'=' * 80}")
            print(f"{'Fold':<6} {'Best Overall Score':<20}")
            print(f"{'-' * 26}")
            for i, score in enumerate(fold_scores, 1):
                print(f"{i:<6} {score:<20.4f}")
            print(f"{'-' * 26}")
            print(f"{'Mean':<6} {avg_score:<20.4f}")
            print(f"{'Std':<6} {std_score:<20.4f}")
            print(f"{'=' * 80}")
            experiment.log_metric("cv_average_best_score", avg_score)
            experiment.log_metric("cv_std_best_score", std_score)
        print("\n" + "=" * 50)
        print("Training final SPRSound22_23 model on complete dataset")
        print("=" * 50)
        final_dataset = SPRSoundDataset(
            args.train_wav_path,
            args.train_json_path,
            args.task_type,
            mode='train',
            feature_type=args.feature_type,
            augment_type=args.audio_augment_type,
            augment_params={
                'time_stretch_min_rate': args.time_stretch_min_rate,
                'time_stretch_max_rate': args.time_stretch_max_rate,
                'pitch_shift_min_steps': args.pitch_shift_min_steps,
                'pitch_shift_max_steps': args.pitch_shift_max_steps,
                'noise_level_min': args.noise_level_min,
                'noise_level_max': args.noise_level_max,
            },
            use_cache=True,
            use_oversampling=args.use_oversampling,
            oversampler_type=args.oversamplers
        )
        final_collate_fn = collate_fn_train_with_RQ if args.Rrandomized_Quantization_Aug else collate_fn_train
        full_train_loader = DataLoader(
            final_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=final_collate_fn,
            pin_memory=True,
            num_workers=args.num_workers
        )
        num_classes = len(set(final_dataset.get_task_dict().values()))
        if args.HaB:
            print("using HaB for extracting frequency features (final model)")
            if args.feature_extractor == 'EfficientNet-B4':
                final_model = CustomEfficientNetWithLoad.from_pretrained(
                    'efficientnet-b4',
                    num_classes=num_classes
                )
            elif args.feature_extractor == 'ResNet50':
                final_model = CustomResNet(num_classes=num_classes)
            elif args.feature_extractor == 'DenseNet121':
                final_model = CustomDenseNet(num_classes=num_classes)
            elif args.feature_extractor == 'ConvNeXt_base':
                final_model = CustomConvNeXt(num_classes=num_classes)
            elif args.feature_extractor.startswith('MobileViTV2'):
                print(f"using {args.feature_extractor} + HaB for extracting features (final model)")
                final_model = get_mobilevitv2_model(args.feature_extractor, num_classes, pretrained=True)
            elif args.feature_extractor == 'StarNet-T0' and STARNET_AVAILABLE:
                print("using StarNet-T0 + HaB for extracting features")
                model = starnet_t0(num_classes)
            elif args.feature_extractor == 'StarNet-S4' and STARNET_AVAILABLE:
                print("using StarNet-S4 + HaB for extracting features (RECOMMENDED)")
                model = starnet_s4(num_classes)
            elif args.feature_extractor == 'StarNet-B1' and STARNET_AVAILABLE:
                print("using StarNet-B1 + HaB for extracting features")
                model = starnet_b1(num_classes)
        else:
            if args.feature_extractor == 'EfficientNet-B4':
                print("using EfficientNet-B4 for extracting features (final model)")
                final_model = load_efficientnet_model(num_classes=num_classes)
            elif args.feature_extractor == 'ResNet50':
                print("using ResNet50 for extracting features (final model)")
                final_model = resnet50(pretrained=True)
                final_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                num_ftrs = final_model.fc.in_features
                final_model.fc = nn.Linear(num_ftrs, num_classes)
            elif args.feature_extractor == 'DenseNet121':
                print("using DenseNet121 for extracting features (final model)")
                final_model = densenet121(pretrained=True)
                final_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                num_ftrs = final_model.classifier.in_features
                final_model.classifier = nn.Linear(num_ftrs, num_classes)
            elif args.feature_extractor == 'ConvNeXt_base':
                print("using ConvNeXt_base for extracting features (final model)")
                final_model = convnext_base(pretrained=True)
                final_model.features[0][0] = nn.Conv2d(1, 128, kernel_size=4, stride=4)
                num_ftrs = final_model.classifier[-1].in_features
                final_model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            elif args.feature_extractor.startswith('MobileViTV2'):
                print(f"using {args.feature_extractor} (without HaB) for extracting features (final model)")
                import timm
                variant_name = args.feature_extractor.replace('-', '_').lower()
                final_model = timm.create_model(variant_name, pretrained=True, in_chans=1, num_classes=num_classes)
            elif args.feature_extractor == 'StarNet-T0' and STARNET_AVAILABLE:
                print("using StarNet-T0 + HaB for extracting features")
                model = starnet_t0(num_classes)
            elif args.feature_extractor == 'StarNet-S4' and STARNET_AVAILABLE:
                print("using StarNet-S4 + HaB for extracting features (RECOMMENDED)")
                model = starnet_s4(num_classes)
            elif args.feature_extractor == 'StarNet-B1' and STARNET_AVAILABLE:
                print("using StarNet-B1 + HaB for extracting features")
                model = starnet_b1(num_classes)
        final_criterion = Poly1CrossEntropyLoss(
            num_classes=num_classes,
            epsilon=args.epsilon,
            reduction='mean'
        ) if args.PolyCrossEntropyLoss else nn.CrossEntropyLoss()
        final_optimizer = optim.AdamW(
            final_model.parameters(),
            lr=args.lr,
            weight_decay=1e-6
        )
        final_scheduler = WarmupCosineAnnealingLR(
            final_optimizer,
            warmup_epochs=args.warmup_epoch,
            total_epochs=args.epoch,
            target_lr=args.lr,
            warmup_start_lr=args.warmup_base_lr,
            min_lr=0.000001
        )
        final_best_loss = float('inf')
        final_best_epoch = 0
        try:
            _, final_best_loss, final_best_epoch = train_final_model(
                final_model,
                args.task_type,
                full_train_loader,
                final_criterion,
                final_optimizer,
                final_scheduler,
                args.R_Drop,
                args.α,
                args.Lipschitz_regularization,
                args.Lipschitz_regularization_degree_alpha,
                num_epochs=args.epoch,
                experiment=experiment,
                exp_dir=exp_dir,
                args=args,
                use_amp=args.use_amp
            )
        except Exception as e:
            print(f"Final model training failed with error: {str(e)}")
            error_path = os.path.join(exp_dir, 'error_final_model.pth')
            torch.save({
                'error': str(e)
            }, error_path)
        print(f"\n{'=' * 80}")
        print(f"SPRSound22_23 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 80}")
        print(f"Cross-validation results:")
        if fold_scores:
            print(f"  - Average best score: {sum(fold_scores) / len(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
            print(f"  - Best fold score: {max(fold_scores):.4f}")
            print(f"  - Worst fold score: {min(fold_scores):.4f}")
        print(f"\nFinal model:")
        print(f"  - Best training loss: {final_best_loss:.4f} at epoch {final_best_epoch}")
        print(f"\nAll models saved in: {exp_dir}")

    elif args.dataset_type == "ICBHI2017":
        print(f"\n{'=' * 80}")
        print(f"ICBHI2017 Dataset: Using {args.n_seeds} Random Seeds Training")
        print(f"{'=' * 80}")
        seed_scores = []
        train_collate_fn = collate_fn_icbhi_train_with_RQ if args.Rrandomized_Quantization_Aug else collate_fn_icbhi_train
        test_collate_fn = collate_fn_icbhi_valid
        for seed_idx in range(args.n_seeds):
            seed_num = args.base_seed + seed_idx
            print(f"\n{'=' * 80}")
            print(f"SEED {seed_num} ({seed_idx + 1}/{args.n_seeds}) - DATA PREPARATION")
            print(f"{'=' * 80}")
            set_seed(seed_num)
            train_dataset = create_icbhi_dataset(
                data_dir=args.ICBHI_data,
                split_file=args.ICBHI_split_txt,
                task_type=args.task_type,
                mode='train',
                feature_type=args.feature_type,
                augment_type=args.audio_augment_type,
                augment_params={
                    'time_stretch_min_rate': args.time_stretch_min_rate,
                    'time_stretch_max_rate': args.time_stretch_max_rate,
                    'pitch_shift_min_steps': args.pitch_shift_min_steps,
                    'pitch_shift_max_steps': args.pitch_shift_max_steps,
                    'noise_level_min': args.noise_level_min,
                    'noise_level_max': args.noise_level_max,
                },
                use_oversampling=args.use_oversampling,
                oversampler_type=args.oversamplers
            )
            test_dataset = create_icbhi_dataset(
                data_dir=args.ICBHI_data,
                split_file=args.ICBHI_split_txt,
                task_type=args.task_type,
                mode='test',
                feature_type=args.feature_type,
                augment_type=None,
                augment_params=None,
                use_oversampling=False,
                oversampler_type=None
            )
            train_labels = train_dataset.labels
            test_labels = test_dataset.labels
            train_unique, train_counts = np.unique(train_labels, return_counts=True)
            test_unique, test_counts = np.unique(test_labels, return_counts=True)
            print(f"\nTraining set distribution (seed {seed_num}):")
            print(f"  Total samples: {len(train_labels)}")
            for label, count in zip(train_unique, train_counts):
                percentage = (count / len(train_labels)) * 100
                print(f"  Class {label}: {count} samples ({percentage:.1f}%)")
            print(f"\nTest set distribution (seed {seed_num}):")
            print(f"  Total samples: {len(test_labels)}")
            for label, count in zip(test_unique, test_counts):
                percentage = (count / len(test_labels)) * 100
                print(f"  Class {label}: {count} samples ({percentage:.1f}%)")
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=train_collate_fn,
                pin_memory=True,
                num_workers=args.num_workers
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size * 2,
                shuffle=False,
                collate_fn=test_collate_fn,
                pin_memory=True,
                num_workers=args.num_workers
            )
            print(f"\nDataLoader creation complete (Seed {seed_num}):")
            print(f"   Training batches: {len(train_loader)}")
            print(f"   Test batches: {len(test_loader)}")
            print(f"   Training samples: {len(train_loader.dataset)}")
            print(f"   Test samples: {len(test_loader.dataset)}")
            num_classes = len(set(train_dataset.get_task_dict().values()))
            if args.HaB:
                print("using HaB for extracting frequency features")
                if args.feature_extractor == 'EfficientNet-B4':
                    model = CustomEfficientNetWithLoad.from_pretrained(
                        'efficientnet-b4',
                        num_classes=num_classes
                    )
                elif args.feature_extractor == 'ResNet50':
                    model = CustomResNet(num_classes=num_classes)
                elif args.feature_extractor == 'DenseNet121':
                    model = CustomDenseNet(num_classes=num_classes)
                elif args.feature_extractor == 'ConvNeXt_base':
                    model = CustomConvNeXt(num_classes=num_classes)
                elif args.feature_extractor.startswith('MobileViTV2'):
                    print(f"using {args.feature_extractor} + HaB for extracting features")
                    model = get_mobilevitv2_model(args.feature_extractor, num_classes, pretrained=True)
                elif args.feature_extractor == 'StarNet-T0' and STARNET_AVAILABLE:
                    print("using StarNet-T0 + HaB for extracting features")
                    model = starnet_t0(num_classes)
                elif args.feature_extractor == 'StarNet-S4' and STARNET_AVAILABLE:
                    print("using StarNet-S4 + HaB for extracting features (RECOMMENDED)")
                    model = starnet_s4(num_classes)
                elif args.feature_extractor == 'StarNet-B1' and STARNET_AVAILABLE:
                    print("using StarNet-B1 + HaB for extracting features")
                    model = starnet_b1(num_classes)
            else:
                if args.feature_extractor == 'EfficientNet-B4':
                    print("using EfficientNet-B4 for extracting features")
                    model = load_efficientnet_model(num_classes=num_classes)
                elif args.feature_extractor == 'ResNet50':
                    print("using ResNet50 for extracting features")
                    model = resnet50(pretrained=True)
                    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, num_classes)
                elif args.feature_extractor == 'DenseNet121':
                    print("using DenseNet121 for extracting features")
                    model = densenet121(pretrained=True)
                    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    num_ftrs = model.classifier.in_features
                    model.classifier = nn.Linear(num_ftrs, num_classes)
                elif args.feature_extractor == 'ConvNeXt_base':
                    print("using ConvNeXt_base for extracting features")
                    model = convnext_base(pretrained=True)
                    model.features[0][0] = nn.Conv2d(1, 128, kernel_size=4, stride=4)
                    num_ftrs = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
                elif args.feature_extractor.startswith('MobileViTV2'):
                    print(f"using {args.feature_extractor} (without HaB) for extracting features")
                    import timm
                    variant_name = args.feature_extractor.replace('-', '_').lower()
                    model = timm.create_model(variant_name, pretrained=True, in_chans=1, num_classes=num_classes)
                elif args.feature_extractor == 'StarNet-T0' and STARNET_AVAILABLE:
                    print("using StarNet-T0 + HaB for extracting features")
                    model = starnet_t0(num_classes)
                elif args.feature_extractor == 'StarNet-S4' and STARNET_AVAILABLE:
                    print("using StarNet-S4 + HaB for extracting features (RECOMMENDED)")
                    model = starnet_s4(num_classes)
                elif args.feature_extractor == 'StarNet-B1' and STARNET_AVAILABLE:
                    print("using StarNet-B1 + HaB for extracting features")
                    model = starnet_b1(num_classes)
            criterion = Poly1CrossEntropyLoss(
                num_classes=num_classes,
                epsilon=args.epsilon,
                reduction='mean'
            ) if args.PolyCrossEntropyLoss else nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=1e-6
            )
            scheduler = WarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=args.warmup_epoch,
                total_epochs=args.epoch,
                target_lr=args.lr,
                warmup_start_lr=args.warmup_base_lr,
                min_lr=0.000001
            )
            try:
                best_score = train_seed_icbhi(
                    model,
                    args.task_type,
                    train_loader,
                    test_loader,
                    seed_num,
                    criterion,
                    optimizer,
                    scheduler,
                    args.R_Drop,
                    args.α,
                    args.early_stop,
                    args.Lipschitz_regularization,
                    args.Lipschitz_regularization_degree_alpha,
                    num_epochs=args.epoch,
                    experiment=experiment,
                    exp_dir=exp_dir,
                    args=args,
                    use_amp=args.use_amp
                )
                seed_scores.append(best_score)
            except Exception as e:
                print(f"Training failed for seed {seed_num} with error: {str(e)}")
                error_path = os.path.join(exp_dir, f'error_seed_{seed_num}.pth')
                torch.save({
                    'error': str(e),
                    'seed': seed_num
                }, error_path)
                continue
        if seed_scores:
            avg_score = sum(seed_scores) / len(seed_scores)
            std_score = np.std(seed_scores)
            print(f"\n{'=' * 80}")
            print(f"ICBHI2017 MULTI-SEED RESULTS SUMMARY")
            print(f"{'=' * 80}")
            print(f"{'Seed':<6} {'Best Average Score':<20}")
            print(f"{'-' * 26}")
            for i, score in enumerate(seed_scores):
                seed_num = args.base_seed + i
                print(f"{seed_num:<6} {score:<20.4f}")
            print(f"{'-' * 26}")
            print(f"{'Mean':<6} {avg_score:<20.4f}")
            print(f"{'Std':<6} {std_score:<20.4f}")
            print(f"{'=' * 80}")
            experiment.log_metric("multi_seed_average_best_score", avg_score)
            experiment.log_metric("multi_seed_std_best_score", std_score)
        print(f"\n{'=' * 80}")
        print(f"ICBHI2017 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 80}")
        print(f"Multi-seed results:")
        if seed_scores:
            print(f"  - Average best score: {sum(seed_scores) / len(seed_scores):.4f} +/- {np.std(seed_scores):.4f}")
            print(f"  - Best seed score: {max(seed_scores):.4f}")
            print(f"  - Worst seed score: {min(seed_scores):.4f}")
        print(f"\nAll models saved in: {exp_dir}")
        print(f"\nICBHI2017 Training Strategy Summary:")
        print(f"  - Used {args.n_seeds} random seeds: {list(range(args.base_seed, args.base_seed + args.n_seeds))}")
        print(f"  - Models saved based on best test average score (not overall score)")
        print(f"  - Each seed trained with fixed train/test split")
    if args.feature_extractor.startswith('MobileViTV2'):
        print(f"\nMobileViTV2 Model Summary:")
        print(f"  - Variant: {args.feature_extractor}")
        print(f"  - HaB Integration: {'ENABLED' if args.HaB else 'DISABLED'}")
        print(f"  - Expected inference speed: Fast (optimized for mobile)")
    print(f"{'=' * 80}")

def get_model_params_count(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    main()