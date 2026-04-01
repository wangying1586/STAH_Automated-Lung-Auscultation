#!/bin/bash

# GPU-Optimized Lung Sound SPRSound22_23 Training Script
# Usage: ./train.sh <experiment_config> <GPU_device_id>
# Example: ./train.sh icbhi_hab_taf 0

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$2

# Environment variables
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_BACKENDS_CUDNN_BENCHMARK=1

timestamp=$(date +"%Y%m%d_%H%M%S")
experiment_config=$1

if [ -z "$experiment_config" ]; then
    echo "Error: Please specify experiment configuration"
    echo ""
    echo "Available configurations:"
    echo "  icbhi_baseline     - ICBHI2017 baseline (no HaB, no TAF)"
    echo "  icbhi_hab          - ICBHI2017 + HaB module"
    echo "  icbhi_taf          - ICBHI2017 + TAF module"
    echo "  icbhi_hab_taf      - ICBHI2017 + HaB + TAF (full)"
    echo "  icbhi_mobilevit    - ICBHI2017 + MobileViTV2 + HaB"
    echo "  sprsound_baseline  - SPRSound baseline"
    echo "  sprsound_full      - SPRSound full experiment"
    echo "  sprsound_mobilevit - SPRSound + MobileViTV2 + HaB"
    echo "  custom             - Custom configuration"
    echo ""
    echo "Usage: ./train.sh <config_name> <GPU_device_id>"
    echo "Example: ./train.sh icbhi_mobilevit 0"
    exit 1
fi

# Configuration settings
case "$experiment_config" in
    "icbhi_baseline")
        dataset_type="ICBHI2017"
        task_type=11
        feature_extractor="EfficientNet-B4"
        HaB="o"
        TAF="o"
        oversampler="o"
        feature_augmentation="none"
        RQ_aug="o"
        batch_size=64
        ;;
    "icbhi_hab")
        dataset_type="ICBHI2017"
        task_type=12
        feature_extractor="EfficientNet-B4"
        HaB="w"
        TAF="o"
        oversampler="w"
        feature_augmentation="none"
        RQ_aug="w"
        batch_size=64
        ;;
    "icbhi_taf")
        dataset_type="ICBHI2017"
        task_type=12
        feature_extractor="EfficientNet-B4"
        HaB="o"
        TAF="w"
        oversampler="w"
        feature_augmentation="none"
        RQ_aug="w"
        batch_size=48
        ;;
    "icbhi_hab_taf")
        dataset_type="ICBHI2017"
        task_type=12
        feature_extractor="StarNet-S4"
        HaB="w"
        TAF="w"
        oversampler="o"
        feature_augmentation="none"
        RQ_aug="w"
        batch_size=32
        ;;
    "icbhi_mobilevit")
        dataset_type="ICBHI2017"
        task_type=11
        feature_extractor="MobileViTV2-125"
        HaB="w"
        TAF="w"
        oversampler="o"
        feature_augmentation="none"
        RQ_aug="w"
        batch_size=32
        ;;
    "sprsound_baseline")
        dataset_type="SPRSound"
        task_type=12
        feature_extractor="EfficientNet-B4"
        HaB="o"
        TAF="o"
        oversampler="w"
        feature_augmentation="none"
        RQ_aug="w"
        batch_size=64
        ;;
    "sprsound_full")
        dataset_type="SPRSound"
        task_type=12
        feature_extractor="EfficientNet-B4"
        HaB="w"
        TAF="w"
        oversampler="w"
        feature_augmentation="TS"
        RQ_aug="w"
        batch_size=32
        ;;
    "sprsound_mobilevit")
        dataset_type="SPRSound"
        task_type=12
        feature_extractor="MobileViTV2-075"
        HaB="w"
        TAF="w"
        oversampler="w"
        feature_augmentation="TS"
        RQ_aug="w"
        batch_size=24
        ;;
    "custom")
        dataset_type="ICBHI2017"
        task_type=12
        feature_extractor="MobileViTV2-100"
        HaB="w"
        TAF="w"
        oversampler="w"
        feature_augmentation="none"
        RQ_aug="w"
        batch_size=32
        ;;
    *)
        echo "Error: Unknown experiment configuration '$experiment_config'"
        exit 1
        ;;
esac

# MobileViTV2 specific optimizations
case "$feature_extractor" in
    "MobileViTV2-050")
        batch_size_multiplier=2.0
        lr_multiplier=1.5
        ;;
    "MobileViTV2-075")
        batch_size_multiplier=1.5
        lr_multiplier=1.2
        ;;
    "MobileViTV2-100")
        batch_size_multiplier=1.0
        lr_multiplier=1.0
        ;;
    "MobileViTV2-125")
        batch_size_multiplier=0.8
        lr_multiplier=0.8
        ;;
    "MobileViTV2-150")
        batch_size_multiplier=0.6
        lr_multiplier=0.7
        ;;
    "MobileViTV2-175")
        batch_size_multiplier=0.5
        lr_multiplier=0.6
        ;;
    "MobileViTV2-200")
        batch_size_multiplier=0.4
        lr_multiplier=0.5
        ;;
    *)
        batch_size_multiplier=1.0
        lr_multiplier=1.0
        ;;
esac

if [[ "$feature_extractor" == MobileViTV2* ]]; then
    batch_size=$(echo "scale=0; $batch_size * $batch_size_multiplier / 1" | bc -l)
    if [ $batch_size -lt 4 ]; then
        batch_size=4
    fi
    base_lr=0.0001
    optimized_lr=$(echo "scale=6; $base_lr * $lr_multiplier" | bc -l)
else
    optimized_lr=0.0001
fi

# Optimization parameters
num_workers=1
prefetch_factor=4
pin_memory="True"
use_amp="True"
early_stop="False"
warmup="True"

# Dataset paths (update these to your actual paths)
ICBHI_data_dir="./datasets/ICBHI2017"
ICBHI_split_file="./datasets/ICBHI2017/meta_information/ICBHI_challenge_train_test.txt"

SPR_train_wav_path="./datasets/SPRSound/wav"
SPR_train_json_path="./datasets/SPRSound/json"

if [ "$dataset_type" = "ICBHI2017" ]; then
    data_dir="$ICBHI_data_dir"
    split_file="$ICBHI_split_file"
    if [ "$task_type" != "11" ] && [ "$task_type" != "12" ]; then
        task_type=12
    fi
else
    data_dir="$SPR_train_wav_path"
    split_file="$SPR_train_json_path"
fi

# Parameter conversion
if [ "$HaB" = "w" ]; then
    HaB_param="True"
else
    HaB_param="False"
fi

if [ "$TAF" = "w" ]; then
    R_Drop_param="True"
    Lipschitz_regularization_param="True"
    PolyCrossEntropyLoss_param="True"
else
    R_Drop_param="False"
    Lipschitz_regularization_param="False"
    PolyCrossEntropyLoss_param="False"
fi

if [ "$oversampler" = "w" ]; then
    use_oversampling_param="True"
    oversamplers_param="Borderline_SMOTE2"
else
    use_oversampling_param="False"
    oversamplers_param="None"
fi

if [ "$RQ_aug" = "w" ]; then
    RQ_aug_param="True"
else
    RQ_aug_param="False"
fi

case "$feature_augmentation" in
    "TS")
        audio_augment_type_param="time_stretch"
        ;;
    "PS")
        audio_augment_type_param="pitch_shift"
        ;;
    "NI")
        audio_augment_type_param="noise_injection"
        ;;
    "RQ")
        audio_augment_type_param="none"
        RQ_aug_param="True"
        ;;
    *)
        audio_augment_type_param="none"
        ;;
esac

# Logging
log_dir="./logs"
mkdir -p $log_dir
log_file="${log_dir}/train_${experiment_config}_${dataset_type}_${feature_extractor}_task${task_type}_bs${batch_size}_${timestamp}.log"

# Display configuration
echo "Experiment: $experiment_config"
echo "Dataset: $dataset_type"
echo "Task: $task_type"
echo "Batch size: $batch_size"
echo "Feature extractor: $feature_extractor"
echo "HaB: $HaB_param"
echo "TAF: $R_Drop_param, $Lipschitz_regularization_param, $PolyCrossEntropyLoss_param"
echo "Oversampling: $use_oversampling_param ($oversamplers_param)"
echo "Augmentation: $audio_augment_type_param"
echo "Log: $log_file"

# Start training
nohup python train.py \
    --dataset_type "$dataset_type" \
    --task_type $task_type \
    --ICBHI_data "$ICBHI_data_dir" \
    --ICBHI_split_txt "$ICBHI_split_file" \
    --train_wav_path "$SPR_train_wav_path" \
    --train_json_path "$SPR_train_json_path" \
    --feature_type 'log-mel' \
    --feature_extractor "$feature_extractor" \
    --HaB $HaB_param \
    --R_Drop $R_Drop_param \
    --α 2 \
    --Lipschitz_regularization $Lipschitz_regularization_param \
    --Lipschitz_regularization_degree_alpha 0.0001 \
    --PolyCrossEntropyLoss $PolyCrossEntropyLoss_param \
    --epsilon 1 \
    --use_oversampling $use_oversampling_param \
    --oversamplers "$oversamplers_param" \
    --batch_balance_sampler False \
    --Rrandomized_Quantization_Aug $RQ_aug_param \
    --audio_augment_type "$audio_augment_type_param" \
    --batch_size $batch_size \
    --warmup $warmup \
    --epoch 100 \
    --warmup_epoch 10 \
    --lr $optimized_lr \
    --early_stop $early_stop \
    --use_amp $use_amp \
    --num_workers $num_workers \
    --pin_memory $pin_memory \
    --prefetch_factor $prefetch_factor > "$log_file" 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "Monitor with: tail -f $log_file"