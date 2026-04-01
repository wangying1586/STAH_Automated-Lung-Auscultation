from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from sklearn.metrics import confusion_matrix


def init_comet_experiment():
    """Initialize and return a new Comet experiment"""
    return Experiment(
        api_key="amErNSTjMM7BiByvvLlZnXy2n",
        project_name="j1",
        workspace="wangying1586"
    )


def log_hyperparameters(experiment, args):
    """Log hyperparameters to Comet experiment"""
    hyper_params = {
        "task_type": args.task_type,
        "feature_type": args.feature_type,
        "feature_extractor": args.feature_extractor,
        "HaB": args.HaB,
        "R_Drop": args.R_Drop,
        "α": args.α,
        "Lipschitz_regularization": args.Lipschitz_regularization,
        "Lipschitz_regularization_degree_alpha": args.Lipschitz_regularization_degree_alpha,
        "PolyCrossEntropyLoss": args.PolyCrossEntropyLoss,
        "epsilon": args.epsilon,
        "use_oversampling": args.use_oversampling,
        "oversamplers": args.oversamplers,
        "batch_balance_sampler": args.batch_balance_sampler,
        "Rrandomized_Quantization_Aug": args.Rrandomized_Quantization_Aug,  # 修正拼写
        "audio_augment_type": args.audio_augment_type,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "warmup": args.warmup,
        "warmup_epoch": args.warmup_epoch,
        "warmup_base_lr": args.warmup_base_lr,
        "lr": args.lr,
        "early_stop": args.early_stop,
        "use_amp": args.use_amp,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "prefetch_factor": args.prefetch_factor,
        "n_splits": args.n_splits,
        # 音频增强参数
        "time_stretch_min_rate": args.time_stretch_min_rate,
        "time_stretch_max_rate": args.time_stretch_max_rate,
        "pitch_shift_min_steps": args.pitch_shift_min_steps,
        "pitch_shift_max_steps": args.pitch_shift_max_steps,
        "noise_level_min": args.noise_level_min,
        "noise_level_max": args.noise_level_max,
    }
    experiment.log_parameters(hyper_params)


def get_task_labels(task_type):
    """Return labels based on task type"""
    task_labels = {
        11: ['Normal', 'Adventitious'],
        12: ['Normal', 'Rhonchi', 'Wheeze', 'Stridor', 'Coarse Crackle',
             'Fine Crackle', 'Wheeze+Crackle'],
        21: ['Normal', 'Poor Quality', 'Adventitious'],
        22: ['Normal', 'Poor Quality', 'CAS', 'DAS', 'CAS & DAS']
    }
    return task_labels.get(task_type, [])


def log_model_to_comet(experiment, model, model_name):
    """Log PyTorch model to Comet"""
    log_model(experiment, model=model, model_name=model_name)