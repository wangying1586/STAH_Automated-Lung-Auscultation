import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from contextlib import nullcontext
from torch.cuda.amp import autocast
from datasets.SPRSound_dataloader import get_class_names

def evaluate_model(model, task_type, val_loader, experiment, epoch, criterion, early_stop, val_set_type=None,
                   val_set_year=None, use_amp=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    val_loss_sum = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in tqdm(enumerate(val_loader),
                                                total=len(val_loader),
                                                desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)

            # 使用自动混合精度
            with autocast() if use_amp else nullcontext():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 计算当前批次的损失并累加
            val_loss_sum += loss.item()

            experiment.log_metric("val_loss", val_loss_sum / (batch_idx + 1),
                                  step=(epoch * len(val_loader) + batch_idx))

        val_loss_avg = val_loss_sum / len(val_loader)
        print(f'Epoch {epoch}: Validation Loss = {val_loss_avg}')

        # print(
        #     f"verifying the output labels for task_type_{task_type}: [[[[ epoch_{epoch},  all_preds_{all_preds},  all_labels_{all_labels} ]]]]")

        # 计算验证集的指标
        conf_matrix = confusion_matrix(all_labels, all_preds)
        (sensitivity, specificity, average_sensitivity, average_specificity,
         overall_average_score, overall_harmonic_score, overall_score) = calculate_metrics(all_labels, all_preds,
                                                                                           task_type)

        # 记录验证集指标到Comet
        suffix = f"task_type_{task_type}_{val_set_type}_{val_set_year}" if val_set_type else f"task_type_{task_type}_{val_set_year}"

        # 记录验证集的指标到Comet，添加合适的后缀来区分不同验证集的指标
        experiment.log_metric(f"val_accuracy_{suffix}", 100 * correct / total, step=epoch)
        experiment.log_metric(f"val_sensitivity_{suffix}", sensitivity, step=epoch)
        experiment.log_metric(f"val_specificity_{suffix}", specificity, step=epoch)
        experiment.log_metric(f"average_sensitivity_{suffix}", average_sensitivity, step=epoch)
        experiment.log_metric(f"average_specificity_{suffix}", average_specificity, step=epoch)
        experiment.log_metric(f"val_average_score_{suffix}", overall_average_score, step=epoch)
        experiment.log_metric(f"val_harmonic_score_{suffix}", overall_harmonic_score, step=epoch)
        experiment.log_metric(f"val_overall_score_{suffix}", overall_score, step=epoch)

        # 记录混淆矩阵到Comet
        if task_type == 11:
            # 修改混淆矩阵记录部分
            class_names = get_class_names(task_type)
            experiment.log_confusion_matrix(
                matrix=conf_matrix,
                labels=class_names,  # 使用相同的类别名称列表
                step=epoch,
                file_name=f"confusion_matrix_{suffix}.png",
                title=f"Confusion Matrix {suffix}"
            )
        elif task_type == 12:
            # 修改混淆矩阵记录部分
            class_names = get_class_names(task_type)
            experiment.log_confusion_matrix(
                matrix=conf_matrix,
                labels=class_names,  # 使用相同的类别名称列表
                step=epoch,
                file_name=f"confusion_matrix_{suffix}.png",
                title=f"Confusion Matrix {suffix}"
            )
        elif task_type == 21:
            # 修改混淆矩阵记录部分
            class_names = get_class_names(task_type)
            experiment.log_confusion_matrix(
                matrix=conf_matrix,
                labels=class_names,  # 使用相同的类别名称列表
                step=epoch,
                file_name=f"confusion_matrix_{suffix}.png",
                title=f"Confusion Matrix {suffix}"
            )
        elif task_type == 22:
            # 修改混淆矩阵记录部分
            class_names = get_class_names(task_type)
            experiment.log_confusion_matrix(
                matrix=conf_matrix,
                labels=class_names,  # 使用相同的类别名称列表
                step=epoch,
                file_name=f"confusion_matrix_{suffix}.png",
                title=f"Confusion Matrix {suffix}"
            )

    return (100 * correct / total, sensitivity, specificity, average_sensitivity, average_specificity,
            overall_average_score, overall_harmonic_score, overall_score), val_loss_avg

def calculate_metrics(all_labels, all_preds, task_type):
    # 获取固定的类别顺序
    class_names = get_class_names(task_type)
    num_classes = len(class_names)

    # 使用固定顺序计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds,
                                   labels=range(num_classes))

    # print("\nConfusion Matrix:")
    # print(conf_matrix)
    # print("\nClass order:", class_names)  # 打印类别顺序以便验证

    # num_classes = {11: 2, 12: 7, 21: 3, 22: 5}[task_type]
    # conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    # print("\nConfusion Matrix:")
    # print(conf_matrix)

    if task_type == 11:  # 二分类问题
        # print("\nBinary SPRSound22_23 Metrics:")
        tn = conf_matrix[0, 0]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]
        tp = conf_matrix[1, 1]

        # print(f"TN = {tn}")
        # print(f"FP = {fp}")
        # print(f"FN = {fn}")
        # print(f"TP = {tp}")

        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        # print(f"Sensitivity = {tp} / ({tp} + {fn}) = {sensitivity:.4f}")
        # print(f"Specificity = {tn} / ({tn} + {fp}) = {specificity:.4f}")

        sensitivities = sensitivity
        specificities = specificity
        macro_sensitivity = sensitivity
        macro_specificity = specificity

    else:  # 多分类问题
        # print("\nMulticlass SPRSound22_23 Metrics:")
        total_samples = np.sum(conf_matrix)
        # print(f"Total samples: {total_samples}")

        sensitivities = []
        specificities = []

        for i in range(num_classes):
            # print(f"\nClass {i} metrics:")

            # True Positive: 对角线元素
            tp = conf_matrix[i, i]

            # False Negative: 该行中除了TP外的所有元素之和
            row_sum = np.sum(conf_matrix[i, :])
            fn = row_sum - tp

            # False Positive: 该列中除了TP外的所有元素之和
            col_sum = np.sum(conf_matrix[:, i])
            fp = col_sum - tp

            # True Negative: 总样本数减去该行和该列的和，加回重复减去的TP
            tn = total_samples - (row_sum + col_sum - tp)

            # print(f"TP = {tp}")
            # print(f"FN = {fn}")
            # print(f"FP = {fp}")
            # print(f"TN = {tn}")

            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

            # print(f"Sensitivity = {tp} / ({tp} + {fn}) = {sensitivity:.4f}")
            # print(f"Specificity = {tn} / ({tn} + {fp}) = {specificity:.4f}")

            sensitivities.append(sensitivity)
            specificities.append(specificity)

        sensitivities = np.array(sensitivities)
        specificities = np.array(specificities)

        # print("\nPer-class metrics:")
        # print(f"Sensitivities: {sensitivities}")
        # print(f"Specificities: {specificities}")

        macro_sensitivity = np.mean(sensitivities)
        macro_specificity = np.mean(specificities)

        # print("\nMacro averages:")
        # print(f"Macro-Sensitivity: {macro_sensitivity:.4f}")
        # print(f"Macro-Specificity: {macro_specificity:.4f}")

        # 综合分数计算不变
    average_score = (macro_sensitivity + macro_specificity) / 2
    harmonic_score = 2 * macro_sensitivity * macro_specificity / (macro_sensitivity + macro_specificity + 1e-9)
    overall_score = (average_score + harmonic_score) / 2

    # print("\nFinal scores:")
    # print(f"Average Score = ({macro_sensitivity:.4f} + {macro_specificity:.4f}) / 2 = {average_score:.4f}")
    # print(
    #     f"Harmonic Score = 2 * ({macro_sensitivity:.4f} * {macro_specificity:.4f}) / ({macro_sensitivity:.4f} + {macro_specificity:.4f}) = {harmonic_score:.4f}")
    # print(f"Overall Score = ({average_score:.4f} + {harmonic_score:.4f}) / 2 = {overall_score:.4f}")

    return (sensitivities, specificities, macro_sensitivity, macro_specificity,
            average_score, harmonic_score, overall_score)