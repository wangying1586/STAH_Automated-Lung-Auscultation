import numpy as np

is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False

import smote_variants as sv


def perform_oversampling(task_type, X, y, oversampler_name):
    # print(f"Performing oversampling with {oversampler_name}...")
    # print(f"Original class distribution: {np.bincount(y)}")
    # for i in np.unique(y):
    #     print(f"Original class {i} - samples: {np.sum(y == i)}")
    # print(f"Original class distribution: {np.bincount(y)}")

    """
    执行给定过采样器的过采样操作，并绘制相应的过采样后数据集的可视化图像。

    参数:
    task_type (int): 任务类型标识，用于文件名等地方。
    X (numpy.ndarray): 特征数据。
    y (numpy.ndarray): 标签数据。
    oversampler_name (str): 过采样器的名称。

    返回:
    None
    """

    # 根据过采样器名称选择对应的过采样器实例化对象.
    if oversampler_name == "distance_SMOTE":
        oversampler = sv.distance_SMOTE()
    elif oversampler_name == "SMOTE":
        oversampler = sv.SMOTE()
    elif oversampler_name == "SYMPROD":
        oversampler = sv.SYMPROD()
    elif oversampler_name == "SN_SMOTE":
        oversampler = sv.SN_SMOTE()
    elif oversampler_name == "ADASYN":
        oversampler = sv.ADASYN()
    elif oversampler_name == "ASMOBD":
        oversampler = sv.ASMOBD()
    elif oversampler_name == "AMSCO":
        oversampler = sv.AMSCO()
    elif oversampler_name == "ANS":
        oversampler = sv.ANS()
    elif oversampler_name == "Supervised_SMOTE":
        oversampler = sv.Supervised_SMOTE()
    elif oversampler_name == "Borderline_SMOTE1":
        oversampler = sv.Borderline_SMOTE1()
    elif oversampler_name == "Borderline_SMOTE2":
        oversampler = sv.Borderline_SMOTE2()
    elif oversampler_name == "kmeans_SMOTE":
        oversampler = sv.kmeans_SMOTE()
    elif oversampler_name == "SMOTE_IPF":
        oversampler = sv.SMOTE_IPF()
    elif oversampler_name == "ProWSyn":
        oversampler = sv.ProWSyn()
    elif oversampler_name.startswith("polynom_fit_SMOTE_star"):
        oversampler = sv.polynom_fit_SMOTE_star()
    elif oversampler_name.startswith("polynom_fit_SMOTE_poly"):
        oversampler = sv.polynom_fit_SMOTE_poly()
    elif oversampler_name.startswith("polynom_fit_SMOTE_mesh"):
        oversampler = sv.polynom_fit_SMOTE_mesh()
    elif oversampler_name.startswith("polynom_fit_SMOTE_bus"):
        oversampler = sv.polynom_fit_SMOTE_bus()
    else:
        raise ValueError(f"不支持的过采样器名称: {oversampler_name}")

    X_samp, y_samp = oversampler.sample(X, y)

    # 打印过采样后每个类别的样本数量
    # for i in np.unique(y_samp):
    #     print(f"{oversampler_name} oversampled class {i} - samples: {np.sum(y_samp == i)}")

    return X_samp, y_samp


def perform_multioversampling(task_type, X, y, oversampler_name):
    """
    执行给定过采样器的过采样操作，并绘制相应的过采样后数据集的可视化图像。

    参数:
    task_type (int): 任务类型标识，用于文件名等地方。
    X (numpy.ndarray): 特征数据。
    y (numpy.ndarray): 标签数据。
    oversampler_name (str): 过采样器的名称。

    返回:
    None
    """
    print(f"Performing multi-class oversampling with {oversampler_name}...")

    # for i in np.unique(y):
    #     print(f"Original class {i} - samples: {np.sum(y == i)}")

    # 根据名称选择对应的过采样器并进行采样
    oversampler = sv.MulticlassOversampling(oversampler_name)

    if oversampler:
        X_samp, y_samp = oversampler.sample(X, y)
    else:
        return {}

    # 打印过采样后每个类别的样本数量
    # for i in np.unique(y_samp):
    #     print(f"{oversampler_name} oversampled class {i} - samples: {np.sum(y_samp == i)}")

    return X_samp, y_samp
