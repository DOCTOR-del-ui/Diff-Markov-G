import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

idx = 2
window = 96
# 特征列名
feature_cols = ['s2', 's3', 's4', 's7', 's8', 's9', 's11',
                's12', 's13', 's14', 's15', 's17', 's20', 's21']

modelname = "DiffTS"
submodelname = "markovpro"

gen_data_path = os.path.join("OUTPUT", f"cmapsst{submodelname}{idx}_{window}", f"ddpm_fake_cmapsst{submodelname}{idx}_{window}.npy")
gen_data = np.load(gen_data_path)

print(gen_data.shape)

""" ori_data_path = os.path.join("OUTPUT", f"cmapss_mts{idx}_{window}", "samples", f"cmapss_mts{idx}_norm_truth_{window}_train.npy")
ori_data = np.load(ori_data_path)

print(ori_data.shape) """

gen_save_dir = os.path.join("figures", f"CMAPSS_{modelname}_{submodelname}", "gen", f"FD00{idx}_{window}")
""" ori_save_dir = os.path.join("figures", f"CMAPSS_{modelname}", "ori", f"FD00{idx}_{window}") """
os.makedirs(gen_save_dir, exist_ok=True)
""" os.makedirs(ori_save_dir, exist_ok=True) """

# 随机选取100个样本
index = np.random.choice(gen_data.shape[0], 50, replace=False)
sampled_gendata = gen_data[index]

gen_length = sampled_gendata.shape[0]
for di in range(1, gen_length+1):
    # 创建图形和子图（7 行 2 列）
    fig, axes = plt.subplots(7, 2, figsize=(12, 18))
    axes = axes.flatten()

    # 绘制每个特征的曲线
    for i in range(14):
        axes[i].plot(range(sampled_gendata[di-1].shape[0]), sampled_gendata[di-1][:, i])
        axes[i].set_title(f"{feature_cols[i]}(Window {window})")
        axes[i].set_xlabel('Time step')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)

    fig.suptitle(f"Generated Features FD00{idx}_{window}_{di}", fontsize=15)

    # 调整布局，防止标题重叠
    plt.tight_layout()
    plt.savefig(os.path.join(gen_save_dir, f"gen_fig FD00{idx}_{window} {di}.png"))
    

""" # 随机选取100个样本
index = np.random.choice(ori_data.shape[0], 50, replace=False)
sampled_oridata = ori_data[index]

ori_length = sampled_oridata.shape[0]
for di in range(1, ori_length+1):
    # 创建图形和子图（7 行 2 列）
    fig, axes = plt.subplots(7, 2, figsize=(12, 18))
    axes = axes.flatten()

    # 绘制每个特征的曲线
    for i in range(14):
        axes[i].plot(range(sampled_oridata[di-1].shape[0]), sampled_oridata[di-1][:, i])
        axes[i].set_title(f"{feature_cols[i]}(Window {window})")
        axes[i].set_xlabel('Time step')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)
        
    fig.suptitle(f"Origin Features FD00{idx}_{window}_{di}", fontsize=15)

    # 调整布局，防止标题重叠
    plt.tight_layout()
    plt.savefig(os.path.join(ori_save_dir, f"ori_fig FD00{idx}_{window} {di}.png")) """