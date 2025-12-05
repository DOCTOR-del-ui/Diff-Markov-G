import os
import argparse
import numpy as np
import pandas as pd

try:
    from Utils.discriminative_metric import *
    from Utils.predictive_metric import *
except Exception:
    pass

try:
    from Utils.context_fid import *
    from Utils.cross_correlation import *
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--addname', type=str, default=None)
    parser.add_argument('--index', type=int, default=None)
    parser.add_argument('--window', type=int, default=None)
    parser.add_argument('--metric', type=str, default=None)
    parser.add_argument('--selfeval', action='store_true', help='Enable self-evaluation')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    dataset = args.dataset
    addname = args.addname
    index = args.index
    window = args.window
    metric = args.metric
    selfeval = args.selfeval
    
    #np.random.seed(54321)
    
    output_dir = os.path.join("OUTPUT", f"{dataset}{addname}{index}_{window}")
    
    try:
        true_path = os.path.join(output_dir, "samples", f"{dataset}{addname}{index}_norm_truth_{window}_train.npy")
        true_data_r = np.load(true_path)
    except Exception:
        true_path = os.path.join("OUTPUT", f"cmapsst{index}_{window}", "samples", f"cmapsst{index}_norm_truth_{window}_train.npy")
        true_data_r = np.load(true_path)
    
    if selfeval:
        gen_data = true_data_r.copy()
    else:
        gen_path = os.path.join(output_dir, f"ddpm_fake_{dataset}{addname}{index}_{window}.npy")
        gen_data_r = np.load(gen_path)
    
    test_num = 10
    curr_res_list = []
    
    for i in range(test_num):
        if true_data_r.shape[0] < gen_data_r.shape[0]:
            N = true_data_r.shape[0]
            """ if N >= 12800:
                N = 12800 """
            idx = np.random.choice(N, size=N, replace=False)  # 随机不重复抽样索引
            gen_data = gen_data_r[idx]
            true_data = true_data_r.copy()
        else:
            N = gen_data_r.shape[0]
            """ if N >= 12800:
                N = 12800 """
            idx = np.random.choice(N, size=N, replace=False)  # 随机不重复抽样索引
            true_data = true_data_r[idx]
            gen_data = gen_data_r.copy()
    
        if metric == "discrim":
            discriminative_score, fake_acc, real_acc = discriminative_score_metrics(true_data, gen_data)
            curr_res_list.append(discriminative_score)
            print(f"Test {i+1}/{test_num}")
            
        elif metric == "pred":
            predictive_score = predictive_score_metrics(true_data, gen_data)
            curr_res_list.append(predictive_score)
            print(f"Test {i+1}/{test_num}")
            
        elif metric == "fid":
            fid_score = Context_FID(true_data, gen_data)
            curr_res_list.append(fid_score)
            print(f"Test {i+1}/{test_num}")
            
        elif metric == "cross":
            true_data = torch.as_tensor(true_data, dtype=torch.float32)
            gen_data = torch.as_tensor(gen_data, dtype=torch.float32)
            name = f"{dataset}{addname}{index}_{window}"
            cross_correlation_loss = CrossCorrelLoss(x_real = true_data, name = name)
            cross_correlation = cross_correlation_loss(gen_data)
            curr_res_list.append(cross_correlation.item())
            print(f"Test {i+1}/{test_num}")
    print(curr_res_list)
    print(
        f"Average {metric} tests: {np.mean(curr_res_list)}   "
        f"Min: {np.min(curr_res_list)}   "
        f"Std: {np.std(curr_res_list)}"
    )

    
    