"""
检查训练后模型的theta参数分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.NetworkParser import parse_topology, parse_histories, parse_tunnels
from utils.scenario import subScenarios
from algorithms.TUFTTESolver_old import TUFTTESolver, Availability

def check_theta_distribution(model_suffix="_quick", scale=15.0):
    """
    检查模型的theta分布
    
    Args:
        model_suffix: 模型后缀名
        scale: 训练时使用的scale
    """
    print("="*70)
    print("Theta Distribution Analysis")
    print("="*70)
    
    # 加载网络
    print("\n1. Loading network...")
    network = parse_topology("Abilene")
    parse_histories(network)
    network.reduce_data(1000, 30)
    parse_tunnels(network, k=8)
    network.tunnel_type = "int"  # 或者 "ksp_8"，取决于你训练时用的
    network.set_scale(scale)
    network.prepare_solution_format()
    
    # 设置场景
    prob_failure = []
    edge_included = []
    for edge in network.edges:
        if set(edge) in edge_included:
            continue
        prob_failure.append(network.edges[edge].prob_failure)
        edge_included.append(set(edge))
    
    scenarios_all = subScenarios(prob_failure, cutoff=1e-6)
    network.set_scenario(scenarios_all)
    
    # 加载模型
    print(f"\n2. Loading model...")
    model_path = f"data/{network.name}/model_A_scale{scale}{model_suffix}.pkl"
    try:
        model = torch.load(model_path, weights_only=False)
        model.eval()
        print(f"   ✓ Model loaded: {model_path}")
    except FileNotFoundError:
        print(f"   ✗ Model not found: {model_path}")
        return
    
    # 收集theta值
    print(f"\n3. Collecting theta values on test set...")
    from torch.utils.data import DataLoader
    from algorithms.TUFTTESolver_old import TUFTTEDataset
    
    fake_opts = [0] * (len(network.test_hists._tms) - 12)
    test_data = TUFTTEDataset(network.test_hists._tms, fake_opts, hist_len=12)
    test_loader = DataLoader(test_data, shuffle=False)
    
    # 获取模型所在设备
    device = next(model.parameters()).device
    print(f"   Model device: {device}")
    
    all_thetas = []
    with torch.no_grad():
        for (x, _, _) in test_loader:
            x = x.to(device)  # 移到模型所在设备
            theta = model.predict_only(x)  # 只预测theta，不做优化
            all_thetas.append(theta.detach().cpu().numpy())  # 移回CPU后转numpy
    
    all_thetas = np.concatenate(all_thetas, axis=0)  # shape: (num_samples, num_od_pairs)
    
    # 统计信息
    print(f"\n4. Statistics:")
    print(f"   Number of test samples: {all_thetas.shape[0]}")
    print(f"   Number of OD pairs: {all_thetas.shape[1]}")
    print(f"\n   Theta statistics (across all samples and OD pairs):")
    print(f"   - Min:    {all_thetas.min():.4f}")
    print(f"   - Max:    {all_thetas.max():.4f}")
    print(f"   - Mean:   {all_thetas.mean():.4f}")
    print(f"   - Median: {np.median(all_thetas):.4f}")
    print(f"   - Std:    {all_thetas.std():.4f}")
    
    # 查看极端值比例
    extreme_low = (all_thetas < 0.1).sum() / all_thetas.size * 100
    extreme_high = (all_thetas > 1.9).sum() / all_thetas.size * 100
    print(f"\n   Extreme values:")
    print(f"   - θ < 0.1:  {extreme_low:.2f}%")
    print(f"   - θ > 1.9:  {extreme_high:.2f}%")
    
    # 按OD pair统计
    print(f"\n   Per-OD-pair statistics:")
    theta_means = all_thetas.mean(axis=0)
    theta_stds = all_thetas.std(axis=0)
    print(f"   - OD with lowest mean theta:  {theta_means.min():.4f} (OD pair {theta_means.argmin()})")
    print(f"   - OD with highest mean theta: {theta_means.max():.4f} (OD pair {theta_means.argmax()})")
    print(f"   - Avg std across OD pairs:    {theta_stds.mean():.4f}")
    
    # 可视化
    print(f"\n5. Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 整体分布直方图
    axes[0, 0].hist(all_thetas.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(1.0, color='red', linestyle='--', linewidth=2, label='θ=1.0 (neutral)')
    axes[0, 0].set_xlabel('Theta Value', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Overall Theta Distribution', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 各个OD pair的平均theta
    axes[0, 1].bar(range(len(theta_means)), theta_means, alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(1.0, color='red', linestyle='--', linewidth=2, label='θ=1.0')
    axes[0, 1].set_xlabel('OD Pair Index', fontsize=12)
    axes[0, 1].set_ylabel('Mean Theta', fontsize=12)
    axes[0, 1].set_title('Mean Theta per OD Pair', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Box plot (抽样部分OD pairs，避免太密集)
    sample_indices = np.linspace(0, all_thetas.shape[1]-1, min(20, all_thetas.shape[1]), dtype=int)
    boxplot_data = [all_thetas[:, i] for i in sample_indices]
    axes[1, 0].boxplot(boxplot_data, labels=[f"{i}" for i in sample_indices])
    axes[1, 0].axhline(1.0, color='red', linestyle='--', linewidth=2, label='θ=1.0')
    axes[1, 0].set_xlabel('OD Pair Index (sampled)', fontsize=12)
    axes[1, 0].set_ylabel('Theta Value', fontsize=12)
    axes[1, 0].set_title('Theta Distribution per OD Pair (Box Plot)', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 时间序列：随测试样本变化
    sample_ods = [0, all_thetas.shape[1]//2, all_thetas.shape[1]-1]  # 头、中、尾3个OD
    for od_idx in sample_ods:
        axes[1, 1].plot(all_thetas[:, od_idx], label=f'OD {od_idx}', alpha=0.7)
    axes[1, 1].axhline(1.0, color='red', linestyle='--', linewidth=2, label='θ=1.0')
    axes[1, 1].set_xlabel('Test Sample Index', fontsize=12)
    axes[1, 1].set_ylabel('Theta Value', fontsize=12)
    axes[1, 1].set_title('Theta Over Test Samples (Selected ODs)', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f"theta_distribution{model_suffix}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Plot saved: {output_file}")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    
    return all_thetas

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check theta distribution")
    parser.add_argument("--suffix", type=str, default="_quick", help="Model suffix")
    parser.add_argument("--scale", type=float, default=15.0, help="Training scale")
    
    args = parser.parse_args()
    
    thetas = check_theta_distribution(model_suffix=args.suffix, scale=args.scale)
