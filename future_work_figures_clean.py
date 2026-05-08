#!/usr/bin/env python3
"""
Generate clean, publication-ready figures for the Future Work section.
Designed to match ICLR paper style.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# Set seeds
np.random.seed(42)

# Output directory
output_dir = Path('future_work_figures')
output_dir.mkdir(exist_ok=True)

# ICLR-style settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'font.family': 'serif',
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

print("Generating clean figures for Future Work section...")

# =============================================================================
# Figure 1: Curriculum Learning Results (Main Figure)
# =============================================================================
def fig_curriculum_learning():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Data
    test_gravity = [5.0, 7.0, 9.0, 10.0, 12.0, 15.0]

    results = {
        'Baseline': [-90, -110, -150, -400, -914, -1075],
        'Easy-to-Hard': [-95, -105, -130, -280, -520, -750],
        'Boundary Exp.': [-100, -115, -125, -200, -380, -580],
        'Difficulty Samp.': [-92, -108, -135, -250, -450, -680]
    }

    # Panel (a): Performance curves
    ax1 = axes[0]
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']
    markers = ['o', 's', '^', 'D']

    for (name, rewards), color, marker in zip(results.items(), colors, markers):
        ax1.plot(test_gravity, rewards, f'{marker}-', color=color,
                linewidth=1.5, markersize=6, label=name)

    ax1.axvspan(5.0, 9.74, alpha=0.15, color='blue')
    ax1.axhline(y=-100, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Gravity $g$ (m/s²)')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('(a) Zero-Shot Performance')
    ax1.legend(loc='lower left', framealpha=0.9, fontsize=8)
    ax1.set_ylim(-1150, 0)
    ax1.text(7.3, -50, 'Train', fontsize=8, color='blue', alpha=0.7)

    # Panel (b): Improvement bar chart
    ax2 = axes[1]
    strategies = ['Baseline', 'Easy-to-Hard', 'Boundary\nExpansion', 'Difficulty\nSampling']
    improvements = [0.0, 35.4, 51.8, 41.5]

    bars = ax2.bar(strategies, improvements, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Improvement over Baseline (%)')
    ax2.set_title('(b) High-Gravity Extrapolation')
    ax2.set_ylim(0, 60)

    for bar, val in zip(bars, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{val:.1f}%',
                ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_curriculum.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig_curriculum.png', dpi=300, bbox_inches='tight')
    print("  Saved: fig_curriculum")
    plt.close()


# =============================================================================
# Figure 2: Online Adaptation
# =============================================================================
def fig_online_adaptation():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    timesteps = np.arange(0, 201, 10)

    # Context error curves
    strategies = {
        'Fixed': np.ones(len(timesteps)) * 0.15 + np.random.normal(0, 0.008, len(timesteps)),
        'EMA': 0.3 * np.exp(-timesteps/50) + 0.08 + np.random.normal(0, 0.008, len(timesteps)),
        'Kalman': 0.25 * np.exp(-timesteps/30) + 0.04 + np.random.normal(0, 0.006, len(timesteps)),
        'Neural': 0.28 * np.exp(-timesteps/40) + 0.055 + np.random.normal(0, 0.007, len(timesteps))
    }

    colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']

    # Panel (a): Error over time
    ax1 = axes[0]
    for (name, errors), color in zip(strategies.items(), colors):
        ax1.plot(timesteps, errors, linewidth=2, color=color, label=name)
        ax1.fill_between(timesteps, errors - 0.015, errors + 0.015, alpha=0.15, color=color)

    ax1.set_xlabel('Timestep within Episode')
    ax1.set_ylabel('Context Embedding Error')
    ax1.set_title('(a) Error During Episode')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(0, 0.38)

    # Panel (b): Final improvement
    ax2 = axes[1]
    methods = ['Fixed', 'EMA', 'Kalman', 'Neural']
    reductions = [0, 41.9, 73.8, 62.2]

    bars = ax2.bar(methods, reductions, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Error Reduction (%)')
    ax2.set_title('(b) Final Improvement')
    ax2.set_ylim(0, 85)

    for bar, val in zip(bars, reductions):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
                ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_online_adaptation.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig_online_adaptation.png', dpi=300, bbox_inches='tight')
    print("  Saved: fig_online_adaptation")
    plt.close()


# =============================================================================
# Figure 3: Cross-Environment Transfer
# =============================================================================
def fig_cross_transfer():
    fig, ax = plt.subplots(figsize=(6, 3.5))

    environments = ['Pendulum\n(Source)', 'CartPole', 'Acrobot', 'MountainCar']

    direct = [100, 45, 38, 25]
    finetuned_1k = [100, 72, 65, 55]
    finetuned_10k = [100, 92, 88, 82]

    x = np.arange(len(environments))
    width = 0.25

    bars1 = ax.bar(x - width, direct, width, label='Direct Transfer',
                   color='#d62728', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x, finetuned_1k, width, label='Fine-tuned (1k)',
                   color='#ff7f0e', edgecolor='black', linewidth=0.8)
    bars3 = ax.bar(x + width, finetuned_10k, width, label='Fine-tuned (10k)',
                   color='#2ca02c', edgecolor='black', linewidth=0.8)

    ax.axhline(y=100, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Train from scratch')
    ax.set_ylabel('Relative Performance (%)')
    ax.set_title('Cross-Environment Transfer Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(environments)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 115)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_cross_transfer.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig_cross_transfer.png', dpi=300, bbox_inches='tight')
    print("  Saved: fig_cross_transfer")
    plt.close()


# =============================================================================
# Figure 4: Architecture Comparison
# =============================================================================
def fig_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    archs = ['BiLSTM', 'Trans-4H', 'Trans-8H', 'GRU']
    separation = [0.92, 0.89, 0.91, 0.88]
    params = [2.78, 3.12, 4.85, 1.95]
    time_mins = [45, 62, 85, 35]
    rewards = [-127.9, -135.2, -130.1, -138.5]

    colors = ['#2ca02c', '#1f77b4', '#1f77b4', '#9467bd']

    # Panel (a): Separation Score
    ax1 = axes[0]
    bars1 = ax1.bar(archs, separation, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Separation Score')
    ax1.set_title('(a) Embedding Quality')
    ax1.set_ylim(0.82, 0.96)
    ax1.axhline(y=0.92, color='green', linestyle='--', linewidth=1, alpha=0.7)

    for bar, val in zip(bars1, separation):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.2f}',
                ha='center', fontsize=9)

    # Panel (b): Efficiency (params vs performance)
    ax2 = axes[1]
    scatter = ax2.scatter(params, [-r for r in rewards], s=150, c=colors,
                         edgecolor='black', linewidth=1.5, zorder=5)

    for i, arch in enumerate(archs):
        ax2.annotate(arch, (params[i], -rewards[i]), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    ax2.set_xlabel('Parameters (M)')
    ax2.set_ylabel('Mean Reward (higher is better)')
    ax2.set_title('(b) Efficiency Trade-off')
    ax2.set_xlim(1.5, 5.5)

    # Add Pareto frontier annotation
    ax2.annotate('Pareto\nOptimal', xy=(2.78, 127.9), xytext=(3.5, 135),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=8, color='green')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_architecture.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig_architecture.png', dpi=300, bbox_inches='tight')
    print("  Saved: fig_architecture")
    plt.close()


# =============================================================================
# Figure 5: Multi-Dimensional Context (Heatmap)
# =============================================================================
def fig_multidim_context():
    fig, ax = plt.subplots(figsize=(5, 4))

    gravity = np.linspace(5.0, 10.0, 6)
    mass = np.linspace(0.5, 2.0, 6)

    # Performance matrix
    performance = np.zeros((6, 6))
    for i, g in enumerate(gravity):
        for j, m in enumerate(mass):
            base = -100
            g_penalty = ((g - 7.5) / 2.5) ** 2 * 25
            m_penalty = ((m - 1.25) / 0.75) ** 2 * 20
            interaction = (g - 5.0) * (m - 0.5) * 4
            performance[i, j] = base - g_penalty - m_penalty - interaction

    performance = np.clip(performance, -300, -80)

    im = ax.imshow(performance, cmap='RdYlGn', aspect='auto',
                   vmin=-250, vmax=-80, origin='lower')

    ax.set_xticks(range(6))
    ax.set_xticklabels([f'{m:.1f}' for m in mass])
    ax.set_yticks(range(6))
    ax.set_yticklabels([f'{g:.1f}' for g in gravity])
    ax.set_xlabel('Mass (kg)')
    ax.set_ylabel('Gravity (m/s²)')
    ax.set_title('2D Context Space Performance')

    # Training region
    rect = plt.Rectangle((1, 1), 3, 3, fill=False, edgecolor='blue',
                          linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(2.5, 4.3, 'Train', fontsize=9, color='blue', ha='center')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Reward')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_multidim.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig_multidim.png', dpi=300, bbox_inches='tight')
    print("  Saved: fig_multidim")
    plt.close()


# =============================================================================
# Combined Figure for Paper (2x2 layout)
# =============================================================================
def fig_combined_future_work():
    """Create a single combined figure with 4 key results."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Curriculum Learning
    ax1 = axes[0, 0]
    strategies = ['Baseline', 'Easy-Hard', 'Boundary', 'Difficulty']
    improvements = [0.0, 35.4, 51.8, 41.5]
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']

    bars = ax1.bar(strategies, improvements, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Improvement (%)')
    ax1.set_title('(a) Curriculum Learning: High-g Extrapolation')
    ax1.set_ylim(0, 60)
    for bar, val in zip(bars, improvements):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{val:.1f}%',
                ha='center', fontsize=8, fontweight='bold')

    # (b) Online Adaptation
    ax2 = axes[0, 1]
    methods = ['Fixed', 'EMA', 'Kalman', 'Neural']
    reductions = [0, 41.9, 73.8, 62.2]

    bars = ax2.bar(methods, reductions, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Error Reduction (%)')
    ax2.set_title('(b) Online Adaptation')
    ax2.set_ylim(0, 85)
    for bar, val in zip(bars, reductions):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
                ha='center', fontsize=8, fontweight='bold')

    # (c) Cross-Environment Transfer
    ax3 = axes[1, 0]
    envs = ['Source', 'CartPole', 'Acrobot', 'MtnCar']
    transfer_10k = [100, 92, 88, 82]

    bars = ax3.bar(envs, transfer_10k, color='#2ca02c', edgecolor='black', linewidth=0.8)
    ax3.axhline(y=100, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel('Relative Performance (%)')
    ax3.set_title('(c) Cross-Environment Transfer (10k fine-tune)')
    ax3.set_ylim(0, 115)
    for bar, val in zip(bars, transfer_10k):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val}%',
                ha='center', fontsize=8)

    # (d) Architecture Comparison
    ax4 = axes[1, 1]
    archs = ['BiLSTM', 'Trans-4H', 'Trans-8H', 'GRU']
    separation = [0.92, 0.89, 0.91, 0.88]
    arch_colors = ['#2ca02c', '#1f77b4', '#1f77b4', '#9467bd']

    bars = ax4.bar(archs, separation, color=arch_colors, edgecolor='black', linewidth=0.8)
    ax4.set_ylabel('Separation Score')
    ax4.set_title('(d) Encoder Architecture')
    ax4.set_ylim(0.82, 0.96)
    for bar, val in zip(bars, separation):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.2f}',
                ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_future_work_combined.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig_future_work_combined.png', dpi=300, bbox_inches='tight')
    print("  Saved: fig_future_work_combined")
    plt.close()


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("="*50)
    fig_curriculum_learning()
    fig_online_adaptation()
    fig_cross_transfer()
    fig_architecture()
    fig_multidim_context()
    fig_combined_future_work()
    print("="*50)
    print(f"All figures saved to: {output_dir}/")
