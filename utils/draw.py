import matplotlib.pyplot as plt
import re

plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 'large'

# ============================================================
# Global Config for Plotting
# ============================================================

# Config format: list of dicts, each dict contains:
#   - files: list of log file paths
#   - labels: list of labels for each file
#   - output: output image path
#   - title: (optional) plot title

PLOT_CONFIG = [
    # Example: Compare different alpha values
    # {
    #     "name": "alpha_comparison",
    #     "files": [
    #         "./log/bert-base-uncased_aca_agnews_epoch100_C0.1_noniid_dirichlet_alpha0.01_localep1_ep1_lr0.02.txt",
    #         "./log/bert-base-uncased_aca_agnews_epoch100_C0.1_noniid_dirichlet_alpha1.0_localep1_ep1_lr0.04.txt",
    #     ],
    #     "labels": ["alpha 0.01", "alpha 1.0"],
    #     "output": "./png/alpha_comparison.png",
    # },
    # Example: Compare different init methods
    # {
    #     "name": "init_comparison",
    #     "files": [
    #         "./log/bert-base-uncased_aca_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1_ep1_lr0.04.txt",
    #         "./log/bert-base-uncased_kaming_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1_ep1_lr0.1.txt",
    #         "./log/bert-base-uncased_loraga_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1_ep1_lr0.1.txt",
    #     ],
    #     "labels": ["ACA", "Kaiming", "LoRAGA"],
    #     "output": "./png/init_comparison.png",
    # },
]


def parse_file(file_name):
    """
    Parse log file and extract round, loss, train_acc, test_acc.
    
    Expected format:
        Round   0, Average loss 1.377, Train accuracy 0.563, Test accuracy 37.882
    
    Returns:
        rounds: list of round numbers
        loss: list of average loss values
        train_acc: list of train accuracy values
        test_acc: list of test accuracy values
    """
    rounds, loss, train_acc, test_acc = [], [], [], []

    # Pattern matches: Round   0, Average loss 1.377, Train accuracy 0.563, Test accuracy 37.882
    pattern = r"^Round\s+(\d+),\s*Average loss ([\d.]+),\s*Train accuracy ([\d.]+),\s*Test accuracy ([\d.]+)$"
    
    with open(file_name, 'r') as file:
        content = file.read()
        matches = re.findall(pattern, content, re.MULTILINE)
        for match in matches:
            rounds.append(int(match[0]))
            loss.append(float(match[1]))
            train_acc.append(float(match[2]))
            test_acc.append(float(match[3]))
    
    stats = statisitc(rounds, test_acc)
    print(f"{file_name}")
    print(f"  Max accuracy: {stats['max_acc']:.3f} at round {stats['round_max']}")
    print(f"  First exceed threshold: round {stats['first_exceed']}")
    print(f"  Window exceed ratio: {stats['window_exceed_ratio']:.3f}")
    print(f"  Last window avg: {stats['last_window_avg']:.3f}")
    print(f"  Total records: {len(rounds)}")

    return rounds, loss, train_acc, test_acc


def draw_acc(data_list, labels, output_path, xlabel='Round', ylabel='Test Accuracy'):
    """
    Draw accuracy curves for multiple experiments.
    
    Args:
        data_list: list of tuples (rounds, loss, train_acc, test_acc)
        labels: list of labels for each curve
        output_path: output image path
        xlabel: x-axis label
        ylabel: y-axis label
    """
    fig, ax = plt.subplots()
    
    for i, (rounds, _, _, test_acc) in enumerate(data_list):
        ax.plot(rounds, test_acc, label=labels[i])
    
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


def draw_loss(data_list, labels, output_path, xlabel='Round', ylabel='Loss'):
    """
    Draw loss curves for multiple experiments.
    
    Args:
        data_list: list of tuples (rounds, loss, train_acc, test_acc)
        labels: list of labels for each curve
        output_path: output image path
        xlabel: x-axis label
        ylabel: y-axis label
    """
    fig, ax = plt.subplots()
    
    for i, (rounds, loss, _, _) in enumerate(data_list):
        ax.plot(rounds, loss, label=labels[i])
    
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


def draw_train_test_acc(data_list, labels, output_path):
    """
    Draw both train and test accuracy curves.
    
    Args:
        data_list: list of tuples (rounds, loss, train_acc, test_acc)
        labels: list of labels for each curve
        output_path: output image path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (rounds, _, train_acc, test_acc) in enumerate(data_list):
        ax1.plot(rounds, train_acc, label=labels[i])
        ax2.plot(rounds, test_acc, label=labels[i])
    
    ax1.legend()
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Train Accuracy')
    ax2.legend()
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Test Accuracy')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


def statisitc(rounds, acc, tol=0.5, window_size=5, acc_threshold=None):
    """
    Compute statistics for accuracy data.
    
    Args:
        rounds: list of round numbers
        acc: list of accuracy values
        tol: tolerance for finding round reaching max accuracy
        window_size: size of sliding window for threshold statistics
        acc_threshold: accuracy threshold for window statistics (default: max_acc - 1.0)
    
    Returns:
        dict with statistics:
            - round_max: round where accuracy reaches max within tolerance
            - max_acc: maximum accuracy
            - first_exceed: first round exceeding acc_threshold (None if never)
            - window_exceed_ratio: ratio of windows where avg accuracy exceeds threshold
            - last_window_avg: average accuracy in last window
    """
    max_acc = max(acc)
    round_max = 0
    for i in range(len(acc)):
        if acc[i] + tol >= max_acc:
            round_max = rounds[i]
            break
    
    # Set default threshold if not provided
    if acc_threshold is None:
        acc_threshold = max_acc - 1.0
    
    # Find first round exceeding threshold
    first_exceed = None
    for i in range(len(acc)):
        if acc[i] >= acc_threshold:
            first_exceed = rounds[i]
            break
    
    # Compute window statistics: ratio of windows where avg accuracy exceeds threshold
    window_exceed_count = 0
    total_windows = max(1, len(acc) - window_size + 1)
    for i in range(len(acc) - window_size + 1):
        window_avg = sum(acc[i:i + window_size]) / window_size
        if window_avg >= acc_threshold:
            window_exceed_count += 1
    window_exceed_ratio = window_exceed_count / total_windows
    
    # Average accuracy in last window
    if len(acc) >= window_size:
        last_window_avg = sum(acc[-window_size:]) / window_size
    else:
        last_window_avg = sum(acc) / len(acc) if acc else 0
    
    return {
        'round_max': round_max,
        'max_acc': max_acc,
        'first_exceed': first_exceed,
        'window_exceed_ratio': window_exceed_ratio,
        'last_window_avg': last_window_avg,
    }


def run_from_config(config):
    """
    Run plotting from a single config dict.
    
    Args:
        config: dict with keys 'name', 'files', 'labels', 'output'
    """
    print(f"\n=== Processing: {config['name']} ===")
    data_list = []
    for file_path in config['files']:
        data = parse_file(file_path)
        data_list.append(data)
    draw_acc(data_list, config['labels'], config['output'])


def run_all_configs():
    """
    Run all plotting configs defined in PLOT_CONFIG.
    """
    for config in PLOT_CONFIG:
        run_from_config(config)


if __name__ == '__main__':
    # Run all configs
    run_all_configs()
