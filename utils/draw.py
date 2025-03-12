import matplotlib.pyplot as plt
import re
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 'large'

def parse_file(file_name):
    rounds, loss, acc = [], [], []

    # pattern1
    # Round   0, Average loss 1.377 Accuracy 25.776
    # Round   1, Average loss 0.950 Accuracy 27.316
    # pattern2
    # Round   1, Average loss 1.069, Train accuracy 0.563, Test accuracy 37.882
    # Round  93, Average loss 0.201, Train accuracy 0.962, Test accuracy 88.06
    pattern1 = r"Round\s+(\d+), Average loss ([\d.]+) Accuracy ([\d.]+)"
    pattern2 = r"^Round\s+(\d+),\s*Average loss ([\d.]+),\s*Train accuracy ([\d.]+),\s*Test accuracy ([\d.]+)$"
    with open(file_name, 'r') as file:
        flag = True
        content = file.read()
        matches = re.findall(pattern1, content, re.MULTILINE)
        if not matches:
            flag = False
            matches = re.findall(pattern2, content, re.MULTILINE)
        for match in matches:
            rounds.append(int(match[0]))
            loss.append(float(match[1]))
            if flag:
                acc.append(float(match[2]))
            else:
                acc.append(float(match[3]))
            
    print(f"{file_name} (round, max_acc) {statisitc(rounds, acc)}")
    print(len(rounds), len(loss), len(acc))

    return rounds, loss, acc

def draw_max_accuracy():
    # ./log/avg_rank4.log max accuracy: 86.623
    # 1000 1000 1000
    # ./log/piece3-1.log max accuracy: 84.253
    # 1000 1000 1000
    # ./log/piece3-2.log max accuracy: 86.558
    # 1000 1000 1000
    # ./log/piece3-3.log max accuracy: 86.948
    # 1000 1000 1000
    label = ['All Layers', 'Layers 1-4', 'Layers 5-8', 'Layers 9-12']
    max_acc = [86.623, 84.253, 86.558, 86.948]

    fig, ax = plt.subplots()
    bar_container = ax.bar(label, max_acc)
    ax.set(ylabel='Max Accuracy', ylim=(83, 89))
    ax.bar_label(bar_container, fmt='{:,.2f}')
    plt.tight_layout()
    plt.savefig('./png/lora_max_accuracy.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

def draw_time():
    # ./log/avg_rank4.log max accuracy: 86.623
    # 1000 1000 1000
    # ./log/piece3-1.log max accuracy: 84.253
    # 1000 1000 1000
    # ./log/piece3-2.log max accuracy: 86.558
    # 1000 1000 1000
    # ./log/piece3-3.log max accuracy: 86.948
    # 1000 1000 1000
    label = ['All Layers', 'Layers 1-4', 'Layers 5-8', 'Layers 9-12']
    max_acc = [86.623, 84.253, 86.558, 86.948]

    fig, ax = plt.subplots()
    bar_container = ax.bar(label, max_acc)
    ax.set(ylabel='Max Accuracy', ylim=(83, 89))
    ax.bar_label(bar_container, fmt='{:,.2f}')
    plt.tight_layout()
    plt.savefig('./png/lora_max_accuracy.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

def draw_loss():
    pass

def draw_acc(round, loss, acc, label, path):
    num = len(label)

    fig, ax = plt.subplots()
    for i in range(num):
        ax.plot(round[i], acc[i], label=label[i])
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(path)

def draw_accuracy(r_all, a_all, r_s1, a_s1, r_s2, a_s2, r_s3, a_s3):
    # plt.rc('lines', linewidth=2.5)
    fig, ax = plt.subplots()
    # label = ['All Layers', 'Layers 1-4', 'Layers 5-8', 'Layers 9-12']
    label = ['[8, 8, 8]', '[4, 8, 12]', '[4, 4, 16]', '[12, 8, 4]', '[12, 12, 12]']

    ax.plot(r_all, a_all, label=label[0], linestyle='-')
    ax.plot(r_s1, a_s1, label=label[1], linestyle='--')
    ax.plot(r_s2, a_s2, label=label[2], linestyle='-.')
    # ax.plot(r_s3, a_s3, label=label[3], linestyle=':')

    ax.legend(handlelength=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('./png/lora_accuracy.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

def statisitc(round, acc, tol=0.5):
    max_acc = max(acc)
    round_ = 0
    for i in range(len(acc)):
        if acc[i] + tol >= max_acc:
            round_ = round[i]
            break
    return round_, max_acc
if __name__ == '__main__':
    # draw_max_accuracy()

    ACA, KAIMING, LORAGA, PISSA, GAUSS = 0, 1, 1, 1, 0 # 1, 1, 1, 0, 0
    alpha = 0

    if alpha == 0.01:
    # alpha 0.01
        aca = './log/bert-base-uncased_aca_agnews_epoch100_C0.1_noniid_dirichlet_alpha0.01_localep1_ep1_lr0.02.txt'
        gauss = './log/bert-base-uncased_gauss_agnews_epoch100_C0.1_noniid_dirichlet_alpha0.01_localep1_ep1_lr0.1.txt'
        loraga = './log/bert-base-uncased_loraga_agnews_epoch100_C0.1_noniid_dirichlet_alpha0.01_localep1_ep1_lr0.04.txt'
        pissa = './log/bert-base-uncased_pissa_agnews_epoch100_C0.1_noniid_dirichlet_alpha0.01_localep1_ep1_lr0.1.txt'
        label = ['aca', 'gauss', 'loraga', 'pissa']
        r_all, l_all, a_all = parse_file(aca)
        r_s1, l_s1, a_s1 = parse_file(gauss)
        r_s2, l_s2, a_s2 = parse_file(loraga)
        r_s3, l_s3, a_s3 = parse_file(pissa)
        draw_acc([r_all, r_s1, r_s2, r_s3], [l_all, l_s1, l_s2, l_s3], [a_all, a_s1, a_s2, a_s3], label, './png/alpha-0.01.png')
    elif alpha == 1.0:
        aca = './log/bert-base-uncased_aca_agnews_epoch100_C0.1_noniid_dirichlet_alpha1.0_localep1__ep1_lr0.04.txt'
        gauss = './log/bert-base-uncased_gauss_agnews_epoch100_C0.1_noniid_dirichlet_alpha1.0_localep1_rank8.txt'
        loraga = './log/bert-base-uncased_loraga_agnews_epoch100_C0.1_noniid_dirichlet_alpha1.0_localep1__ep1_lr0.1.txt'
        # pissa = './log/bert-base-uncased_pissa_agnews_epoch100_C0.1_noniid_dirichlet_alpha1.0_localep1_ep1_lr0.2_0.txt'
        pissa = './log/bert-base-uncased_pissa_agnews_epoch100_C0.1_noniid_dirichlet_alpha1.0_localep1_ep1_lr0.2.txt'
        # pissa = './log/bert-base-uncased_pissa_agnews_epoch100_C0.1_noniid_dirichlet_alpha1.0_localep1_ep1_lr0.1.txt'

        svd = './log/bert-base-uncased_svd_agnews_epoch100_C0.1_noniid_dirichlet_alpha1.0_localep1__ep1_lr0.1.txt'
        label = ['aca', 'gauss', 'loraga', 'pissa', 'svd']
        r_all, l_all, a_all = parse_file(aca)
        r_s1, l_s1, a_s1 = parse_file(gauss)
        r_s2, l_s2, a_s2 = parse_file(loraga)
        r_s3, l_s3, a_s3 = parse_file(pissa)
        r_s4, l_s4, a_s4 = parse_file(svd)
        draw_acc([r_all, r_s1, r_s2, r_s3, r_s4], [l_all, l_s1, l_s2, l_s3, l_s4], [a_all, a_s1, a_s2, a_s3, a_s4], label, './png/alpha-1.0.png')
    elif alpha == 10.0:
        aca = './log/bert-base-uncased_aca_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1__ep1_lr0.04.txt'
        gauss = './log/bert-base-uncased_gaussian_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1_ep1_lr0.1.txt'
        kaming = './log/bert-base-uncased_kaming_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1_ep1_lr0.1.txt'
        loraga = './log/bert-base-uncased_loraga_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1__ep1_lr0.1.txt'
        pissa = './log/bert-base-uncased_pissa_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1_ep1_lr0.1.txt'
        kaming_plus = './log/bert-base-uncased_kaming_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1_ep1_lr0.04.txt'
        svd = './log/bert-base-uncased_svd_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1__ep1_lr0.2.txt'
        label = ['aca', 'gauss', 'kaming', 'loraga', 'pissa', 'kaming_plus', 'svd']
        r_all, l_all, a_all = parse_file(aca)
        r_s1, l_s1, a_s1 = parse_file(gauss)
        r_s2, l_s2, a_s2 = parse_file(kaming)
        r_s3, l_s3, a_s3 = parse_file(loraga)
        r_s4, l_s4, a_s4 = parse_file(pissa)
        r_s5, l_s5, a_s5 = parse_file(kaming_plus)
        r_s6, l_s6, a_s6 = parse_file(svd)
        draw_acc([r_all, r_s1, r_s2, r_s3, r_s4, r_s5, r_s6], [l_all, l_s1, l_s2, l_s3, l_s4, l_s5, l_s6], [a_all, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6], label, './png/alpha-10.0.png')
        # draw_acc([r_all, r_s1, r_s2, r_s4], [l_all, l_s1, l_s2, l_s4], [a_all, a_s1, a_s2, a_s4], label, './png/alpha-10.0.png')
    else:
        pass

    
    if ACA:
        alpha_0_01 = './log/bert-base-uncased_aca_agnews_epoch100_C0.1_noniid_dirichlet_alpha0.01_localep1_ep1_lr0.02.txt'
        alpha_1_0 = './log/bert-base-uncased_aca_agnews_epoch100_C0.1_noniid_dirichlet_alpha1.0_localep1__ep1_lr0.04.txt'
        alpha_5_0 = './log/bert-base-uncased_aca_agnews_epoch100_C0.1_noniid_dirichlet_alpha5.0_localep1_ep1_lr0.04.txt'
        alpha_10_0 = './log/bert-base-uncased_aca_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1__ep1_lr0.04.txt'
        alpha_1000 = './log/bert-base-uncased_aca_agnews_epoch100_C0.1_noniid_dirichlet_alpha1000.0_localep1_ep1_lr0.04.txt'
        label = ['alpha 0.01', 'alpha 1.0', 'alpha 5.0', 'alpha 10.0', 'alpha 1000']
        r_0_01, l_0_01, a_0_01 = parse_file(alpha_0_01)
        r_1_0, l_1_0, a_1_0 = parse_file(alpha_1_0)
        r_5_0, l_5_0, a_5_0 = parse_file(alpha_5_0)
        r_10_0, l_10_0, a_10_0 = parse_file(alpha_10_0)
        r_1000, l_1000, a_1000 = parse_file(alpha_1000)
        draw_acc([r_0_01, r_1_0, r_5_0, r_10_0, r_1000], [l_0_01, l_1_0, l_5_0, l_10_0, l_1000], [a_0_01, a_1_0, a_5_0, a_10_0, a_1000], label, './png/aca_alpha.png')
    if KAIMING:
        alpha_0_01 = './log/bert-base-uncased_kaiming_agnews_epoch100_C0.1_noniid_dirichlet_alpha0.01_localep1_ep1_lr0.1.txt'
        # alpha_0_01 = './log/bert-base-uncased_kaiming_agnews_epoch93_C0.1_noniid_dirichlet_alpha0.01_localep1_ep1_lr0.04.txt'
        alpha_1_0 = './log/bert-base-uncased_kaiming_agnews_epoch100_C0.1_noniid_dirichlet_alpha1.0_localep1_rank8_lr0.1.txt'
        alpha_5_0 = './log/bert-base-uncased_kaming_agnews_epoch100_C0.1_noniid_dirichlet_alpha5.0_localep1_ep1_lr0.1.txt'
        alpha_10_0 = './log/bert-base-uncased_kaming_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1_ep1_lr0.1.txt'
        alpha_1000 = './log/bert-base-uncased_kaiming_agnews_epoch100_C0.1_noniid_dirichlet_alpha1000.0_localep1_rank8.txt'
        label = ['alpha 0.01', 'alpha 1.0', 'alpha 5.0', 'alpha 10.0', 'alpha 1000']
        r_0_01, l_0_01, a_0_01 = parse_file(alpha_0_01)
        r_1_0, l_1_0, a_1_0 = parse_file(alpha_1_0)
        r_5_0, l_5_0, a_5_0 = parse_file(alpha_5_0)
        r_10_0, l_10_0, a_10_0 = parse_file(alpha_10_0)
        r_1000, l_1000, a_1000 = parse_file(alpha_1000)
        draw_acc([r_0_01, r_1_0, r_5_0, r_10_0, r_1000], [l_0_01, l_1_0, l_5_0, l_10_0, l_1000], [a_0_01, a_1_0, a_5_0, a_10_0, a_1000], label, './png/kaiming_alpha.png')
    if LORAGA:
        # alpha_0_01 = './log/bert-base-uncased_loraga_agnews_epoch100_C0.1_noniid_dirichlet_alpha0.01_localep1_ep1_lr0.04.txt'
        alpha_0_01 = './log/bert-base-uncased_loraga_agnews_epoch118_C0.1_noniid_dirichlet_alpha0.01_localep1_ep1_lr0.04.txt'
        alpha_1_0 = './log/bert-base-uncased_loraga_agnews_epoch100_C0.1_noniid_dirichlet_alpha1.0_localep1__ep1_lr0.1.txt'
        alpha_5_0 = './log/bert-base-uncased_loraga_agnews_epoch100_C0.1_noniid_dirichlet_alpha5.0_localep1_ep1_lr0.1.txt'
        alpha_10_0 = './log/bert-base-uncased_loraga_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1__ep1_lr0.1.txt'
        alpha_1000 = './log/bert-base-uncased_loraga_agnews_epoch100_C0.1_noniid_dirichlet_alpha1000.0_localep1_ep1_lr0.1.txt'
        label = ['alpha 0.01', 'alpha 1.0', 'alpha 5.0', 'alpha 10.0', 'alpha 1000']
        r_0_01, l_0_01, a_0_01 = parse_file(alpha_0_01)
        r_1_0, l_1_0, a_1_0 = parse_file(alpha_1_0)
        r_5_0, l_5_0, a_5_0 = parse_file(alpha_5_0)
        r_10_0, l_10_0, a_10_0 = parse_file(alpha_10_0)
        r_1000, l_1000, a_1000 = parse_file(alpha_1000)
        draw_acc([r_0_01, r_1_0, r_5_0, r_10_0, r_1000], [l_0_01, l_1_0, l_5_0, l_10_0, l_1000], [a_0_01, a_1_0, a_5_0, a_10_0, a_1000], label, './png/loraga_alpha.png')
    if PISSA:
        # alpha_0_01 = './log/bert-base-uncased_pissa_agnews_epoch100_C0.1_noniid_dirichlet_alpha0.01_localep1_ep1_lr0.1.txt'
        alpha_0_01 = './log/bert-base-uncased_pissa_agnews_epoch100_C0.1_noniid_dirichlet_alpha0.01_localep1_ep1_lr0.04.txt'
        alpha_1_0 = './log/bert-base-uncased_pissa_agnews_epoch100_C0.1_noniid_dirichlet_alpha1.0_localep1_ep1_lr0.1.txt'
        alpha_5_0 = './log/bert-base-uncased_pissa_agnews_epoch100_C0.1_noniid_dirichlet_alpha5.0_localep1_ep1_lr0.1.txt'
        alpha_10_0 = './log/bert-base-uncased_pissa_agnews_epoch100_C0.1_noniid_dirichlet_alpha10.0_localep1_ep1_lr0.1.txt'
        alpha_1000 = './log/bert-base-uncased_pissa_agnews_epoch100_C0.1_noniid_dirichlet_alpha1000.0_localep1_ep1_lr0.1.txt'
        
        label = ['alpha 0.01', 'alpha 1.0', 'alpha 5.0', 'alpha 10.0', 'alpha 1000']
        r_0_01, l_0_01, a_0_01 = parse_file(alpha_0_01)
        r_1_0, l_1_0, a_1_0 = parse_file(alpha_1_0)
        r_5_0, l_5_0, a_5_0 = parse_file(alpha_5_0)
        r_10_0, l_10_0, a_10_0 = parse_file(alpha_10_0)
        r_1000, l_1000, a_1000 = parse_file(alpha_1000)
        draw_acc([r_0_01, r_1_0, r_5_0, r_10_0, r_1000], [l_0_01, l_1_0, l_5_0, l_10_0, l_1000], [a_0_01, a_1_0, a_5_0, a_10_0, a_1000], label, './png/pissa_alpha.png')
    if GAUSS:
        pass
    else:
        pass
    pass