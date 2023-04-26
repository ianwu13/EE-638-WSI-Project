import matplotlib.pyplot as plt

from slurm_data_getter import get_full_dir_info, get_baseline_slurm_info


def plot_best_accuracy(data, save_pth, baseline_acc=None):
    for n_l in [1,2,3]:
        acc_by_n_layers = []
        for n_edges in [2, 4, 8, 16, 32]:
            best_accuracy = max([i['average_score'] for i in data[n_edges][n_l]])
            acc_by_n_layers.append(best_accuracy)
        plt.plot([2, 4, 8, 16, 32], acc_by_n_layers, label=f'c16_{n_l}_layers')

    if baseline_acc is not None:
        plt.axhline(y=baseline_acc, color='r', linestyle='dotted')

    plt.ylabel('Accuracy/Avg Score')
    plt.xlabel('Number of Edges per Node')
    plt.xticks([2, 4, 8, 16, 32])
    plt.legend()
    
    plt.savefig(save_pth)
    
    plt.clf()


def plot_best_test_loss(data, save_pth):
    for n_l in [1,2,3]:
        loss_by_n_layers = []
        for n_edges in [2, 4, 8, 16, 32]:
            best_train_loss = min([i['test_loss'] for i in data[n_edges][n_l]])
            loss_by_n_layers.append(best_train_loss)
        plt.plot([2, 4, 8, 16, 32], loss_by_n_layers, label=f'c16_{n_l}_layers')

    plt.ylabel('Testing Loss')
    plt.xlabel('Number of Edges per Node')
    plt.xticks([2, 4, 8, 16, 32])
    plt.legend()

    plt.savefig(save_pth)

    plt.clf()


def main():
    c16_graphconv_path = '../../slurms/c16_graphconv'
    tcga_graphconv_path = '../../slurms/tcga_graphconv'
    baseline_c16_path = '../../slurms/baseline_dsmil/slurm-14491708.out'
    baseline_tcga_path = '../../slurms/baseline_dsmil/slurm-14488808.out'

    # dir path --> dict{n_knn_edges: dict{n_layers: list[results_by_epoch_dicts]}}

    c16_data = get_full_dir_info(c16_graphconv_path)
    c16_baseline_data = get_baseline_slurm_info(baseline_c16_path)
    print('C16')
    baseline = max([i['average_score'] for i in c16_baseline_data])
    print(baseline)
    plot_best_accuracy(c16_data, 'c16.png', baseline_acc=baseline)

    tcga_data = get_full_dir_info(tcga_graphconv_path)
    tcga_baseline_data = get_baseline_slurm_info(baseline_tcga_path)
    print('TCGA')
    baseline = max([i['average_score'] for i in tcga_baseline_data])
    print(baseline)
    plot_best_accuracy(tcga_data, 'tcga.png', baseline_acc=baseline)


if __name__ == '__main__':
    main()
