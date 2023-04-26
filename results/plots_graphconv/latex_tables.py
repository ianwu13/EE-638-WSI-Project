import matplotlib.pyplot as plt

from sklearn.metrics import recall_score

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


def table_results_c16(data, baseline_data, rd=4):
    print('\\begin{figure}[h!]')
    print('\\centering')
    print('\\begin{tabular}{ |p{2.5cm}||p{1.5cm}|p{1.5cm}|p{1.5cm}|  }')
    print(' \\hline')
    print(' \\multicolumn{4}{|c|}{\\textbf{Camelyon 16 Results}} \\\\')
    print(' \\hline')
    print(' Model&Accuracy&Sensitivity&AUC\\\\')
    print(' \\hline')
    sor = sorted(baseline_data, key=lambda x: -x['average_score'])
    sensitivity = recall_score(sor[0]['labels'], sor[0]['preds'], pos_label=0)
    print(f" DSMIL Baseline&{round(sor[0]['average_score'], rd)}&{round(sensitivity, rd)}&{round(sor[0]['auc_tumor_1'], rd)} \\\\")
    print(' \hline')
    # print('BASELINE')
    # print(sor[0]['average_score'])
    # print(sensitivity)
    # print(sor[0]['auc_tumor_1'])

    for n_l in [1,2,3]:
        for n_edges in [2, 4, 8, 16, 32]:
            dat = data[n_edges][n_l]

            sor = sorted(dat, key=lambda x: -x['average_score'])
            sensitivity = recall_score(sor[0]['labels'], sor[0]['preds'], pos_label=0)
            
            print(f" {n_l} Layers, k={n_edges}&{round(sor[0]['average_score'], rd)}&{round(sensitivity, rd)}&{round(sor[0]['auc_tumor_1'], rd)} \\\\")
            # print(f'NL: {n_l}; NE: {n_edges}')
            # print(sor[0]['average_score'])
            # print(sensitivity)
            # print(sor[0]['auc_tumor_1'])
        print(' \hline')

    # print(' \hline')
    print('\end{tabular}')
    print('  \\caption{Table of Results for Camelyon 16}')
    print('  \\label{c16_res}')
    print('\\end{figure}')


def table_results_tcga(data, baseline_data, rd=4):
    print('\\begin{figure*}[t!]')
    print('\\centering')
    print('\\begin{tabular}{ |p{3cm}||p{2cm}|p{3cm}|p{3cm}|p{2cm}|p{2cm}|  }')
    print(' \\hline')
    print(' \\multicolumn{6}{|c|}{\\textbf{TCGA Results}} \\\\')
    print(' \\hline')
    print(' Model&Accuracy&Sensitivity LUAD&Sensitivity LUSC&AUC LUAD&AUC LUSC\\\\')
    print(' \\hline')

    sor = sorted(baseline_data, key=lambda x: -x['average_score'])
    sensitivity_lusc = recall_score(sor[0]['labels'], sor[0]['preds'], pos_label=0)
    sensitivity_luad = recall_score(sor[0]['labels'], sor[0]['preds'], pos_label=1)
    print(f" DSMIL Baseline&{round(sor[0]['average_score'], rd)}&{round(sensitivity_luad, rd)}&{round(sensitivity_lusc, rd)}&{round(sor[0]['auc_LUAD'], rd)}&{round(sor[0]['auc_LUSC'], rd)} \\\\")
    print(' \hline')
    # print('BASELINE')
    # print(sor[0]['average_score'])
    # print(sensitivity_luad)
    # print(sensitivity_lusc)
    # print(sor[0]['auc_LUAD'])
    # print(sor[0]['auc_LUSC'])

    for n_l in [1,2,3]:
        for n_edges in [2, 4, 8, 16, 32]:
            dat = data[n_edges][n_l]

            sor = sorted(dat, key=lambda x: -x['average_score'])
            sensitivity_lusc = recall_score(sor[0]['labels'], sor[0]['preds'], pos_label=0)
            sensitivity_luad = recall_score(sor[0]['labels'], sor[0]['preds'], pos_label=1)
            
            print(f" {n_l} Layers, k={n_edges}&{round(sor[0]['average_score'], rd)}&{round(sensitivity_luad, rd)}&{round(sensitivity_lusc, rd)}&{round(sor[0]['auc_LUAD'], rd)}&{round(sor[0]['auc_LUSC'], rd)} \\\\")
            # print(f'NL: {n_l}; NE: {n_edges}')
            # print(sor[0]['average_score'])
            # print(sensitivity_luad)
            # print(sensitivity_lusc)
            # print(sor[0]['auc_LUAD'])
            # print(sor[0]['auc_LUSC'])
        print(' \hline')

    # print(' \hline')
    print('\end{tabular}')
    print('  \\caption{Table of Results for TCGA}')
    print('  \\label{tcga_res}')
    print('\\end{figure*}')


def main():
    c16_graphconv_path = '../../slurms/c16_graphconv'
    tcga_graphconv_path = '../../slurms/tcga_graphconv'
    baseline_c16_path = '../../slurms/baseline_dsmil/slurm-14491708.out'
    baseline_tcga_path = '../../slurms/baseline_dsmil/slurm-14488808.out'

    # dir path --> dict{n_knn_edges: dict{n_layers: list[results_by_epoch_dicts]}}

    c16_data = get_full_dir_info(c16_graphconv_path)
    c16_baseline_data = get_baseline_slurm_info(baseline_c16_path, dset='Camelyon16')
    print()
    print('C16')
    print()
    table_results_c16(c16_data, c16_baseline_data)
    '''
    baseline = max([i['average_score'] for i in c16_baseline_data])
    print(baseline)
    plot_best_accuracy(c16_data, 'c16.png', baseline_acc=baseline)
    '''
    
    tcga_data = get_full_dir_info(tcga_graphconv_path)
    tcga_baseline_data = get_baseline_slurm_info(baseline_tcga_path, dset='TCGA-lung-default')
    print()
    print('TCGA')
    print()
    table_results_tcga(tcga_data, tcga_baseline_data)
    '''
    baseline = max([i['average_score'] for i in tcga_baseline_data])
    print(baseline)
    plot_best_accuracy(tcga_data, 'tcga.png', baseline_acc=baseline)
    '''


if __name__ == '__main__':
    main()
