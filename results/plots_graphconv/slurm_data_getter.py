import os


# line string --> Dict of data
def line_to_dict(line, dataset):
    # EXAMPLE CAMELYON16 LINE:
    # Epoch [98/100] train loss: 0.3339 test loss: 0.3934, average score: 0.8750, AUC: class-0>>0.9353333333333333|class-1>>0.9506666666666668
    # EXAMPLE TCGA LINE:
    # Epoch [1/100] train loss: 0.5210 test loss: 0.3977, average score: 0.9190, auc_LUAD: 0.9753, auc_LUSC: 0.9787
    splt = line.split(' ')
    data = {}
    
    data['epoch'] = int(splt[2].split('/')[0].lstrip('['))
    data['train_loss'] = float(splt[5])
    data['test_loss'] = float(splt[8].rstrip(','))
    data['average_score'] = float(splt[11].rstrip(','))

    # TODO: DATASET DEPENDENT
    if dataset == 'TCGA-lung-default':
        data['auc_LUAD'] = float(splt[13].rstrip(','))
        data['auc_LUSC'] = float(splt[15])
    elif dataset == 'Camelyon16':
        tmp = splt[13].split('|')
        data['auc_norm_0'] = float(tmp[0].split('>>')[1])
        data['auc_tumor_1'] = float(tmp[1].split('>>')[1])
    else:
        try:
            data['auc_norm_0'] = float(splt[13].split('>>')[1])
        except:
            data['auc_norm_0'] = None

    return data


# slurm_output path --> (n_layers, list[results_by_epoch_dicts])
def get_slurm_info(slurm_pth):
    f = open(slurm_pth, 'r')

    line = f.readline()
    while line != '':
        if line.startswith('*-*-ID-*-* '):
            break
        line = f.readline()
    
    if line == '':
        raise Exception('Slurm file at path {slurm_pth} does not contain identifier line (starts with "*-*-ID-*-* ")')
        
    n_layers = int(line.split(' ')[5])
    dset = line.split(' ')[-1].rstrip('\n')

    dat = []
    while line != '':
        if line.startswith(' Epoch ['):
            dat_dict = line_to_dict(line, dset)
            line = f.readline()  # TRUE LABELS
            line = f.readline().lstrip('[').rstrip(']\n')
            try:
                dat_dict['labels'] = [float(i) for i in line.split(',')]
            except:
                for _ in range(11):
                    line = f.readline()
                try:
                    line = line.split('[')[-1].rstrip(']\n')
                    dat_dict['labels'] = [float(i) for i in line.split(',')]
                except:
                    print(line)
                    exit()

            line = f.readline()  # PREDICTIONS
            line = f.readline().lstrip('[').rstrip(']\n')
            dat_dict['preds'] = [float(i) for i in line.split(',')]

            dat.append(dat_dict)

        line = f.readline()
    f.close()

    return (n_layers, dat)


# slurm_output path --> list[results_by_epoch_dicts]
def get_baseline_slurm_info(slurm_pth, dset=None):
    f = open(slurm_pth, 'r')

    line = f.readline()

    dat = []
    while line != '':
        if line.startswith(' Epoch ['):
            dat_dict = line_to_dict(line, dset)
            line = f.readline()  # TRUE LABELS
            line = f.readline().lstrip('[').rstrip(']\n')
            dat_dict['labels'] = [float(i) for i in line.split(',')]

            line = f.readline()  # PREDICTIONS
            line = f.readline().lstrip('[').rstrip(']\n')
            dat_dict['preds'] = [float(i) for i in line.split(',')]

            dat.append(dat_dict)

        line = f.readline()
    f.close()

    return dat


# dir path --> dict{n_knn_edges: dict{n_layers: list[results_by_epoch_dicts]}}
def get_full_dir_info(dir_path):
    dat = {}
    for n_knn_edges in [2, 4, 8, 16, 32]:
        edge_dir_pth = '/'.join([dir_path, f'{n_knn_edges}_edges'])
        slurm_paths = list(filter(lambda x: x.startswith('slurm-'), os.listdir(edge_dir_pth)))

        dat2 = {}
        for pth in slurm_paths:
            dat3 = get_slurm_info('/'.join([edge_dir_pth, pth]))
            dat2[dat3[0]] = dat3[1]

        dat[n_knn_edges] = dat2
    
    return dat
