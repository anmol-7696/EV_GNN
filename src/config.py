from time import strftime


class Parameters():
    # Trainer parameters
    accelerator = 'gpu'
    log_every_n_steps = 300
    max_epochs = 300
    enable_progress_bar = True
    check_val_every_n_epoch = 4
    node_features = 24
    time_series_step = 4

    # Datasets
    dataset_name = 'denmark'  # ['METR-LA', 'Electricity']
    dataset_path = r'/mnt/c/Users/Grid/Desktop/PhD/EV/EV_GNN/data/aarhus dataset/citypulse_traffic_raw_data_surrey_feb_jun_2014/traffic_feb_june'

    dropout = 0.0
    lr = 3e-4  #
    test_eval = 10
    model = 'GraphWavenet'  # 'gcn', 'gat', 'gcn1d', 'gcn1d-big', 'gConvLSTM', 'gConvGRU', 'DCRNN', 'GraphWavenet', 'AGCRNModel', 'miniLSTM', 'miniGRU'

    early_stop_callback_flag = False

    emb_dim = 32
    lags = 24
    prediction_window = 24
    batch_size = 128
    num_nodes = 1
    num_of_nodes_limit = 20  # -1 for all nodes
    logging = False
    save_ckpts = False
    save_logs = True
    chkpt_dir = ''
    reproducible = True
    seed = 42

    num_layers=2



