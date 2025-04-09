from time import strftime


class Parameters():
    # Trainer parameters
    accelerator = 'gpu'
    log_every_n_steps = 300
    max_epochs = 300
    enable_progress_bar = False
    check_val_every_n_epoch = 4
    node_features = 23
    time_series_step = 4

    # Datasets
    dataset_name = 'METR-LA'  # ['METR-LA', 'Electricity']
    dataset_path = '..'

    dropout = 0.0
    lr = 3e-4  #
    test_eval = 10
    model = 'dgm_gcn'  # 'dgm_gcn', 'dgm_gat', 'gcn', 'gat', 'mlp'

    early_stop_callback_flag = False

    lags = 24
    prediction_window = 24
    batch_size = 128
    num_nodes = 31
    logging = False
    save_ckpts = False
    save_logs = True
    chkpt_dir = ''
    reproducible = True
    seed = 42

    num_layers=2



