from time import strftime


class Parameters:
    # LT Trainer parameters
    accelerator = 'gpu'
    log_every_n_steps = 300
    max_epochs = 300
    enable_progress_bar = True
    check_val_every_n_epoch = 4
    node_features = 24

    # Datasets and paths
    dataset_name = 'denmark'  # ['METR-LA', 'Electricity']
    traffic_temporal_data_folder = r'/mnt/c/Users/Grid/Desktop/PhD/EV/code/EV_GNN/data/traffic/denmark/citypulse_traffic_raw_data_surrey_feb_jun_2014/traffic_feb_june'
    traffic_metadata_file = r'/mnt/c/Users/Grid/Desktop/PhD/EV/code/EV_GNN/data/traffic/denmark/trafficMetaData.csv'
    ev_temporal_data_folder = r'/mnt/c/Users/Grid/Desktop/PhD/EV/code/EV_GNN/data/ev/denmark/denamark_ev_station_availability/available_connectors_counts'
    ev_metadata_file = r'/mnt/c/Users/Grid/Desktop/PhD/EV/code/EV_GNN/data/ev/denmark/denamark_ev_station_availability/charging_stations.csv'
    chkpt_dir = ''


    # Training parameters
    device = 'cuda'
    model = 'GraphWavenet'  # 'gcn', 'gat', 'gcn1d', 'gcn1d-big', 'gConvLSTM', 'gConvGRU', 'DCRNN', 'GraphWavenet', 'AGCRNModel', 'miniLSTM', 'miniGRU'
    lags = 24
    prediction_window = 24
    time_series_step = 4
    batch_size = 128
    train_ratio = 0.7
    val_test_ratio = 0.5
    num_workers = 1
    num_nodes = 0
    num_of_traffic_nodes_limit = 50  # -1 for all nodes
    num_of_ev_nodes_limit = None
    early_stop_callback_flag = False
    lr = 3e-4
    test_eval = 10
    seed = 42

    # Model parameters
    emb_dim = 32
    dropout = 0.0
    num_layers=2
    graph_distance_threshold = 10

    # Execution flags
    logging = False
    save_ckpts = False
    save_logs = True
    reproducible = True
    verbose = False

    def __init__(self, params=None):
        # Show parser args
        if params.verbose:
            print("Parameters:")
            for name, value in vars(params).items():
                print(f"  {name}: {value}")

        # Copy all default class attributes as istance attribute
        for attr, value in type(self).__dict__.items():
            if not attr.startswith('__') and not callable(value):
                setattr(self, attr, value)

        # If a Namespace is passed, overwrite the values
        if params is not None:
            # If Namespace argparse, convert in dict
            items = (vars(params).items()
                     if not isinstance(params, dict)
                     else params.items())
            for name, val in items:
                # Assign only if attribute exists and val is not None
                if hasattr(self, name) and val is not None:
                    setattr(self, name, val)

        # Actions to execute when instance is created
        if self.num_of_traffic_nodes_limit != -1:
            self.num_nodes = self.num_of_traffic_nodes_limit




