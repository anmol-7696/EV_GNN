import glob
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as DataLoaderPyg
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse

from tsl.data import SpatioTemporalDataset, TemporalSplitter, SpatioTemporalDataModule
from tsl.data.preprocessing import MinMaxScaler
from tsl.datasets import MetrLA, Elergone
from torch_geometric.data import Data

from src.config import Parameters
from src.dataset.utils import create_adjacency_matrix


class EVDataModule(LightningDataModule):
    def __init__(self, run_params):
        super().__init__()
        self.run_params = run_params
        self.train_data = None
        self.test_data = None

        dataset = TrafficCSVTimeSeriesDataset(run_params, root_dir=run_params.dataset_path)
        self.num_station = dataset.number_of_station
        len_dataset = len(dataset)
        train_ratio = 0.7
        val_test_ratio = 0.5
        train_snapshots = int(train_ratio * len_dataset)
        val_test_snapshots = len_dataset - train_snapshots
        val_snapshots = int(val_test_ratio * val_test_snapshots)
        test_snapshots = len_dataset - train_snapshots - val_snapshots
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(dataset,
                                                                                       [train_snapshots, val_snapshots,
                                                                                        test_snapshots])

        if self.train_data is None:
            raise Exception("Dataset %s not supported" % self.run_params.dataset)
        self.train_loader = DataLoaderPyg(self.train_data, batch_size=run_params.batch_size,
                                          shuffle=True, drop_last=True)  #num_workers=4)
        self.val_loader = DataLoaderPyg(self.val_data, batch_size=run_params.batch_size, drop_last=True)  # num_workers=4)
        self.test_loader = DataLoaderPyg(self.test_data, batch_size=run_params.batch_size, drop_last=True)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class TrafficCSVTimeSeriesDataset(Dataset):
    _DEFAULT_COLUMNS = [
        "avgMeasuredTime",
        "avgSpeed",
        "extID",
        "medianMeasuredTime",
        "TIMESTAMP",
        "vehicleCount"]

    def __init__(self, params, root_dir, columns=None, lags=24, prediction_window=24, time_series_step=4, dtype=torch.float32, device="cpu"):
        self.root_dir = root_dir
        self.params = params
        self.columns = columns or self._DEFAULT_COLUMNS
        self.dtype = dtype
        self.device = torch.device(device)
        self.encoded_data = []
        self.lags = lags
        self.prediction_window = prediction_window
        self.time_series_step = time_series_step

        self.filepaths = sorted(glob.glob(os.path.join(root_dir, "*.csv")))
        if not self.filepaths:
            raise RuntimeError(f"Nessun CSV trovato in {root_dir}")

        self._load_files()
        self.preprocess_data()
        self.read_dataset()

    def preprocess_data(self):
        stacked_target = self.data_tensor
        stacked_target = stacked_target[:,:,-1]
        scaler = MinMaxScaler()
        scaler.fit(stacked_target)
        standardized_target = scaler.transform(stacked_target).T
        self.number_of_station = standardized_target.shape[1]

        self.features = [standardized_target[i: i + self.lags, :].T

                         for i in range(0, standardized_target.shape[0] - self.lags - self.prediction_window,
                                        self.time_series_step)
                         ]

        self.targets = [standardized_target[i:i + self.prediction_window, :].T
                        for i in range(self.lags, standardized_target.shape[0] - self.prediction_window,
                                       self.time_series_step)
                        ]
        print('Oki!')

    def _load_files(self, pad_value=-1.0):
        """
        Costruisce:
            self.df_all      # DataFrame indicizzato per TIMESTAMP, colonne MultiIndex (site, feature)
            self.data_tensor # tensore (N, T_max, M) con padding = pad_value
        """
        dfs = []  # lista DataFrame per sito
        sites = []  # id del sito (usiamo extID o il nome file)
        cont = 0
        for path in self.filepaths[:self.params.num_of_nodes_limit]:
            print(cont)
            cont += 1
            df = pd.read_csv(path, usecols=self.columns)

            # --- controlli e preparazione  -------------------------
            missing = set(self.columns) - set(df.columns)
            if missing:
                raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

            df = df.sort_values("TIMESTAMP").reset_index(drop=True)

            # ts in datetime → indice
            ts = pd.to_datetime(df["TIMESTAMP"])
            df["TIMESTAMP"] = ts
            df = df.set_index("TIMESTAMP")

            if df.index.has_duplicates:
                # scegli una strategia
                # df = df[~df.index.duplicated(keep="first")]          # A. scarta
                df = df.groupby(level=0).mean()  # B. media
                print('duplicati!"')

            # cast numerico, riordino colonne
            feature_cols = [c for c in self.columns if c != "TIMESTAMP"]
            df = df[feature_cols]  # mantiene ordine voluto
            df = df.apply(pd.to_numeric, errors="coerce")

            site_id = str(df["extID"].iloc[0])  # o usa os.path.basename(path).split(".")[0]
            # rimuovi colonna extID, _id, REPORT_ID se non ti servono come feature
            # df = df.drop(columns=["extID", "_id", "REPORT_ID"])

            dfs.append(df)
            sites.append(site_id)

        # ----------------------------------------------------------
        # 1️⃣ unione degli indici (outer join)
        union_index = dfs[0].index
        for d in dfs[1:]:
            union_index = union_index.union(d.index)

        # 2️⃣ riallinea e pad con valore costante
        dfs_aligned = [
            d.reindex(union_index).fillna(pad_value) for d in dfs
        ]

        # 3️⃣ concatena in colonna con MultiIndex (site, feature)
        df_all = pd.concat(dfs_aligned, axis=1, keys=sites, names=["site", "feature"])
        self.df_all = df_all.sort_index()  # (T_max, N*M)

        # 4️⃣ opzionale: converti in tensore (N, T_max, M)
        #     Manteniamo le feature nell'ordine self.columns
        B = len(dfs_aligned)
        T_max = len(union_index)
        M = len(self.columns)

        # costruisci array (B, T_max, M)
        data = np.stack([d.values.astype(np.float32) for d in dfs_aligned], axis=0)
        self.data_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)

        # memorizza dimensioni
        self.N, self.T, self.M = self.data_tensor.shape

    def get_edges(self):
        data = pd.read_csv(r'/mnt/c/Users/Grid/Desktop/PhD/EV/EV_GNN/src/dataset/denmark/trafficMetaData.csv')
        coordinates = list()
        for row in data.iterrows():
            p1_lat = row[1]['POINT_1_LAT']
            p1_long = row[1]['POINT_1_LNG']
            p2_lat = row[1]['POINT_2_LAT']
            p2_long = row[1]['POINT_2_LNG']
            p_mean_lat = (p1_lat + p2_lat) / 2.0
            p_mean_lng = (p1_long + p2_long) / 2.0
            coordinates.append((p_mean_lat, p_mean_lng))

        threshold = 10  # Soglia di distanza in km (ad esempio 1500 km)
        adj_matrix = create_adjacency_matrix(coordinates[:self.params.num_of_nodes_limit], threshold)
        edge_index, edge_weights = dense_to_sparse(adj_matrix)
        return edge_index.to('cuda'), edge_weights.to('cuda')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def read_dataset(self):
        edge_idx, edge_attr = self.get_edges()
        for i in range(len(self.features)):
            self.encoded_data.append(Data(x=torch.FloatTensor(self.features[i]),
                                          edge_index=edge_idx.long(),
                                          edge_attr=edge_attr.float(),
                                          y=torch.FloatTensor(self.targets[i])))

# class DatasetDenmark(Dataset):
#
#     def __init__(self, run_params):
#         self.params = run_params
#         self.lags = None
#         self.n_features = 1 * run_params.lags
#         self.features = None
#         self.features_corrupted = None
#         self.targets = None
#         self.features_temperatures = None
#         self.targets_temperatures = None
#         self.features_winds = None
#         self.targets_winds = None
#         self.number_of_station = None
#         self.encoded_data = []
#         self._read_json_data()
#         self._get_targets_and_features()
#         self.read_dataset(self.params.lags)
#
#     def _read_json_data(self):
#         with open(self.params.dataset_path) as f:
#             self._dataset = json.load(f)
#
#     def _get_targets_and_features(self):
#         # Power
#         stacked_target = np.stack(self._dataset["block"])
#
#         # # #ENDMOD
#         scaler = MinMaxScaler()
#         scaler.fit(stacked_target)
#         standardized_target = scaler.transform(stacked_target)
#         self.number_of_station = stacked_target.shape[1]
#
#         self.features = [standardized_target[i: i + self.lags, :].T
#
#                          for i in range(0, standardized_target.shape[0] - self.lags - self.params.prediction_window,
#                                         self.params.time_series_step)
#                          ]
#
#         self.targets = [standardized_target[i:i + self.params.prediction_window, :].T
#                         for i in range(self.lags, standardized_target.shape[0] - self.params.prediction_window,
#                                        self.params.time_series_step)
#                         ]
#
#     def __len__(self):
#         return len(self.features)
#
#     def __getitem__(self, idx):
#         return self.encoded_data[idx]
#
#     def read_dataset(self, lags):
#         self.lags = lags
#         self._get_edges()
#         self._get_targets_and_features()
#         for i in range(len(self.features)):
#             self.encoded_data.append(Data(x=torch.FloatTensor(self.features[i]),
#                                           edge_index=torch.LongTensor(self._edges),
#                                           edge_attr=torch.FloatTensor(self._edge_weights),
#                                           y=torch.FloatTensor(self.targets[i])))


def get_datamodule(run_params):
    # TSL-datasets style
    if run_params.dataset_name in ['METR-LA', 'electricity', 'solar']:
        if run_params.dataset_name == 'METR-LA':
            dataset = MetrLA(root='../data')
        elif run_params.dataset_name == 'electricity':
            dataset = Elergone(root='../data')
        else:
            raise ValueError(f'Dataset {run_params.dataset_name} not recognized')

        run_params.num_nodes = dataset.shape[1]
        connectivity = dataset.get_connectivity(threshold=0.1,
                                                include_self=True,
                                                normalize_axis=1,
                                                layout="edge_index")
        df_dataset = dataset.dataframe()
        # Initialize MinMaxScaler
        scaler = MinMaxScaler()

        # Apply the scaler to the DataFrame
        df_dataset = pd.DataFrame(scaler.fit_transform(df_dataset), columns=df_dataset.columns)
        torch_dataset = SpatioTemporalDataset(target=df_dataset,
                                              connectivity=connectivity,  # edge_index
                                              horizon=run_params.prediction_window,
                                              window=run_params.lags,
                                              stride=1)
        # Normalize data using mean and std computed over time and node dimensions

        splitter = TemporalSplitter(val_len=0.2, test_len=0.1)
        data_module_instance = SpatioTemporalDataModule(
            dataset=torch_dataset,
            # scalers=scalers,
            splitter=splitter,
            batch_size=run_params.batch_size,
            workers=2
        )

    elif run_params.dataset_name in ['denmark']:
        data_module_instance = EVDataModule(run_params)

        # If nodes are limited change number of nodes, otherwise use all nodes
        if run_params.num_of_nodes_limit == -1:
            run_params.num_nodes = data_module_instance.num_station
        else:
            run_params.num_nodes = run_params.num_of_nodes_limit
    else:
        raise ValueError('Define dataset name correct!')

    return data_module_instance, run_params


if __name__ == '__main__':
    # Parameters
    run_params = Parameters()
    run_params.dataset_path = r'/mnt/c/Users/Grid/Desktop/PhD/EV/EV_GNN/data/aarhus dataset/citypulse_traffic_raw_data_surrey_feb_jun_2014/traffic_feb_june'

    # Datamodule
    dm = EVDataModule(run_params)
    print('DM created!')