import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as DataLoaderPyg
import pandas as pd
import torch

from tsl.data import SpatioTemporalDataset, TemporalSplitter, SpatioTemporalDataModule
from tsl.data.preprocessing import MinMaxScaler
from tsl.datasets import MetrLA, Elergone



class MyDataModule(pl.LightningDataModule):
    def __init__(self, run_params):
        super().__init__()
        self.run_params = run_params
        self.train_data = None
        self.test_data = None

        dataset = Dataset_custom(self.run_params)
        self.num_station = dataset.number_of_station
        len_dataset = len(dataset)
        train_ratio = 0.7
        val_test_ratio = 0.5
        train_snapshots = int(train_ratio * len_dataset)
        val_test_snapshots = len_dataset - train_snapshots
        val_snapshots = int(val_test_ratio * val_test_snapshots)
        test_snapshots = len_dataset - train_snapshots - val_snapshots
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(dataset,
                                                             [train_snapshots, val_snapshots, test_snapshots])

        if self.train_data is None:
            raise Exception("Dataset %s not supported" % self.run_params.dataset)
        self.train_loader = DataLoaderPyg(self.train_data, batch_size=run_params.batch_size, shuffle=True)  #num_workers=4)
        self.val_loader = DataLoaderPyg(self.val_data, batch_size=run_params.batch_size)  # num_workers=4)
        self.test_loader = DataLoaderPyg(self.test_data, batch_size=run_params.batch_size)


    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader



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

    elif run_params.dataset_name in ['PV', 'wind']:
        if run_params.dataset_name == 'PV':
            run_params.dataset_path = '../data/Generated_time_series_output_31_with_weigth_multivariate_and_time.json'
        elif run_params.dataset_name == 'wind':
            run_params.dataset_path = '../data/wind_dataset.json'
        data_module_instance = MyDataModule(run_params)
        run_params.num_nodes = data_module_instance.num_station
    else:
        raise ValueError('Define dataset name correct!')

    return data_module_instance, run_params