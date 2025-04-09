from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from src.model.model_classic import BaselineModelPV


def get_model(run_params):
    if run_params.model in ['gcn', 'gat', 'mlp', 'gcn1d', 'gcn1d-big', 'gConvLSTM', 'gConvGRU', 'DCRNN', 'GraphWavenet', 'AGCRNModel', 'miniLSTM', 'miniGRU']:
        model = BaselineModelPV(run_params)
    else:
        raise Exception('Error in select the model!')
    return model


def get_callbacks(run_params):
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='../checkpoints',
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='val_mse',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_mse',
        min_delta=0.00,
        patience=4,
        verbose=False,
        mode='min')

    if run_params.save_ckpts and run_params.early_stop_callback_flag:
        callbacks = [checkpoint_callback, early_stop_callback]
    elif run_params.save_ckpts:
        callbacks = [checkpoint_callback]
    elif run_params.early_stop_callback_flag:
        callbacks = [early_stop_callback]
    else:
        callbacks = []
    return callbacks


