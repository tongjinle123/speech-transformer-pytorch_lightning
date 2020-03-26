from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from src.model.transformer_rezero.lightning_model import LightningModel
import torch.backends.cudnn as cudnn
import random
import torch as t
import os
import argparse
from warnings import filterwarnings
from pytorch_lightning.logging.test_tube_logger import TestTubeLogger
filterwarnings('ignore')


def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--nb_gpu_nodes', type=int, default=1)
    parent_parser.add_argument('-seed', default=1, type=int)
    parent_parser.add_argument('-epochs', default=200, type=int)
    parser = LightningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    model = LightningModel(hparams)
    if hparams.seed is not None:
        random.seed(hparams.seed)
        t.manual_seed(hparams.seed)
        cudnn.deterministic = True
    exp_root = 'exp'
    log_folder = 'lightning_logs'
    log_root = os.path.join(exp_root, log_folder)
    logger = TestTubeLogger(exp_root, name=log_folder, version=2000)
    checkpoint = ModelCheckpoint(filepath='exp/lightning_logs/version_2000/checkpoints/',
                                 monitor='val_mer', verbose=1, save_top_k=-1)
    trainer = Trainer(
        logger=logger,
        early_stop_callback=False,
        accumulate_grad_batches=3,
        checkpoint_callback=checkpoint,
        # checkpoint_callback=checkpoint,
        # fast_dev_run=True,
        # overfit_pct=0.03,
        # profiler=True,
        default_save_path='exp/',
        val_check_interval=0.3,
        log_save_interval=50000,
        row_log_interval=50000,
        gpus=1,
        val_percent_check=1,
        # distributed_backend='dp',
        nb_gpu_nodes=hparams.nb_gpu_nodes,
        max_nb_epochs=hparams.epochs,
        gradient_clip_val=5.0,
        min_nb_epochs=3000,
        use_amp=True,
        precision=16,
        nb_sanity_val_steps=0,
        progress_bar_refresh_rate=1,
        resume_from_checkpoint='exp/lightning_logs/version_2000/checkpoints/epoch=100_v2.ckpt'
    )
    # if hparams.evaluate:
    #     trainer.run_evaluation()
    # else:
    trainer.fit(model)

if __name__ == '__main__':
    main(get_args())