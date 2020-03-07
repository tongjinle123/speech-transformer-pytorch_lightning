import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from collections import OrderedDict
import torch as t
from src.data_loader.load_data.build_text_loader import build_text_data_loader
from src_test.model.rnn_lm.rnn_lm import RNNLM, ClassifierWithState
from src.utils.radam import AdamW, RAdam
from src.utils.lookahead import Lookahead
import numpy as np
from src.utils.vocab import Vocab



class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super(LightningModel, self).__init__()
        self.hparams = hparams
        self.__build_model()
        self.lr = 0

    def __build_model(self):
        self.vocab = Vocab(self.hparams.vocab_path)
        self.rnn = RNNLM(
            n_vocab=self.vocab.vocab_size, n_layers=self.hparams.n_layers, n_units=self.hparams.n_units,
            n_embed=self.hparams.n_embeded, typ=self.hparams.type
        )
        self.model = ClassifierWithState(predictor=self.rnn)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser])
        parser.add_argument('--n_layers', default=2, type=int)
        parser.add_argument('--n_units', default=256, type=int)
        parser.add_argument('--n_embeded', default=300, type=int)
        parser.add_argument('--type', default='gru', type=str)
        parser.add_argument('--lr', default=3e-4, type=float)
        parser.add_argument('--vocab_path', default='testing_vocab.model', type=str)
        parser.add_argument('--train_batch_size', default=32, type=int)
        parser.add_argument('--train_loader_num_workers', default=16, type=int)
        parser.add_argument('--val_batch_size', default=32, type=int)
        parser.add_argument('--val_loader_num_workers', default=16, type=int)

        return parser

    def forward(self, target, target_length):
        _, loss = self.model.forward(target, target_length)
        return loss

    def training_step(self, batch, batch_nb):
        target, target_length = batch[0], batch[1]
        loss = self.forward(target, target_length)

        tqdm_dict = {'loss': loss, 'lr': self.lr}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_nb):
        target, target_length = batch[0], batch[1]
        loss = self.forward(target, target_length)

        tqdm_dict = {'loss': loss, 'lr': self.lr}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_end(self, outputs):
        val_loss = t.stack([i['loss'] for i in outputs]).mean()
        print('val_loss', val_loss.item())
        return {'val_loss': val_loss, 'log': {'val_loss': val_loss, }}

    @pl.data_loader
    def train_dataloader(self):
        dataloader = build_text_data_loader(
            [
                'data/filterd_manifest/ce_200.csv',
                'data/filterd_manifest/c_500_train.csv',
                'data/filterd_manifest/aidatatang_200zh_train.csv',
                'data/filterd_manifest/data_aishell_train.csv',
                # 'data/filterd_manifest/AISHELL-2.csv',
                # 'data/filterd_manifest/magic_data_train.csv',
                # 'data/manifest/libri_100.csv',
                # 'data/manifest/libri_360.csv',
                # 'data/manifest/libri_500.csv'
            ],
            vocab_path=self.hparams.vocab_path,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.train_loader_num_workers,
        )
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        dataloader = build_text_data_loader(
            [
                # 'data/manifest/ce_20_dev.csv',
                # 'data/filterd_manifest/c_500_test.csv',
                # 'data/manifest/ce_20_dev_small.csv',
                # 'aishell2_testing/manifest1.csv',
                'data/filterd_manifest/data_aishell_test.csv'
            ],
            vocab_path=self.hparams.vocab_path,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.train_loader_num_workers,
        )
        return dataloader

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.997))
        optimizer = Lookahead(optimizer)
        return optimizer
