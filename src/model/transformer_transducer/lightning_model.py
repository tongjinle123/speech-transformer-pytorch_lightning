from src.model.transformer_transducer.transformer_transducer import TransformerTransducer
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from src.utils import RAdam
from collections import OrderedDict
import torch as t
from src.bak.data_loader import build_single_dataloader


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super(LightningModel, self).__init__()
        self.hparams = hparams
        self.__build_model()

    def __build_model(self):
        self.transducer = TransformerTransducer(
            num_time_mask=self.hparams.num_time_mask,
            num_freq_mask=self.hparams.num_freq_mask,
            freq_mask_length=self.hparams.freq_mask_length,
            time_mask_length=self.hparams.time_mask_length,
            feature_dim=self.hparams.feature_dim,
            model_size=self.hparams.model_size,
            feed_forward_size=self.hparams.feed_forward_size,
            hidden_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            num_head=self.hparams.num_head,
            num_layer=self.hparams.num_layer,
            vocab_path=self.hparams.vocab_path,
            max_feature_length=self.hparams.max_feature_length,
            max_token_length=self.hparams.max_token_length
        )

    def forward(self, feature, feature_length, target, target_length, cal_rnnt_loss=True):
        joint_out, rnnt_loss = self.transducer.forward(feature, feature_length, target, target_length, cal_rnnt_loss)
        return joint_out, rnnt_loss

    def training_step(self, batch, batch_nb):
        feature, feature_length, target, target_length = batch[0], batch[1], batch[2], batch[3]
        joint_out, rnnt_loss = self.forward(feature, feature_length, target, target_length, cal_rnnt_loss=True)
        tqdm_dict = {'loss': rnnt_loss}
        output = OrderedDict({
            'loss': rnnt_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_nb):
        feature, feature_length, target, target_length = batch[0], batch[1], batch[2], batch[3]
        joint_out, rnnt_loss = self.forward(feature, feature_length, target, target_length, cal_rnnt_loss=True)
        tqdm_dict = {'loss': rnnt_loss}
        output = OrderedDict({
            'loss': rnnt_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    # def training_end(self, outputs):
    #     train_loss = t.stack([i['rnnt_loss'] for i in outputs]).mean()
    #     return {'train_loss': train_loss, 'log': {'val_loss': train_loss}}

    def validation_end(self, outputs):
        val_loss = t.stack([i['loss'] for i in outputs]).mean()
        return {'val_loss': val_loss, 'log': {'val_loss': val_loss}}

    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        dataloader = build_single_dataloader(
            data_path='data/tfrecords/data_aishell_train_117346.tfrecord',
            index_path='data/tfrecord_index/data_aishell_train_117346.index',
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.train_loader_num_workers
        )
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        dataloader = build_single_dataloader(
            data_path='data/tfrecords/data_aishell_test_small_589.tfrecord',
            index_path='data/tfrecord_index/data_aishell_test_small_589.index',
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.val_loader_num_workers
        )
        return dataloader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser])
        parser.add_argument('--num_freq_mask', default=1, type=int)
        parser.add_argument('--num_time_mask', default=1, type=int)
        parser.add_argument('--freq_mask_length', default=10, type=int)
        parser.add_argument('--time_mask_length', default=20, type=int)
        parser.add_argument('--feature_dim', default=320, type=int)
        parser.add_argument('--model_size', default=512, type=int)
        parser.add_argument('--feed_forward_size', default=512, type=int)
        parser.add_argument('--hidden_size', default=64, type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--num_head', default=4, type=int)
        parser.add_argument('--num_layer', default=4, type=int)
        parser.add_argument('--vocab_path', default='testing_vocab.model', type=str)
        parser.add_argument('--max_feature_length', default=1024, type=int)
        parser.add_argument('--max_token_length', default=50, type=int)

        parser.add_argument('--lr', default=3e-4, type=float)
        parser.add_argument('--enable_spec_augment', default=False, type=bool)


        parser.add_argument('--train_batch_size', default=16, type=int)
        parser.add_argument('--train_loader_num_workers', default=4, type=int)
        parser.add_argument('--val_batch_size', default=16, type=int)
        parser.add_argument('--val_loader_num_workers', default=4, type=int)

        return parser

