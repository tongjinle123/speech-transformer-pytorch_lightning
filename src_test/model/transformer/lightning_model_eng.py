import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from collections import OrderedDict
import torch as t
from src.data_loader.load_data.build_raw_loader import build_raw_data_loader2
from src_test.model.transformer.transformer import Transformer
from src.utils.radam import AdamW, RAdam
from src_reshaped.loader.dataloader.audio_loader import build_data_loader, build_predumped_loader
from src.utils.lookahead import Lookahead
from src.utils.score import cal_wer
from src.utils.tokenizer import tokenize
import numpy as np


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super(LightningModel, self).__init__()
        self.hparams = hparams
        self.__build_model()
        self.lr = 0

    def __build_model(self):
        self.transformer = Transformer(
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
            num_encoder_layer=self.hparams.num_encoder_layer,
            num_decoder_layer=self.hparams.num_decoder_layer,
            vocab_path=self.hparams.vocab_path,
            max_feature_length=self.hparams.max_feature_length,
            max_token_length=self.hparams.max_token_length,
            enable_spec_augment=self.hparams.enable_spec_augment,
            share_weight=self.hparams.share_weight,
            smoothing=self.hparams.smoothing,
            mtlalpha=self.hparams.loss_lambda
        )
        # state = t.load('exp/lightning_logs/version_5005/checkpoints/epoch=36.ckpt')['state_dict']
        # self.load_state_dict(state)
        # # print(f'model parameters num: {sum(p.numel() for p in self.parameters())}')


    def forward(self, feature, feature_length, target, target_length):
        loss = self.transformer.forward(feature, feature_length, target, target_length)
        return loss

    def training_step(self, batch, batch_nb):
        feature, feature_length, target, target_length = batch[0], batch[1], batch[2], batch[3]
        loss = self.forward(feature, feature_length, target, target_length)
        acc = self.transformer.acc
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
        tqdm_dict = {'train_loss': loss, 'acc': acc, 'switch': self.transformer.loss_switch,'att':self.transformer.loss_att, 'ctc':self.transformer.loss_ctc,'lr': self.lr}
        output = OrderedDict({
            'loss': loss,
            'acc': acc,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_nb):
        feature, feature_length, target, target_length = batch[0], batch[1], batch[2], batch[3]
        loss = self.forward(feature, feature_length, target, target_length)
        acc = self.transformer.acc
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
        tqdm_dict = {'val_loss': loss, 'acc': acc, 'switch': self.transformer.loss_switch, 'att': self.transformer.loss_att, 'ctc':self.transformer.loss_ctc,'lr': self.lr}
        output = OrderedDict({
            'val_loss': loss,
            'acc': acc,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_end(self, outputs):
        val_loss = t.stack([i['val_loss'] for i in outputs]).mean()
        acc = np.mean([i['acc'] for i in outputs])
        print('val_loss', val_loss.item())
        print('acc', acc)
        return {'val_loss': val_loss, 'val_acc': acc, 'log': {'val_loss': val_loss, 'val_acc': acc}}

    def train_dataloader(self):
        manifest_list = [
                            'data/filterd_manifest/ce_200.csv',
                            'data/filterd_manifest/c_500_train.csv',
                            # 'data/filterd_manifest/aidatatang_200zh_train.csv',
                            # 'data/filterd_manifest/data_aishell_train.csv',
                            # 'data/filterd_manifest/AISHELL-2.csv',
                            # 'data/filterd_manifest/magic_data_train.csv',
                            # 'data/manifest/libri_100.csv',
                            # 'data/manifest/libri_360.csv',
                            # 'data/manifest/libri_500.csv'
                        ]
        dataloader = build_data_loader(
            manifest_list=manifest_list, batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.train_loader_num_workers, left_frames=5, skip_frames=4
        )

        return dataloader

    def val_dataloader(self):

        manifest_list = [
            # 'data/filterd_manifest/c_500_test.csv',
            'data/manifest/ce_20_dev.csv'
        ]
                            # 'data/manifest/ce_20_dev.csv',
                            # 'data/filterd_manifest/c_500_test_small.csv',
                            # 'data/manifest/ce_20_dev_small.csv',
                            # 'aishell2_testing/manifest1.csv',
                            # 'data/filterd_manifest/data_aishell_test.csv'

        dataloader = build_data_loader(
            manifest_list=manifest_list, batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.train_loader_num_workers, left_frames=5, skip_frames=4
        )

        return dataloader

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        lr = self.hparams.factor * (
                (self.hparams.model_size ** -0.5) * min(
            (self.global_step + 1) ** -0.5, (self.global_step + 1) * (self.hparams.warm_up_step ** -1.5))
        )
        self.lr = lr
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        # optimizer = Lookahead(optimizer)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser])
        parser.add_argument('--num_freq_mask', default=2, type=int)
        parser.add_argument('--num_time_mask', default=2, type=int)
        parser.add_argument('--freq_mask_length', default=30, type=int)
        parser.add_argument('--time_mask_length', default=20, type=int)
        parser.add_argument('--feature_dim', default=480, type=int)
        parser.add_argument('--model_size', default=256, type=int)
        parser.add_argument('--feed_forward_size', default=2048, type=int)
        parser.add_argument('--hidden_size', default=64, type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--num_head', default=4, type=int)
        parser.add_argument('--num_encoder_layer', default=6, type=int)
        parser.add_argument('--num_decoder_layer', default=6, type=int)
        parser.add_argument('--vocab_path', default='testing_vocab.model', type=str)
        parser.add_argument('--max_feature_length', default=1024, type=int)
        parser.add_argument('--max_token_length', default=100, type=int)
        parser.add_argument('--share_weight', default=True, type=bool)
        parser.add_argument('--loss_lambda', default=0.3, type=float)
        parser.add_argument('--smoothing', default=0.1, type=float)

        parser.add_argument('--lr', default=3e-4, type=float)
        parser.add_argument('--warm_up_step', default=25000, type=int)
        parser.add_argument('--factor', default=1, type=int)
        parser.add_argument('--enable_spec_augment', default=True, type=bool)

        parser.add_argument('--train_batch_size', default=32, type=int)
        parser.add_argument('--train_loader_num_workers', default=32, type=int)
        parser.add_argument('--val_batch_size', default=32, type=int)
        parser.add_argument('--val_loader_num_workers', default=16, type=int)

        return parser
