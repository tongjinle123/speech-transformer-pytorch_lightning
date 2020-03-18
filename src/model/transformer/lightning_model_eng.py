import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from collections import OrderedDict
import torch as t
from src.loader.dataloader.audio_loader import build_data_loader
from src.model.transformer.transformer import Transformer
from src.utils.radam import AdamW, RAdam
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
        )

    def forward(self, feature, feature_length, target, target_length, cal_ce_loss=True):
        output, output_token, spec_output, feature_length, ori_token, ori_token_length, ce_loss, switch_loss = self.transformer.forward(
            feature, feature_length, target,  target_length, cal_ce_loss)

        return output, output_token, spec_output, feature_length, ori_token, ori_token_length, ce_loss, switch_loss

    def decode(self, feature, feature_length, decode_type='greedy'):
        assert decode_type in ['greedy', 'beam']
        output = self.transformer.inference(feature, feature_length, decode_type=decode_type)
        return output

    def training_step(self, batch, batch_nb):
        feature, feature_length, target, target_length = batch[0], batch[1], batch[2], batch[3]
        model_output, output_token, spec_output, feature_length, ori_token, ori_token_length, ce_loss, switch_loss = self.forward(
            feature, feature_length, target, target_length, True)
        ctc_loss = self.transformer.cal_ctc_loss(spec_output, feature_length, ori_token, ori_token_length)
        loss = self.hparams.loss_lambda * ce_loss + (1 - self.hparams.loss_lambda) * ctc_loss + switch_loss * 0.1
        tqdm_dict = {'ce': ce_loss, 'ctc': ctc_loss, 'switch': switch_loss, 'lr': self.lr}
        output = OrderedDict({
            'loss': loss,
            'ce': ce_loss,
            'ctc': ctc_loss,
            'switch': switch_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_nb):
        feature, feature_length, target, target_length = batch[0], batch[1], batch[2], batch[3]
        model_output, output_token, spec_output, feature_length, ori_token, ori_token_length, ce_loss, switch_loss = self.forward(
            feature, feature_length, target, target_length, True)
        result_string_list = [' '.join(tokenize(i)) for i in self.transformer.inference(feature, feature_length)]
        target_string_list = [' '.join(tokenize(self.transformer.vocab.id2string(i.tolist()))) for i in output_token]
        print(result_string_list[0])
        print(target_string_list[0])
        mers = [cal_wer(i[0], i[1]) for i in zip(target_string_list, result_string_list)]
        mer = np.mean(mers)
        ctc_loss = self.transformer.cal_ctc_loss(spec_output, feature_length, ori_token, ori_token_length)
        loss = self.hparams.loss_lambda * ce_loss + (1 - self.hparams.loss_lambda) * ctc_loss + switch_loss * 0.1
        tqdm_dict = {'loss': loss, 'ce': ce_loss, 'ctc': ctc_loss, 'switch': switch_loss, 'mer': mer, 'lr': self.lr}
        output = OrderedDict({
            'loss': loss,
            'ce': ce_loss,
            'ctc': ctc_loss,
            'switch': switch_loss,
            'mer': mer,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_end(self, outputs):
        val_loss = t.stack([i['loss'] for i in outputs]).mean()
        ce_loss = t.stack([i['ce'] for i in outputs]).mean()
        ctc_loss = t.stack([i['ctc'] for i in outputs]).mean()
        switch_loss = t.stack([i['switch'] for i in outputs]).mean()
        mer = np.mean([i['mer'] for i in outputs])
        print('val_loss', val_loss.item())
        print('ce', ce_loss.item())
        print('ctc', ctc_loss.item())
        print('switch_loss', switch_loss.item())
        print('mer', mer)
        ret = {'val_loss': val_loss, 'val_ce_loss': ce_loss, 'val_ctc_loss': ctc_loss, 'val_mer': mer,
               'log': {
                   'val_loss': val_loss, 'val_ctc_loss': ctc_loss, 'val_ce_loss': ce_loss, 'val_mer': mer
               }
               }

        return ret

    @pl.data_loader
    def train_dataloader(self):
        print('building train loader !!!!')
        dataloader = build_data_loader(
            [
                # 'data/filterd_manifest/ce_200.csv',
                # 'data/manifest/libri_train.csv',
                # 'data/filterd_manifest/c_500_train.csv',
                # 'data/filterd_manifest/aidatatang_200zh_train.csv',
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
            left_frames=5,
            skip_frames=4,
            min_duration=1,
            max_duration=7,
            given_rate=None
        )
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        print('building val loader!!!!')
        dataloader = build_data_loader(
            [
                # 'data/filterd_manifest/ce_200.csv',
                # 'data/manifest/libri_test.csv',
                # 'data/filterd_manifest/c_500_train.csv',
                # 'data/filterd_manifest/aidatatang_200zh_train.csv',
                'data/filterd_manifest/data_aishell_dev.csv',
                # 'data/filterd_manifest/AISHELL-2.csv',
                # 'data/filterd_manifest/magic_data_train.csv',
                # 'data/manifest/libri_100.csv',
                # 'data/manifest/libri_360.csv',
                # 'data/manifest/libri_500.csv'
            ],
            vocab_path=self.hparams.vocab_path,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.train_loader_num_workers,
            left_frames=5,
            skip_frames=4,
            min_duration=1,
            max_duration=7,
            given_rate=1.0
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
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-4)
        optimizer = Lookahead(optimizer)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser])
        parser.add_argument('--num_freq_mask', default=2, type=int)
        parser.add_argument('--num_time_mask', default=2, type=int)
        parser.add_argument('--freq_mask_length', default=30, type=int)
        parser.add_argument('--time_mask_length', default=20, type=int)
        parser.add_argument('--feature_dim', default=480, type=int)
        parser.add_argument('--model_size', default=512, type=int)
        parser.add_argument('--feed_forward_size', default=2048, type=int)
        parser.add_argument('--hidden_size', default=64, type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--num_head', default=8, type=int)
        parser.add_argument('--num_encoder_layer', default=6, type=int)
        parser.add_argument('--num_decoder_layer', default=6, type=int)
        parser.add_argument('--vocab_path', default='testing_vocab.model', type=str)
        parser.add_argument('--max_feature_length', default=1024, type=int)
        parser.add_argument('--max_token_length', default=70, type=int)
        parser.add_argument('--share_weight', default=False, type=bool)
        parser.add_argument('--loss_lambda', default=0.8, type=float)
        parser.add_argument('--smoothing', default=0.1, type=float)

        parser.add_argument('--lr', default=3e-4, type=float)
        parser.add_argument('--warm_up_step', default=16000, type=int)
        parser.add_argument('--factor', default=1, type=int)
        parser.add_argument('--enable_spec_augment', default=True, type=bool)

        parser.add_argument('--train_batch_size', default=32, type=int)
        parser.add_argument('--train_loader_num_workers', default=32, type=int)
        parser.add_argument('--val_batch_size', default=32, type=int)
        parser.add_argument('--val_loader_num_workers', default=16, type=int)

        return parser
