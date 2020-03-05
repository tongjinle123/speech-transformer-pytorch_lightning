import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from collections import OrderedDict
import torch as t
from src.model.transformer_lm.lm_dataloader import build_raw_data_loader_lm
from src.model.transformer_lm.transformer_lm import TransformerLM
from src.utils.radam import AdamW, RAdam
from src.utils.lookahead import Lookahead



class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super(LightningModel, self).__init__()
        self.hparams = hparams
        self.__build_model()
        self.lr = 0

    def __build_model(self):
        self.transformer_lm = TransformerLM(
            embedding_size=self.hparams.model_size,
            feed_forward_size=self.hparams.feed_forward_size,
            hidden_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            num_head=self.hparams.num_head,
            num_layer=self.hparams.num_layer,
            vocab_path=self.hparams.vocab_path,
            max_length=self.hparams.max_length,
            use_low_rank=self.hparams.use_low_rank
        )

    def forward(self, target, target_length, cal_ce_loss=True):
        logit, output_token, token_length, ce_loss = self.transformer_lm.forward(target, target_length, cal_ce_loss)
        return logit, output_token, token_length, ce_loss

    def training_step(self, batch, batch_nb):
        target, target_length = batch[0], batch[1]
        logit, output_token, token_length, ce_loss = self.forward(target, target_length, True)
        tqdm_dict = {'loss': ce_loss, 'lr': self.lr}
        output = OrderedDict({
            'loss': ce_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_nb):
        target, target_length = batch[0], batch[1]
        logit, output_token, token_length, ce_loss = self.forward(target, target_length, True)
        tqdm_dict = {'loss': ce_loss, 'lr': self.lr}
        output = OrderedDict({
            'loss': ce_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_end(self, outputs):
        val_loss = t.stack([i['loss'] for i in outputs]).mean()
        print('val_loss', val_loss.item())
        return {'val_loss': val_loss, 'log': {'val_loss': val_loss}}

    @pl.data_loader
    def train_dataloader(self):

        dataloader = build_raw_data_loader_lm(
            [
                'data/filterd_manifest/ce_200.csv',
                # 'data/filterd_manifest/c_500_train.csv',
                # 'data/filterd_manifest/aidatatang_200zh_train.csv',
                # 'data/filterd_manifest/data_aishell_train.csv',
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
        dataloader = build_raw_data_loader_lm(
            [
                # 'data/manifest/ce_20_dev.csv',
                'data/filterd_manifest/c_500_test.csv',
                # 'data/manifest/ce_20_dev_small.csv',
                # 'aishell2_testing/manifest1.csv',
                # 'data/filterd_manifest/data_aishell_test.csv'
            ],
            vocab_path=self.hparams.vocab_path,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.train_loader_num_workers,
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
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.997))
        optimizer = Lookahead(optimizer)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser])
        parser.add_argument('--num_freq_mask', default=2, type=int)
        parser.add_argument('--num_time_mask', default=2, type=int)
        parser.add_argument('--freq_mask_length', default=30, type=int)
        parser.add_argument('--time_mask_length', default=20, type=int)
        parser.add_argument('--feature_dim', default=400, type=int)
        parser.add_argument('--model_size', default=512, type=int)
        parser.add_argument('--feed_forward_size', default=2048, type=int)
        parser.add_argument('--hidden_size', default=64, type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--num_head', default=8, type=int)
        parser.add_argument('--num_layer', default=6, type=int)
        parser.add_argument('--vocab_path', default='testing_vocab_2.model', type=str)
        parser.add_argument('--max_length', default=100, type=int)
        parser.add_argument('--share_weight', default=True, type=bool)
        parser.add_argument('--smoothing', default=0.1, type=float)
        parser.add_argument('--use_low_rank', default=True, type=bool)
        parser.add_argument('--lr', default=3e-4, type=float)
        parser.add_argument('--warm_up_step', default=4000, type=int)
        parser.add_argument('--factor', default=1, type=int)
        parser.add_argument('--enable_spec_augment', default=True, type=bool)

        parser.add_argument('--train_batch_size', default=256, type=int)
        parser.add_argument('--train_loader_num_workers', default=1, type=int)
        parser.add_argument('--val_batch_size', default=256, type=int)
        parser.add_argument('--val_loader_num_workers', default=1, type=int)

        return parser
