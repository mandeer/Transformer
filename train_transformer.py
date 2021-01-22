import argparse
import math
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.datasets import TranslationDataset
from models import Transformer
from bucketIterator import getBucketIterator_Multi30k
from utils import str2bool


def print_performances(header, loss, accu):
    print('%s, ppl: %8.5f, accuracy: %3.3f' % (header, math.exp(min(loss, 100)), 100 * accu), end=' | ')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class Solver(object):
    def __init__(self, config, model, training_data, validation_data, optimizer):
        self.config     = config
        self.model      = model
        self.training_data   = training_data
        self.validation_data = validation_data
        self.optimizer  = optimizer

    def patch_src(self, src, pad_idx):
        src = src.transpose(0, 1)
        return src

    def patch_trg(self, trg, pad_idx):
        trg = trg.transpose(0, 1)
        trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
        return trg, gold

    def cal_performance(self, pred, gold):
        ''' Apply label smoothing if needed '''

        loss = self.cal_loss(pred, gold)

        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(self.config.trg_pad_idx)
        n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()

        return loss, n_correct, n_word

    def cal_loss(self, pred, gold):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''
        gold = gold.contiguous().view(-1)
        if self.config.smoothing:
            eps = 0.1
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(self.config.trg_pad_idx)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=self.config.trg_pad_idx, reduction='sum')
        return loss

    def test_epoch(self):
        self.model.eval()
        total_loss, n_word_total, n_word_correct = 0, 0, 0

        desc = '  - (Validation) '
        with torch.no_grad():
            for batch in tqdm(self.validation_data, mininterval=2, desc=desc, leave=False):
                # prepare data
                src_seq = self.patch_src(batch.src, self.config.src_pad_idx).to(self.config.device)
                trg_seq, gold = map(lambda x: x.to(self.config.device),
                                    self.patch_trg(batch.trg, self.config.trg_pad_idx))

                # forward
                pred = self.model(src_seq, trg_seq)
                loss, n_correct, n_word = self.cal_performance(pred, gold)

                # note keeping
                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss.item()

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy

    def train_epoch(self):
        self.model.train()
        total_loss, n_word_total, n_word_correct = 0, 0, 0

        desc = '  - (Training)   '
        for batch in tqdm(self.training_data, mininterval=2, desc=desc, leave=False):
            # prepare data
            src_seq = self.patch_src(batch.src, self.config.src_pad_idx).to(self.config.device)
            trg_seq, gold = map(lambda x: x.to(self.config.device),
                                self.patch_trg(batch.trg, self.config.trg_pad_idx))

            # forward
            self.optimizer.zero_grad()
            pred = self.model(src_seq, trg_seq)

            # backward and update parameters
            loss, n_correct, n_word = self.cal_performance(pred, gold)
            loss.backward()
            self.optimizer.step_and_update_lr()

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy

    def train(self):
        log_train_file, log_valid_file = None, None
        if self.config.log:
            log_train_file = self.config.log + '.train.log'
            log_valid_file = self.config.log + '.valid.log'
            print('[Info] Training performance will be written to file: {} and {}'.format(
                log_train_file, log_valid_file))
            with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
                log_tf.write('epoch, loss, ppl, accuracy\n')
                log_vf.write('epoch, loss, ppl, accuracy\n')

        valid_losses = []
        for epoch in range(self.config.n_epochs):
            train_loss, train_accu = self.train_epoch()
            print_performances('Training', train_loss, train_accu)
            valid_loss, valid_accu = self.test_epoch()
            print_performances('Validation', valid_loss, valid_accu)

            checkpoint = {'epoch': epoch, 'settings': self.config, 'model': self.model.state_dict()}
            if config.save_model:
                if config.save_mode == 'all':
                    model_name = config.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
                    torch.save(checkpoint, model_name)
                elif config.save_mode == 'best':
                    model_name = config.save_model + '.chkpt'
                    if valid_loss <= min(valid_losses):
                        torch.save(checkpoint, model_name)
                        print('    - [Info] The checkpoint file has been updated.')
            if log_train_file and log_valid_file:
                with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                    log_tf.write('{epoch}, {loss: 8.5f}, {ppl: 8.5f}, {accu:3.3f}\n'.format(
                        epoch=epoch, loss=train_loss, ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu))
                    log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                        epoch=epoch, loss=valid_loss, ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu))
        return


def main(config):
    config.device = torch.device('cuda' if config.use_cuda else 'cpu')
    training_data, validation_data, config = getBucketIterator_Multi30k(config)

    transformer = Transformer(
        n_src_vocab=config.src_vocab_size,
        n_trg_vocab=config.trg_vocab_size,
        src_pad_idx=config.src_pad_idx,
        trg_pad_idx=config.trg_pad_idx,
        d_model=config.d_model,
        d_inner=config.d_inner,
        n_layers=config.n_layers,
        n_head=config.n_head,
        d_k=config.d_k,
        d_v=config.d_v,
        dropout=config.dropout,
        n_position=200,
        trg_emb_prj_weight_sharing=config.proj_share_weight,
        emb_src_trg_weight_sharing=config.embs_share_weight,
    ).to(config.device)
    print(transformer)

    optimizer = ScheduledOptim(optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
                               2.0, config.d_model, config.n_warmup_steps)

    solver = Solver(config, transformer, training_data, validation_data, optimizer)
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('-lang_src',        type=str, default='de_core_news_lg')
    parser.add_argument('-lang_trg',        type=str, default='en_core_web_lg')
    parser.add_argument('-max_word_count',  type=int, default=100)
    parser.add_argument('-min_word_count',  type=int, default=3)
    parser.add_argument('-data_pkl_path',   type=str, default='./data/m30k_deen_shr.pkl')  # all-in-1 data pickle

    parser.add_argument('-train_path',      default=None)  # bpe encoded data
    parser.add_argument('-val_path',        default=None)  # bpe encoded data
    parser.add_argument('-n_epochs',        type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)
    parser.add_argument('-d_model',     type=int, default=512)
    parser.add_argument('-d_inner', type=int, default=2048)
    parser.add_argument('-d_k',     type=int, default=64)
    parser.add_argument('-d_v',     type=int, default=64)
    parser.add_argument('-n_head',      type=int, default=8)
    parser.add_argument('-n_layers',    type=int, default=6)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-dropout',     type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-log', type=str2bool, default='True')
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    config = parser.parse_args()
    if config.use_cuda and not torch.cuda.is_available():
        config.use_cuda = False
        print("WARNING: You have no CUDA device")

    args = vars(config)
    print('------------ Options -------------')
    for key, value in sorted(args.items()):
        print('%16.16s: %16.16s' % (str(key), str(value)))
    print('-------------- End ----------------')

    main(config)
    print('End!!')
