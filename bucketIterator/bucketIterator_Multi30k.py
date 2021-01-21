# -*- coding: utf-8 -*-
import sys
sys.path.append('./')
import os.path
import dill as pickle
import spacy
import torchtext.data
import torchtext.datasets
from torchtext.data import Field, Dataset, BucketIterator
import PreprocessData.Constants as Constants


def gererate_Multi30k_data(config):
    src_lang_model = spacy.load(config.lang_src)
    trg_lang_model = spacy.load(config.lang_trg)

    def tokenize_src(text):
        return [tok.text for tok in src_lang_model.tokenizer(text)]

    def tokenize_trg(text):
        return [tok.text for tok in trg_lang_model.tokenizer(text)]

    SRC = Field(tokenize=tokenize_src, lower=True, pad_token=Constants.PAD_WORD,
                init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD)
    TRG = Field(tokenize=tokenize_trg, lower=True, pad_token=Constants.PAD_WORD,
                init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD)
    MAX_LEN = config.max_word_count
    MIN_FREQ = config.min_word_count

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    train, val, test = torchtext.datasets.Multi30k.splits(
        exts=('.de', '.en'),
        fields=(SRC, TRG),
        root='./data/Multi30k',
        filter_pred=filter_examples_with_length)

    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    print('[Info] Get source language vocabulary size:', len(SRC.vocab))
    TRG.build_vocab(train.trg, min_freq=MIN_FREQ)
    print('[Info] Get target language vocabulary size:', len(TRG.vocab))

    data = {'settings': config,
            'vocab': {'src': SRC, 'trg': TRG},
            'train': train.examples,
            'valid': val.examples,
            'test': test.examples}

    print('[Info] Dumping the processed data to pickle file', config.data_pkl_path)
    pickle.dump(data, open(config.data_pkl_path, 'wb'))
    return


def getBucketIterator_Multi30k(config):
    if not os.path.isfile(config.data_pkl_path):
        gererate_Multi30k_data(config=config)
    batch_size = config.batch_size
    data = pickle.load(open(config.data_pkl_path, 'rb'))

    config.max_token_seq_len = data['settings'].max_len
    config.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    config.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    config.src_vocab_size = len(data['vocab']['src'].vocab)
    config.trg_vocab_size = len(data['vocab']['trg'].vocab)

    # ========= Preparing Model =========#
    if config.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg': data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=config.device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=config.device)

    return train_iterator, val_iterator
