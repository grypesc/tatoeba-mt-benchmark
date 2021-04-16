import pickle
import torch

from collections import Counter
from torchtext.vocab import Vocab, FastText
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class DataPipeline:
    """Provides vocabularies and data loaders for train, valid, test. Uses pretrained FastText embeddings. Can
    either use bos token which is always the first one in src and trg sequences or use null token, which is
    useful for rql"""

    def __init__(self, batch_size, src_lang, trg_lang, null_replaces_bos=False):

        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.bos_or_null_token = "<null>" if null_replaces_bos else "<bos>"

        prefix = src_lang + "_" + trg_lang
        train_filepath = "data/" + prefix + "_train.pickle"
        val_filepath = "data/" + prefix + "_valid.pickle"
        test_filepath = "data/" + prefix + "_test.pickle"

        self.src_vocab, self.trg_vocab = self.build_vocabs(train_filepath)

        train_data = self.tensor_from_files(train_filepath)
        val_data = self.tensor_from_files(val_filepath)
        test_data = self.tensor_from_files(test_filepath)

        generate_batch = self.generate_null_batch if null_replaces_bos else self.generate_bos_batch
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch, num_workers=0)
        self.valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch, num_workers=0)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch, num_workers=0)

    def build_vocabs(self, filepath):
        src_counter, trg_counter = Counter(), Counter()
        with open(filepath, 'rb') as f:
            pairs = pickle.load(f)
        for pair in pairs:
            src_counter.update(pair[0])
            trg_counter.update(pair[1])
        src_vocab = self.vocab_from_counter(src_counter, self.src_lang)
        trg_vocab = self.vocab_from_counter(trg_counter, self.trg_lang)
        return src_vocab, trg_vocab

    def vocab_from_counter(self, counter, lang):
        vocab = Vocab(counter, specials=['<unk>', self.bos_or_null_token, '<pad>',  '<eos>'], vectors=FastText(language=lang, max_vectors=1000_000))
        zero_vec = torch.zeros(vocab.vectors.size()[0])
        zero_vec = torch.unsqueeze(zero_vec, dim=1)
        vocab.vectors = torch.cat((zero_vec, zero_vec, zero_vec, vocab.vectors), dim=1)
        vocab.vectors[vocab[self.bos_or_null_token]][0] = 1
        vocab.vectors[vocab['<pad>']][1] = 1
        vocab.vectors[vocab['<eos>']][2] = 1
        return vocab

    def tensor_from_files(self, filepath):
        with open(filepath, 'rb') as f:
            pairs = pickle.load(f)
        data = []
        for pair in pairs:
            en_tensor_ = torch.tensor([self.src_vocab[token] for token in pair[0]], dtype=torch.long)
            spa_tensor_ = torch.tensor([self.trg_vocab[token] for token in pair[1]], dtype=torch.long)
            data.append((en_tensor_, spa_tensor_))
        return data

    def generate_null_batch(self, data_batch):
        src_batch, trg_batch, = [], []
        for (en_item, spa__item) in data_batch:
            src_batch.append(torch.cat([en_item, torch.tensor([self.src_vocab['<eos>']])], dim=0))
            trg_batch.append(torch.cat([spa__item, torch.tensor([self.trg_vocab['<eos>']])], dim=0))
        src_batch = pad_sequence(src_batch, padding_value=self.src_vocab['<pad>'])
        trg_batch = pad_sequence(trg_batch, padding_value=self.trg_vocab['<pad>'])
        return src_batch, trg_batch

    def generate_bos_batch(self, data_batch):
        src_batch, trg_batch, = [], []
        for (en_item, spa__item) in data_batch:
            src_batch.append(torch.cat([torch.tensor([self.src_vocab['<bos>']]), en_item, torch.tensor([self.src_vocab['<eos>']])], dim=0))
            trg_batch.append(torch.cat([torch.tensor([self.trg_vocab['<bos>']]), spa__item, torch.tensor([self.trg_vocab['<eos>']])], dim=0))
        src_batch = pad_sequence(src_batch, padding_value=self.src_vocab['<pad>'])
        trg_batch = pad_sequence(trg_batch, padding_value=self.trg_vocab['<pad>'])
        return src_batch, trg_batch
