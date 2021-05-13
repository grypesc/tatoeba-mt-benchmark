import pickle
import torch

from collections import Counter
from torchtext.vocab import Vocab, FastText
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class DataPipeline:
    """Provides vocabularies and data loaders for train, valid, test and will download and use pretrained FastText
     embeddings, if specified. Can either use bos token which is always the first one in src and trg sequences or use
     null token, which is useful for RLST"""

    def __init__(self, batch_size, src_lang, trg_lang, null_replaces_bos=False, token_min_freq=1, use_pretrained_embeds=False):

        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.bos_or_null_token = "<null>" if null_replaces_bos else "<bos>"
        self.token_min_freq = token_min_freq

        prefix = src_lang + "_" + trg_lang
        train_filepath = "data/" + prefix + "_train.pickle"
        val_filepath = "data/" + prefix + "_valid.pickle"
        test_filepath = "data/" + prefix + "_test.pickle"
        long_test_filepath = "data/" + prefix + "_test_long.pickle"

        self.src_vocab, self.trg_vocab = self.build_vocabs(train_filepath, use_pretrained_embeds)

        train_data = self.tensor_from_files(train_filepath)
        val_data = self.tensor_from_files(val_filepath)
        test_data = self.tensor_from_files(test_filepath)
        long_test_data = self.tensor_from_files(long_test_filepath)

        generate_batch = self.generate_null_batch if null_replaces_bos else self.generate_bos_batch
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch, num_workers=0)
        self.valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=generate_batch, num_workers=0)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=generate_batch, num_workers=0)
        self.long_test_loader = DataLoader(long_test_data, batch_size=batch_size, shuffle=False, collate_fn=generate_batch, num_workers=0)

        print("Loaded {} {} training sentences".format(len(train_data), prefix))
        print("Loaded {} validating sentences".format(len(val_data)))
        print("Loaded {} test sentences".format(len(test_data)))
        print("Loaded {} long-test sentences".format(len(long_test_data)))
        print("Source vocabulary has {} unique tokens".format(len(self.src_vocab)))
        print("Target vocabulary has {} unique tokens\n".format(len(self.trg_vocab)))

    def build_vocabs(self, filepath, use_pretrained_embeds):
        src_counter, trg_counter = Counter(), Counter()
        with open(filepath, 'rb') as f:
            pairs = pickle.load(f)
        for pair in pairs:
            src_counter.update(pair[0])
            trg_counter.update(pair[1])
        src_vocab = self.vocab_from_counter(src_counter, self.src_lang, use_pretrained_embeds)
        trg_vocab = self.vocab_from_counter(trg_counter, self.trg_lang, use_pretrained_embeds)
        return src_vocab, trg_vocab

    def vocab_from_counter(self, counter, lang, use_pretrained_embeds):
        vectors = FastText(language=lang, max_vectors=1000_000) if use_pretrained_embeds else None
        vocab = Vocab(counter, specials=['<unk>', self.bos_or_null_token, '<pad>',  '<eos>'], min_freq=self.token_min_freq, vectors=vectors)
        if use_pretrained_embeds:
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
            src_tensor = torch.tensor([self.src_vocab[token] for token in pair[0]], dtype=torch.long)
            trg_tensor = torch.tensor([self.trg_vocab[token] for token in pair[1]], dtype=torch.long)
            data.append((src_tensor, trg_tensor))
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
