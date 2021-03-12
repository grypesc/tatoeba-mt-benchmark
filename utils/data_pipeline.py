import io
import torch

from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, FastText
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class DataPipeline:  # TODO: base class maybe?
    """Provides vocabularies and data loaders for encoder decoder models from english to spanish sentences. Uses
    pretrained FastText embeddings"""

    def __init__(self, batch_size=64):
        train_filepaths = ["data/train_eng.txt", "data/train_spa.txt"]
        val_filepaths = ["data/validation_eng.txt", "data/validation_spa.txt"]
        test_filepaths = ["data/test_eng.txt", "data/test_spa.txt"]
        self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_md')
        self.spa_tokenizer = get_tokenizer('spacy', language='es_core_news_md')
        self.en_vocab = self.build_input_vocab(train_filepaths[0], self.en_tokenizer)
        self.spa_vocab = self.build_output_vocab(train_filepaths[1], self.spa_tokenizer)

        train_data = self.tensor_from_files(train_filepaths)
        val_data = self.tensor_from_files(val_filepaths)
        test_data = self.tensor_from_files(test_filepaths)

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=self.generate_batch, num_workers=0)
        self.valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=self.generate_batch, num_workers=0)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=self.generate_batch, num_workers=0)

    def build_input_vocab(self, filepath, tokenizer):
        counter = Counter()
        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_))
        vocab = Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'], vectors=FastText(language='en', max_vectors=1000_000))
        zero_vec = torch.zeros(vocab.vectors.size()[0])
        zero_vec = torch.unsqueeze(zero_vec, dim=1)
        vocab.vectors = torch.cat((zero_vec, zero_vec, zero_vec, vocab.vectors), dim=1)
        vocab.vectors[vocab["<pad>"]][0] = 1
        vocab.vectors[vocab["<bos>"]][1] = 1
        vocab.vectors[vocab["<eos>"]][2] = 1
        return vocab

    def build_output_vocab(self, filepath, tokenizer):
        counter = Counter()
        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_))
        return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    def tensor_from_files(self, filepaths):
        raw_en_iter = iter(io.open(filepaths[0], encoding="utf8"))
        raw_spa_iter = iter(io.open(filepaths[1], encoding="utf8"))
        data = []
        for (raw_en, raw_spa) in zip(raw_en_iter, raw_spa_iter):
            en_tensor_ = torch.tensor([self.en_vocab[token] for token in self.en_tokenizer(raw_en)[:-1]],
                                      dtype=torch.long)  # [:-1] ignore /n at the line end
            spa_tensor_ = torch.tensor([self.spa_vocab[token] for token in self.spa_tokenizer(raw_spa)[:-1]],
                                       dtype=torch.long)  # [:-1] ignore /n at the line end
            data.append((en_tensor_, spa_tensor_))
        return data

    def generate_batch(self, data_batch):
        en_batch, spa_batch, = [], []
        for (en_item, spa__item) in data_batch:
            en_batch.append(torch.cat([torch.tensor([self.en_vocab['<bos>']]), en_item, torch.tensor([self.en_vocab['<eos>']])], dim=0))
            spa_batch.append(torch.cat([torch.tensor([self.spa_vocab['<bos>']]), spa__item, torch.tensor([self.spa_vocab['<eos>']])], dim=0))
        en_batch = pad_sequence(en_batch, padding_value=self.en_vocab['<pad>'])
        spa_batch = pad_sequence(spa_batch, padding_value=self.spa_vocab['<pad>'])
        return en_batch, spa_batch
