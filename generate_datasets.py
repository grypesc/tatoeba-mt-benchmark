import argparse
import pickle
import random

from tatoebatools import tatoeba, ParallelCorpus
from torchtext.data.utils import get_tokenizer

random.seed(22)
tatoeba.dir = "data/raw"

lang_name_dict = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "zh": "cmn",
    "ru": "rus"
}

tokenizers_dict = {
    "en": "en_core_web_md",
    "de": "de_core_news_md",
    "fr": "fr_core_news_md",
    "es": "es_core_news_md",
    "zh": "zh_core_web_md",
    "ru": "ru_core_news_md"
}

if __name__ == '__main__':
    """This script generates train, valid, test datasets in pickle file format. Before running download raw data from
    Tatoeba and put it inside data directory, check README.md for details."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
                        help='source language',
                        type=str,
                        default='en')
    parser.add_argument('--trg',
                        help='target language',
                        type=str,
                        default='es')
    parser.add_argument('--max_sequence_length',
                        help='max sequence length not including eos and bos tokens',
                        type=int,
                        default=22)
    parser.add_argument('--max_sentences',
                        help='max number of sentences to be split between train/val/test',
                        type=int,
                        default=10 ** 6)
    parser.add_argument('--update',
                        help='whether to download new data from Tatoeba, by default you should provide data from snapshot',
                        default=False,
                        action="store_true")

    args = parser.parse_args()

    pairs = [[s.text, t.text] for s, t in ParallelCorpus(lang_name_dict[args.src], lang_name_dict[args.trg], update=args.update)]
    if len(pairs) == 0:
        raise RuntimeError("No raw data inside data directory, please check out README.md")
    pairs = [[s.replace('\xa0', '') for s in pair] for pair in pairs]  # remove no breaking space, happens in tatoeba
    pairs = [[s.lower() for s in pair] for pair in pairs]
    random.shuffle(pairs)

    src_tokenizer = get_tokenizer('spacy', language=tokenizers_dict[args.src])
    trg_tokenizer = get_tokenizer('spacy', language=tokenizers_dict[args.trg])

    tokenized_pairs = [[src_tokenizer(s), trg_tokenizer(t)] for s, t in pairs]
    tokenized_pairs_str = [[' '.join(src_word for src_word in tokenized_pair[0]), ' '.join(trg_word for trg_word in tokenized_pair[1])] for tokenized_pair in tokenized_pairs]

    unique_src_sentences_dict = {pair[0]: pair[1] for pair in tokenized_pairs_str}  # remove repeated src sentences
    pairs = [[src_sentence, unique_src_sentences_dict[src_sentence]] for src_sentence in unique_src_sentences_dict.keys()]
    pairs = [[pair[0].split(" "), pair[1].split(" ")] for pair in pairs]
    long_pairs = [[s, t] for s, t in pairs if len(t) > args.max_sequence_length]
    pairs = pairs[:args.max_sentences]
    pairs = [[s, t] for s, t in pairs if len(s) <= args.max_sequence_length and len(t) <= args.max_sequence_length]

    train_set = pairs[0:int(0.6 * len(pairs))]
    validation_set = pairs[int(0.6 * len(pairs)):int(0.8 * len(pairs))]
    test_set = pairs[int(0.8 * len(pairs)):]

    prefix = args.src + "_" + args.trg
    with open("data/" + prefix + "_train.pickle", 'wb') as outfile:
        pickle.dump(train_set, outfile)

    with open("data/" + prefix + "_valid.pickle", 'wb') as outfile:
        pickle.dump(validation_set, outfile)

    with open("data/" + prefix + "_test.pickle", 'wb') as outfile:
        pickle.dump(test_set, outfile)

    with open("data/" + prefix + "_test_long.pickle", 'wb') as outfile:
        pickle.dump(long_pairs, outfile)
