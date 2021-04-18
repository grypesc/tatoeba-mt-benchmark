# tatoeba-machine-translation-benchmark
Tatoeba machine translation benchmark and implementations of different seq2seq algorithms. This benchmark is focused on delivering
high flexibility for research purposes. Dataset is tokenized with Spacy tokenizers during dataset generation phase.
DataPipeline objects deliver vocabs containing FastText embeddings and Torch data loaders. Currently implemented:
* enc_dec_attn.py - Bidirectional encoder-decoder with attention.
* rql.py - Recurrent Q-learning algorithm with agents translating on-line.

Setup:
```bash
git clone https://github.com/grypesc/atmt
cd atmt && mkdir data
```
Or
```bash
git clone https://ben.ii.pw.edu.pl/gitlab/recurrent-graph-networks/mt
cd mt && mkdir data
```
Then:
```bash
python3 -m venv venv
source venv/bin/activate
pip install pip -U
pip install -r requirements.txt
python -m spacy download en_core_web_md
python -m spacy download es_core_news_md
python -m spacy download fr_core_news_md
python -m spacy download de_core_news_md
python -m spacy download zh_core_web_md
python -m spacy download ru_core_news_md
```
Create desired datasets by running generate_dataset.py, currently there are 6 languages 
supported: (en, es, fr, de, zh, ru), so that's 30 combinations:

```python3
python generate_datasets.py --src en --trg es
```

Now you are good to run any algorithm you want :).
