# tatoeba-machine-translation-benchmark
Tatoeba machine translation benchmark and implementations of different seq2seq algorithms. This benchmark is focused on delivering
high flexibility for research purposes. Dataset is tokenized with Spacy tokenizers during dataset generation phase.
DataPipeline objects deliver vocabs containing FastText embeddings and Torch data loaders. Currently implemented:
* enc_dec_attn.py - Bidirectional encoder-decoder with attention.
* rlst.py - Recurrent Q-learning algorithm with agents translating on-line.

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
Download raw data from 21.04.2021 snapshot: 
```bash
wget 'https://www.dropbox.com/s/kuiseuq2rn5540s/21-04-2021-tatoeba.zip?dl=1' -O data/21-04-2021-tatoeba.zip
unzip data/21-04-2021-tatoeba.zip -d data/
```
Alternatively, you can specify ```--update``` argument for generate_datasets.py to download the newest data from Tatoeba.

Create desired datasets by running generate_dataset.py, currently there are 6 languages 
supported: (en, es, fr, de, zh, ru), so that's 30 combinations:

```python3
python generate_datasets.py --src en --trg es
```

Now you can run algorithms from command line:
```python3
python enc_dec_attn.py --enc_hid_dim 128 --dec_hid_dim 128 --attn_dim 32
```

Type for more info:
```python3
python enc_dec_attn.py --help
```