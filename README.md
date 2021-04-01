# atmt
Approaches to machine translation  
Machine translation algorithms tested on tatoeba english to spanish dataset. Dataset is tokenized
with Spacy tokenizer by DataPipeline object and tokens are transformed into FastText embeddings. Currently implemented:
* enc_dec_attn.py - RNN encoder-decoder with attention.
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
wget https://www.manythings.org/anki/spa-eng.zip -O data/spa-eng.zip
unzip data/spa-eng.zip -d data
```
Create training, validation, test sets by running generate_dataset.py:

```python3
python generate_datasets.py
```

Now you are good to run any algorithm you want :).
