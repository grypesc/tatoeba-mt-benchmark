# atmt
Approaches to machine translation  
Machine translation algorithms and Tatoeba translation benchmark. Dataset is tokenized
with Spacy tokenizer by DataPipeline object and tokens are transformed into FastText embeddings. Currently implemented:
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
```
Create desired datasets by running generate_dataset.py, currently there are 4 languages 
supported: (en, es, fr, de), so that's 12 combinations:

```python3
python generate_datasets.py --src en --trg es
```

Now you are good to run any algorithm you want :).
