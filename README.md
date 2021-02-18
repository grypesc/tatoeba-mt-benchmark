# atmt
Approaches to machine translation  
Different and new machine translation algorithms tested on tatoeba english to spanish dataset. Currently implemented:
* seq2seq.py - RNN Encoder and decoder with attention without pretrained embeddings.
* seq2seq_embeddings.py - RNN Encoder and decoder with attention with FastText embeddings.
* rql.py - Recurrent Q-learning algorithm with agents translating on-line.

Setup:
```bash
git clone https://github.com/grypesc/atmt
cd atmt && mkdir data
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Download english-spanish sentences dataset from https://www.manythings.org/anki/ and put spa.txt file inside data directory. 
Create training, validation, test sets by running generate_dataset.py:

```python3
python generate_datasets.py
```

Now you are good to run any algorithm you want :).
