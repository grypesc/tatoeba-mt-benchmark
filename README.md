# tatoeba-machine-translation-benchmark
Tatoeba machine translation benchmark supporting 30 pairs of languages. This benchmark is focused on delivering
high flexibility for research purposes. Datasets are tokenized with Spacy tokenizers during dataset generation phase.
DataPipeline objects deliver vocabs and Torch data loaders. FastText pretrained embeddings are also available. Main 
evaluation method is perplexity and BLEU score based on torchtext implementation. Currently implemented models:
* enc_dec_attn.py - Bidirectional encoder-decoder with attention.
* rlst.py - Recurrent Q-learning algorithm with agents translating on-line.
* transformer.py - Standard transformer architecture

Setup:
```bash
git clone https://github.com/grypesc/tatoeba-mt-benchmark
cd tatoeba-mt-benchmark && mkdir data && mkdir checkpoints
```
Or
```bash
git clone https://ben.ii.pw.edu.pl/gitlab/recurrent-graph-networks/mt
cd mt && mkdir data && mkdir checkpoints
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

To repeat paper results for en-es language pair, firstly train models using following commands:
```python3
python enc_dec_attn.py --src en --trg es --embed-dropout 0.2  --decoder-dropout 0.5 --checkpoint-dir  checkpoints/enc-en-es \
--enc-hid-dim 256 --dec-hid-dim 256 --weight-decay 1e-5 --teacher-forcing 1.0 --epochs 50 --attn-dim 64 --batch-size 128 --lr 3e-4 
```
```python3 
python rlst.py --src en --trg es --checkpoint-dir checkpoints/rlst-en-es --rnn-hid-dim 512 --teacher-forcing 1.0 \
--epochs 50 --lr 3e-4 --weight-decay 1e-5 --rnn-num-layers 4 --rnn-dropout 0.5 --embed-dropout 0.2 --N 50000 \
--eta-min 0.02 --eta-max 0.2 --rho 0.99
```
```python3 
python transformer.py --src en --trg es --lr 3e-4 --checkpoint-dir checkpoints/trans-en-es --num-heads 8 --epochs 50 \
--dropout 0.25 --embed-dropout 0.2 --weight-decay 1e-4 --d-ffn 512 --num-layers 6 --batch-size 128 --d-model 256
```
Now test models on test and long test sets:
```python3
python enc_dec_attn.py --src en --trg es --checkpoint-dir checkpoints/enc-en-es --enc-hid-dim 256 --dec-hid-dim 256 \
--attn-dim 64 --batch-size 32 --test --test-seq-max-len 400 --load-model-name enc_dec_attn_best.pth
```
```python3 
python rlst.py --src en --trg es --checkpoint-dir checkpoints/rlst-en-es --rnn-hid-dim 512 --rnn-num-layers 4 --test \
--batch-size 32 --testing-episode-max-time 512 --load-model-name rlst_best.pth
```
```python3 
python transformer.py --src en --trg es --lr 3e-4 --checkpoint-dir checkpoints/trans-en-es --num-heads 8 --epochs 50 \
--d-ffn 512 --num-layers 6 --test --test-seq-max-len 400 --batch-size 32 --d-model 256 --load-model-name transformer_best.pth
```
To repeat results for other language pairs change value of ```--src``` and ```--trg``` parameters. 

Models are saved and evaluated on validation set after every epoch. The benchamrk saves always the last and the best model
according to its BLEU score on validation set. For more information and hyperparameters:
```python3
python rlst.py --help
```

Feel free to add new models, suggestions, issues and make PRs :smiling_face_with_three_hearts:.