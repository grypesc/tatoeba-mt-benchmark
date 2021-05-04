# tatoeba-machine-translation-benchmark
Tatoeba machine translation benchmark supporting 30 pairs of languages. This benchmark is focused on delivering
high flexibility for research purposes. Dataset is tokenized with Spacy tokenizers during dataset generation phase.
DataPipeline objects deliver vocabs containing FastText embeddings and Torch data loaders. Currently implemented models:
* enc_dec_attn.py - Bidirectional encoder-decoder with attention.
* rlst.py - Recurrent Q-learning algorithm with agents translating on-line.

Setup:
```bash
git clone https://github.com/grypesc/tatoeba-mt-benchmark
cd atmt && mkdir data && mkdir checkpoints
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

Now you can train models using command line:
```python3
CUDA_VISIBLE_DEVICES=0 python enc_dec_attn.py --src en --trg es --epochs 30 --enc_hid_dim 128 --dec_hid_dim 128 --attn_dim 32
```
```python3 
CUDA_VISIBLE_DEVICES=0 python rlst.py --src en --trg es --checkpoint_dir checkpoints \
--testing_episode_max_time 64 --batch_size 128 --lr 1e-3 --clip 1.0 --weight_decay 1e-5 \ 
--rnn_hid_dim 768 --rnn_num_layers 2 --rnn_dropout 0.2 --epsilon 0.15 --N 50000 --discount 0.90
```
Models are saved and evaluated on validation set after every epoch.
To test models on test and long test sets use:
```python3
CUDA_VISIBLE_DEVICES=0 python enc_dec_attn.py --src en --trg es --test --batch_size 32 --test_seq_max_len 256 \
--load_model_name enc_dec_attn_best.pth --enc_hid_dim 128 --dec_hid_dim 128 --attn_dim 32
```

Type for more info and hyperparameters:
```python3
python rlst.py --help
```

Feel free to add new models, suggestions and make PRs :smiling_face_with_three_hearts:.