import torch
from torch import nn

from models.transformer.layers import *


class Transformer(nn.Module):
    def __init__(self, parameters, embed_dropout):
        super().__init__()
        self.src_vocab_size = parameters["src_vocab_size"]
        self.trg_vocab_size = parameters["trg_vocab_size"]
        self.d_model = parameters["d_model"]
        self.device = parameters["device"]
        self.d_ff = parameters["d_ff"]
        self.num_heads = parameters["num_heads"]
        self.num_layers = parameters["num_layers"]
        self.d_k = parameters["d_k"]
        self.drop_out_rate = parameters["drop_out_rate"]

        if parameters["src_embedding"] is not None and parameters["trg_embedding"] is not None:
            self.src_embedding = nn.Embedding(self.src_vocab_size, self.d_model).from_pretrained(
                parameters["src_embedding"], freeze=True
            )
            self.trg_embedding = nn.Embedding(self.trg_vocab_size, self.d_model).from_pretrained(
                parameters["trg_embedding"], freeze=True
            )
        else:
            self.src_embedding = nn.Embedding(self.src_vocab_size, self.d_model)
            self.trg_embedding = nn.Embedding(self.trg_vocab_size, self.d_model)
        
        self.drop_out_enc_in = nn.Dropout(embed_dropout)
        self.drop_out_dec_in = nn.Dropout(embed_dropout)

        self.positional_encoder = PositionalEncoder(parameters)
        self.encoder = Encoder(self.d_model, self.d_ff, self.num_heads, self.num_layers, self.d_k, self.drop_out_rate)
        self.decoder = Decoder(self.d_model, self.d_ff, self.num_heads, self.num_layers, self.d_k, self.drop_out_rate)
        self.output_linear = nn.Linear(self.d_model, self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, trg_input, e_mask=None, d_mask=None, training=False, limit=100):
        src_input = self.src_embedding(src_input)  # (B, L) => (B, L, d_model)
        src_input = self.drop_out_enc_in(src_input)
        trg_input = self.trg_embedding(trg_input)  # (B, L) => (B, L, d_model)
        trg_input = self.drop_out_dec_in(trg_input)
        src_input = self.positional_encoder(src_input)  # (B, L, d_model) => (B, L, d_model)
        trg_input = self.positional_encoder(trg_input)  # (B, L, d_model) => (B, L, d_model)

        e_output = self.encoder(src_input, e_mask)  # (B, L, d_model)

        if training:
            d_output = self.decoder(trg_input, e_output, e_mask, d_mask)  # (B, L, d_model)
            return self.softmax(self.output_linear(d_output))  # (B, L, d_model) => # (B, L, trg_vocab_size)
        else:
            return self._predict(trg_input, e_output, e_mask, limit)

    def _predict(self, trg_input, e_output, e_mask, len_lim):
        predictions = torch.tensor([]).to(self.device)
        for step in range(len_lim):
            out = self.decoder(trg_input, e_output, e_mask, None)
            out = self.softmax(self.output_linear(out))[:, -1:, :]
            predictions = torch.cat((predictions, out), dim=1)
            out = torch.argmax(out, dim=-1)
            out = self.trg_embedding(out)
            out = self.drop_out_dec_in(out)
            out = self.positional_encoder(out, step + 1)
            trg_input = torch.cat((trg_input, out), dim=1)

        return predictions


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, d_k, drop_out_rate):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, num_heads, d_k, drop_out_rate) for i in range(num_layers)]
        )
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, d_k, drop_out_rate):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, num_heads, d_k, drop_out_rate) for i in range(num_layers)]
        )
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)
