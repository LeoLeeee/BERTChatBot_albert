# -*- coding:utf-8 -*-

import BERTModel
import data_load_for_Transformer as dlft
import Utility
import os
import torch

# print("Initiating the hyperparameters...")
# num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 16
# lr, num_epochs, device = 0.01, 20000, Utility.try_gpu()
# ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
# key_size, query_size, value_size = 32, 32, 32
# norm_shape = [32]

print("Initiating the hyperparameters...")
num_hiddens, num_layers, dropout, batch_size, num_steps = 128, 2, 0.1, 384, 16
lr, num_epochs, device = 0.001, 100, Utility.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 128, 128, 8
key_size, query_size, value_size = 128, 128, 128
norm_shape = [128]
using_bias = True

print("Building the vocabulary...")
train_iter, vocab = dlft.load_data_xhj_for_Transformer(batch_size, num_steps, load=True)

print("Rebuilding the Model...")
encoder = BERTModel.TransformerEncoder(
    len(vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout, using_bias)
decoder = BERTModel.TransformerDecoder(
    len(vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = BERTModel.EncoderDecoder(encoder, decoder)
try:
    checkpoint_prefix = os.path.join("model_data/model_transformer.pt")
    checkpoint = torch.load(checkpoint_prefix)
    net.load_state_dict(checkpoint['model_state_dict'])
except Exception as e:
    print("Can not load the model with error:", e)

print("Ready to working...")
def predict(src_sentence):
    return BERTModel.predict_seq2seq(net, src_sentence, vocab, vocab, num_steps,
                    device, save_attention_weights=False)


if __name__ == '__main__':
    BERTModel.train_seq2seq(net, train_iter, lr, num_epochs, batch_size, vocab, device)