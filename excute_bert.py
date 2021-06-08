# coding = utf-8

import torch
from torch import nn
import os
import data_load_for_Transformer as dlft
import BERT_pretraining_single
import BERTModel
import Utility

print("Initiating the hyperparameters...")
num_hiddens, num_layers, dropout, batch_size, num_steps = 128, 2, 0.1, 128, 16
lr, num_epochs, device = 0.001, 100, Utility.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 128, 128, 4
key_size, query_size, value_size = 128, 128, 128
norm_shape = [128]
using_bias = True

print("Building the vocabulary...")
train_iter, vocab = dlft.load_data_xhj_for_Transformer(batch_size, num_steps, token='char', load=True)


print("Rebuilding the Model...")

encoder = BERTModel.BERTEncoder( bert_name='./outputs/', hid_in_features=num_hiddens, num_outputs=num_hiddens)
decoder = BERTModel.TransformerDecoder(
    len(vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = BERTModel.EncoderDecoder(encoder, decoder)

try:
    checkpoint_prefix = os.path.join("model_data/model_bert.pt")
    checkpoint = torch.load(checkpoint_prefix)
    net.load_state_dict(checkpoint['model_state_dict'])
    print("Load model success")
except Exception as e:
    print("Can not load the model with error:", e)

print("Ready to working...")
def predict(src_sentence):
    return BERTModel.predict_seq2seq(net, src_sentence, vocab, vocab, num_steps,
                    device, save_attention_weights=False)



if __name__ == '__main__':
    # BERTModel.train_seq2seq(net, train_iter, lr, num_epochs, batch_size, vocab, device)
    pass