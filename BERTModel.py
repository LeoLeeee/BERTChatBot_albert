# coding = utf-8
import torch
from torch import nn
import AttentionModel as am
import numpy as np
import Utility
import math
import os
import data_load_for_Transformer as dlft
import random, time
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerDecoderLayer, TransformerEncoderLayer
from transformers import AutoModel
from transformers import BertTokenizer

def get_attention_mask(X, valid_lens, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_lens[:, None]
    padding_mask = torch.ones_like(X)
    padding_mask[~mask] = value
    return padding_mask

class BERTEncoder(nn.Module):
    def __init__(self, bert_name='./outputs/', hid_in_features=128, num_outputs=128):
        super(BERTEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(bert_name, output_hidden_states=True, return_dict=True)
        self.output = nn.Linear(hid_in_features, num_outputs)

    def forward(self, X, valid_lens=None):
        # tk_X = self.tokenizer(X, is_split_into_words=True)
        token_type_ids = torch.zeros_like(X)
        attention_mask = get_attention_mask(X = X, valid_lens=valid_lens)
        X = self.encoder(input_ids= X, token_type_ids=token_type_ids, attention_mask=attention_mask)
        hidden_states = X.hidden_states[-1]

        # X = self.hidden(X)
        # encoded_X = self.output(X)+X  # imitate the Resnate in case the output is layer is just to be an identity layer.
        return self.output(hidden_states)

def sequence_mask(X, valid_lens, value=True):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_lens[:, None]
    padding_mask = torch.zeros_like(X, dtype=torch.bool)
    padding_mask[~mask] = value
    return padding_mask

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = am.PositionalEncoding(num_hiddens, dropout)
        encoder_layer = TransformerEncoderLayer(num_hiddens, num_heads, dim_feedforward=ffn_num_hiddens, dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        src_key_padding_mask = sequence_mask(X, valid_lens)
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        X = X.permute(1, 0, 2)
        X = self.transformer_encoder(src=X, src_key_padding_mask=src_key_padding_mask)
        X = X.permute(1, 0, 2)
        return X

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = am.PositionalEncoding(num_hiddens, dropout)
        decoder_layer = TransformerDecoderLayer(num_hiddens, num_heads, dim_feedforward=ffn_num_hiddens, dropout=dropout, activation='relu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.dense = nn.Linear(num_hiddens, vocab_size)
    

    def init_state(self, enc_outputs, enc_valid_lens):
        memory_key_padding_mask = sequence_mask(torch.zeros_like(enc_outputs[:,:, 0]).to(enc_outputs.device), enc_valid_lens)
        return [enc_outputs, memory_key_padding_mask]

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, X, state, valid_lens):
        tgt_key_padding_mask = sequence_mask(X, valid_lens)
        sz=X.shape[1]
        tgt_mask = self.generate_square_subsequent_mask(sz)
        tgt_mask = tgt_mask.to(X.device)
        memory = state[0]
        memory_key_padding_mask = state[1]
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        X = X.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        # X = self.transformer_decoder(tgt=X, memory=memory, tgt_mask=tgt_mask,
        #                             memory_key_padding_mask=memory_key_padding_mask)
        X = self.transformer_decoder(tgt=X, memory=memory, tgt_mask=tgt_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask, 
                                     memory_key_padding_mask=memory_key_padding_mask)
        X = X.permute(1, 0, 2)
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, X_valid_lens, Y_valid_lens):
        enc_outputs = self.encoder(enc_X, X_valid_lens)
        # print(enc_outputs.shape, enc_X.shape)
        dec_state = self.decoder.init_state(enc_outputs, X_valid_lens)
        return self.decoder(dec_X, dec_state, Y_valid_lens)

def train_seq2seq(net, data_iter, lr, num_epochs, batch_size, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    torch.nn.init.xavier_uniform_(m._parameters[param])
    # net.apply(xavier_init_weights)
    try:
        checkpoint_prefix = os.path.join("model_data/model_transformer.pt")
        checkpoint = torch.load(checkpoint_prefix)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(device)
        # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    except Exception as e:
        net.apply(xavier_init_weights)
        # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
        net.to(device)
        print("Can not load the model with error:", e)
    
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    loss = am.MaskedSoftmaxCELoss()
    net.train()
    animator = am.Animator(xlabel='epoch', ylabel='loss')

    
    checkpoint_prefix = os.path.join("model_data/model_transformer.pt")
    num_trained = 0
    metric = am.Accumulator(2)  # Sum of training loss, no. of tokens
    for epoch in range(num_epochs):
        timer = Utility.Timer()
        for i, batch in enumerate(data_iter):
            num_trained += 1
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len, Y_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            # Utility.grad_clipping(net, 1)
            optimizer.step()

            with torch.no_grad():
                num_tokens = Y_valid_len.sum()
                metric.add(l.sum(), num_tokens)

            if (num_trained + 1) % 50 == 0:
                
                animator.add(num_trained + 1, (metric[0] / metric[1],))
                # print(f'epoch = {epoch}, loss = {metric[0] / metric[1]:.3f}')
                torch.save({'model_state_dict': net.state_dict(), "optimizer": optimizer.state_dict()},checkpoint_prefix)
                metric = am.Accumulator(2)  # Sum of training loss, no. of tokens

    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence."""
    # Set `net` to eval mode for inference
    net = net.to(device)
    net.eval()
    src_sentence = [word for word in src_sentence]
    print("src",src_sentence)
    src_tokens = src_vocab[src_sentence] + [
        src_vocab['<eos>']]
    print("src_tokens=", src_tokens)
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = dlft.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    print("tp_src_tokens", src_tokens)
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    # enc_outputs = self.encoder(enc_X, X_valid_lens)
    # # print(enc_X.shape, enc_outputs.shape)
    # dec_state = self.decoder.init_state(enc_outputs, X_valid_lens)
    # self.decoder(dec_X, dec_state, Y_valid_lens)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    print("enc_outputs", enc_outputs)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    print("dec_state", dec_state)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    
    output_seq, attention_weight_seq = [], []
  
    for _ in range(num_steps):
        Y_valid_lens = torch.tensor([dec_X.shape[1]], device=device)
        print("Y_valid_lens", Y_valid_lens)
        Y, dec_state = net.decoder(dec_X, dec_state, Y_valid_lens)
        print("Y", Y[:,:,:10], Y.argmax(dim=2), Y.max(dim=2))
        pred_X = Y.argmax(dim=2)[:, -1]
        dec_X = torch.cat((dec_X,  torch.unsqueeze(pred_X, dim=0)), axis=1)
        print("dec_X", dec_X)
        pred = pred_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['[SEP]']:
            print("break", tgt_vocab.to_tokens(pred), pred)
            break
        output_seq.append(pred)


    print(output_seq)
    print(tgt_vocab.to_tokens(output_seq))
    # return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
    return ' '.join(tgt_vocab.to_tokens(output_seq))

def train_bert(net, data_iter, lr, num_epochs, batch_size, tgt_vocab, device, cp = os.path.join("model_data/model_bert.pt")):
    """Train a model for sequence to sequence."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    torch.nn.init.xavier_uniform_(m._parameters[param])
    # net.apply(xavier_init_weights)
    try:
        checkpoint_prefix = cp
        checkpoint = torch.load(checkpoint_prefix)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        print("load model success")
    except Exception as e:
        net.apply(xavier_init_weights)
        net.to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
        print("Can not load the model with error:", e)

    
    loss = am.MaskedSoftmaxCELoss()
    net.train()
    # animator = am.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs*batch_size])
    animator = am.Animator(xlabel='epoch', ylabel='loss')

    
    checkpoint_prefix = cp
    # ratio = 100 / len(data_iter)
    # print("ratio=", ratio)
    num_trained = 0
    for epoch in range(num_epochs):
        timer = Utility.Timer()
        metric = am.Accumulator(2)  # Sum of training loss, no. of tokens
        # print("epoch ...", epoch)
        for i, batch in enumerate(data_iter):
            # if random.random() < (1 - ratio * 1.5):
            #     continue
            num_trained += 1
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # X_tokens = tgt_vocab.to_tokens(X.cpu().detach().numpy())
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len, Y_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            # Utility.grad_clipping(net, 1)
            
            optimizer.step()
            with torch.no_grad():
                num_tokens = Y_valid_len.sum()
                metric.add(l.sum(), num_tokens) 
            
            # if (i + 1) % 100 == 0:
            # print("    batch>>>", i)
            if (num_trained + 1) % 100 == 0:
                animator.add(num_trained + 1, (metric[0] / metric[1],))
                # print(f'epoch = {epoch}, loss = {metric[0] / metric[1]:.3f}')
                torch.save({'model_state_dict': net.state_dict(), "optimizer": optimizer.state_dict()},checkpoint_prefix)
        # if (epoch + 1) % 10 == 0:
        # animator.add(epoch + 1, (metric[0] / metric[1],))
        # # print(f'epoch = {epoch}, loss = {metric[0] / metric[1]:.3f}')
        # torch.save({'model_state_dict': net.state_dict(), "optimizer": optimizer.state_dict()},checkpoint_prefix)
        # sys.stdout.flush()
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')