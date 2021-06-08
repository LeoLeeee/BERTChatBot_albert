# BERTChatBot

Using Transformer (with pytorch) and ALBERT (with hugging face) to code a Chinese Chat Bot

## Files' function

1. seq2seq.ini and getConfig.py are used to config and load the parameter used in the project, such as the path of the checkpoint.
2. plot_util.py and Utility.py are copy the function in "dive into deep learning" to plot data and get some convenient tools.
3. AttentionModel.py are copy and revised based on the "dive into deep learning" to offer some useful tools.
4. BERTModel.py are using pytorch transformer and hugging face pre-trained model to building our model. And I also copy and revised the train and predict function of the models from "dive into deep learning".
5. data_tokenize.ipynb is used to create sentence used for pre-training. The data_tokenize.py is used to save the useful function for other scripts to import.
6. data_load_for_Transformer.py/ipynb are used to construct train_iter for the training from the "train_data/xiaohuangji50w_nofenci.conv" based on the tokenized vocabulary. It also generate the vocab.txt for albert model pre-training. Because I do not find the Chinese character division by the hugging face, so I use the jieba and build the vocab.txt. Then the pre-training model could build the tokenizer from the vocab.txt directly.
7. Albert_pretraining.ipynb is used to pre-train the albert model.
8. excute.py/ipynb are used to train the chat bot net with transformer encoder and decoder, excute_bert.py/ipynb are the net with albert encoder from pre-trained.
9. app.py builds a web to chat with bot based on JS. Changing the importing "excute" python file to use the excute.py or excute_bert.py of different model.

## How to use the code

1. Download the code on your computer.
2. You may need to create a folder named "model_data" to save the checkpoint of the encoder and decoder. You also need to create a folder named "output" to save the checkpoint of the pre-trained model.
3. Run the excute.ipynb until the train loss converge (Loss may converge to ~0.22). This may take several hours based on the 1060Ti GPU.
4. Importing excute.py file in the app.py to use the trained model. Run app.py and open http://127.0.0.1:8808/ in your browser. Then you can chat with your bot.
5. Running data_tokenize.ipynb to create the pre-training data. Pretraining the BERT model with the Albert_pretraining.ipynb before using the Albert model as encoder. Then running excute_bert.ipynb to train the encoder and decoder until the loss converge (Loss may converge to 0.16). This may take several hours based on the 1060Ti GPU.
6. Importing excute_bert.py as excute file in the app.py to use the trained model. Run app.py and open http://127.0.0.1:8808/ in your browser. Then you can chat with your bot.

## Note

1. You may need to install the pytorch and transformer (hugging face pre-train model package.)
2. The training data is not very well, so the effect may not very well. You could changing the corpus.