from typing import List, Dict
import codecs
import torch
import sys
import myutils
from transformers import AutoModel, AutoTokenizer
import numpy as np

# set seed for consistency
torch.manual_seed(8446)
# Set some constants
MLM = 'distilbert-base-cased'
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
EPOCHS = 3
# We have an UNK label for robustness purposes, it makes it easier to run on
# data with other labels, or without labels.
UNK = "[UNK]"
MAX_TRAIN_SENTS=64 #CHANGE
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_TRAIN_SENTS=64

output_list =[]

class ClassModel(torch.nn.Module):
    def __init__(self, nlabels: int, mlm: str):
        """
        Model for classification with transformers.

        The architecture of this model is simple, we just have a transformer
        based language model, and add one linear layer to converts it output
        to our prediction.
    
        Parameters
        ----------
        nlabels : int
            Vocabulary size of output space (i.e. number of labels)
        mlm : str
            Name of the transformers language model to use, can be found on:
            https://huggingface.co/models
        """
        super().__init__()

        # The transformer model to use
        self.mlm = AutoModel.from_pretrained(mlm)

        # Find the size of the output of the masked language model
        if hasattr(self.mlm.config, 'hidden_size'):
            self.mlm_out_size = self.mlm.config.hidden_size
        elif hasattr(self.mlm.config, 'dim'):
            self.mlm_out_size = self.mlm.config.dim
        else: # if not found, guess
            self.mlm_out_size = 768
            
        print(f"Hidden size: {self.mlm_out_size}")

        # Create prediction layer
        self.hidden_to_label = torch.nn.Linear(self.mlm_out_size, nlabels)

    def forward(self, input: torch.tensor):
        """
        Forward pass
    
        Parameters
        ----------
        input : torch.tensor
            Tensor with wordpiece indices. shape=(batch_size, max_sent_len).

        Returns
        -------
        output_scores : torch.tensor
            ?. shape=(?,?)
        """
        # Run transformer model on input
        mlm_out = self.mlm(input)

        # Keep only the last layer: shape=(batch_size, max_len, DIM_EMBEDDING)
        mlm_out = mlm_out.last_hidden_state
        # Keep only the output for the first ([CLS]) token: shape=(batch_size, DIM_EMBEDDING)
        mlm_out = mlm_out[:,:,:].squeeze() 

        # Matrix multiply to get scores for each label: shape=(?,?)
        output_scores = self.hidden_to_label(mlm_out)

        return output_scores

    def run_eval(self, feats_batches, labels_batches):
        """
        Run evaluation: predict and score
    
        Parameters
        ----------
        text_batched : List[torch.tensor]
            list with batches of text, containing wordpiece indices.
        labels_batched : List[torch.tensor]
            list with batches of labels (converted to ints).
        model : torch.nn.module
            The model to use for prediction.
    
        Returns
        -------
        score : float
            accuracy of model on labels_batches given feats_batches
        """
        self.eval()
        match = 0
        total = 0
        for sents, labels in zip(feats_batches, labels_batches):
            output_scores = self.forward(sents)
            predicted_tags  = torch.argmax(output_scores, 2)
            for goldSent, predSent in zip(labels, predicted_tags):
                for goldLabel, predLabel in zip(goldSent, predSent):
                    if goldLabel.item() != 0:
                        total += 1
                        if goldLabel.item() == predLabel.item():
                            match+= 1
        return(match/total)
    
    def run_test(self, X, y):
        """
        Run evaluation: predict and score

        Parameters
        ----------
        feats_batches : torch.tensor
            Batches of text, containing wordpiece indices.
        labels_batches : torch.tensor
            Batches of labels (converted to ints).

        Returns
        -------
        score : float
            Accuracy of the model on labels_batches given feats_batches
        """
        self.eval()
        match = 0
        total = 0
        for sents, labels in zip(X, y):
            # print("Shape test bf forward: ", sents.shape)
            output_scores = self.forward(sents)
            # print("Shape output test bf argmax: ", output_scores.shape)
            predicted_labels = torch.argmax(output_scores, 2)
            for goldSent, predSent in zip(labels, predicted_labels):
                for goldLabel, predLabel in zip(goldSent, predSent):
                    if goldLabel.item() != 0:
                        total += 1
                        if goldLabel.item() == predLabel.item():
                            match += 1
        return match / total

def tok(data, tokzr: AutoTokenizer):

    tok_data = []

    for sent in data:
        tok_data.append(tokzr.encode(sent))

    return tok_data

def labels2lookup(labels, PAD):

    id2label = [PAD, 'O', 'B-LOC', 'I-LOC', 'B-PER', 'B-ORG', 'I-ORG', 'I-PER']
    label2id = {PAD: 0, 'O': 1, 'B-LOC': 2, 'I-LOC': 3, 'B-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'I-PER': 7}
                
    return id2label, label2id

def read_data(path):
    """
    read in iob2 file
    
    :param path: path to read from
    :returns: list with sequences of words and labels for each sentence
    """
    data_words = []
    data_tags = []
    current_words = []
    current_tags = []

    for line in open(path, encoding='utf-8'):
        line = line.strip()

        if line:
            if line[0] == '#':
                continue # skip comments
            tok = line.split('\t')

            current_words.append(tok[1])
            current_tags.append(tok[2])
        else:
            if current_words:  # skip empty lines
                current_words = ' '.join(current_words)
                data_words.append(current_words)
                data_tags.append(current_tags)
            current_words = []
            current_tags = []
    # check for last one
    if current_tags != []:
        current_words = ' '.join(current_words)
        data_words.append(current_words)
        data_tags.append(current_tags)
    return data_words, data_tags

def find_max_len(data):
    
    max_len = max([len(x) for x in data])
    
    return max_len

def pad_data(data, PAD, N):
    
    padded = []
    
    for sent in data:
        
        new_sent = sent.copy()
        
        sent_len = len(sent)
        dif = N - sent_len
        
        for _ in range(dif):
            
            new_sent.append(PAD)
            
        padded.append(new_sent)
        
    return padded

#-----
def predict_and_save(model, test_data, output_file):
    model.eval()
    with open(output_file, 'w') as f:
        for batch_idx in range(len(test_data)):
            output_scores = model.forward(test_data[batch_idx])
            predicted_labels = torch.argmax(output_scores, 2)
            for pred_sent in predicted_labels:
                pred_tags = [id2label[label_id.item()] for label_id in pred_sent]
                f.write(' '.join(pred_tags) + '\n')
#-----

if len(sys.argv) < 2:
    print('Please provide path to training and development data')

if __name__ == '__main__':

    train_text, train_labels = read_data(sys.argv[1])
    dev_text, dev_labels = read_data(sys.argv[2])
    test_text, test_labels = read_data(sys.argv[3])
    train_text = train_text[:MAX_TRAIN_SENTS] 
    train_labels = train_labels[:MAX_TRAIN_SENTS]

    id2label, label2id = labels2lookup(train_labels, UNK)
    NLABELS = len(id2label)
    #print(id2label)
    #print(label2id)
    #print(NLABELS)

    enc_labels_train = train_labels.copy() 

    for i, label_list in enumerate(train_labels):
            for j, label in enumerate(label_list):
                enc_labels_train[i][j]= label2id[label]
                
                
    enc_labels_dev = dev_labels.copy() 

    for i, label_list in enumerate(dev_labels):
            for j, label in enumerate(label_list):
                enc_labels_dev[i][j]= label2id[label]

    enc_labels_test = test_labels.copy() 

    for i, label_list in enumerate(test_labels):
            for j, label in enumerate(label_list):
                enc_labels_test[i][j]= label2id[label]
                
    tokzr = AutoTokenizer.from_pretrained(MLM)
    train_tokked = tok(train_text, tokzr)
    dev_tokked = tok(dev_text, tokzr)
    test_tokked = tok(test_text, tokzr)
    PAD = tokzr.pad_token_id

    MAX_SENT_LEN = find_max_len(train_tokked)
    MAX_SENT_LEN_DEV = find_max_len(dev_tokked)
    MAX_SENT_LEN_TEST = find_max_len(test_tokked)
    #print(f"Max sent len in training set is {MAX_SENT_LEN}")
    #print(f"Max sent len in dev set is {MAX_SENT_LEN_DEV}")

    padded_train_txt = pad_data(train_tokked, PAD, MAX_SENT_LEN)
    padded_dev_txt = pad_data(dev_tokked, PAD, MAX_SENT_LEN_DEV)
    padded_test_txt = pad_data(test_tokked, PAD, MAX_SENT_LEN_TEST)
    padded_train_labels = pad_data(enc_labels_train, PAD, MAX_SENT_LEN)
    padded_dev_labels = pad_data(enc_labels_dev, PAD, MAX_SENT_LEN_DEV)
    padded_test_labels = pad_data(enc_labels_test, PAD, MAX_SENT_LEN_TEST)

    train_labels = torch.tensor(padded_train_labels)
    dev_labels = torch.tensor(padded_dev_labels)
    test_labels = torch.tensor(padded_test_labels)

    train_txt = np.array(padded_train_txt)
    train_txt = torch.tensor(train_txt)
    #print(train_txt.shape)

    dev_txt = np.array(padded_dev_txt)
    dev_txt = torch.tensor(dev_txt)
    #print(dev_txt.shape)

    test_txt = np.array(padded_test_txt)
    test_txt = torch.tensor(test_txt)
    print("Shape test_txt: ", test_txt.shape)
    print("Shape test_labels: ", test_labels.shape)

    num_batches = int(len(train_txt)/BATCH_SIZE)
    #print(f"num batches: {num_batches}")
    #batches data
    train_txt_batches = train_txt[:BATCH_SIZE*num_batches].view(num_batches, BATCH_SIZE, MAX_SENT_LEN)
    train_labels_batches = train_labels[:BATCH_SIZE*num_batches].view(num_batches, BATCH_SIZE, MAX_SENT_LEN)

    #print(train_txt_batches.shape)
    #print(train_labels_batches.shape)

    num_batches_dev = int(len(dev_txt)/BATCH_SIZE)
    #print(f"num batches: {num_batches_dev}")
    dev_txt_batches = dev_txt[:BATCH_SIZE*num_batches_dev].view(num_batches_dev, BATCH_SIZE, MAX_SENT_LEN_DEV)
    dev_labels_batches = dev_labels[:BATCH_SIZE*num_batches_dev].view(num_batches_dev, BATCH_SIZE, MAX_SENT_LEN_DEV)

    #print(dev_txt_batches.shape)
    #print(dev_labels_batches.shape)
    
    #commented this out cuz we don't batch the test datas
    num_batches_test = int(len(test_txt)/BATCH_SIZE)
    test_txt_batches = test_txt[:BATCH_SIZE*num_batches_test].view(num_batches_test, BATCH_SIZE, MAX_SENT_LEN_TEST)
    test_labels_batches = test_labels[:BATCH_SIZE*num_batches_test].view(num_batches_test, BATCH_SIZE, MAX_SENT_LEN_TEST)

    model = ClassModel(NLABELS, MLM)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    #train on a subset of the batches - REMOVE IN TESTING
    dev_txt_subset = dev_txt_batches[:2,:,:]
    dev_label_subset = dev_labels_batches[:2,:,:]

    # We also don't subset test data
    # test_txt_subset = test_txt_batches[:2,:,:]
    # test_label_subset = test_labels_batches[:2,:,:]

    for epoch in range(EPOCHS):
        print('=====================')
        print('starting epoch ' + str(epoch))
        model.train() 
    
        # Loop over batches
        loss = 0
        for batch_idx in range(0, len(train_txt_batches)):

            print(f"---running batch idx {batch_idx}---")
            print(f"size of current batch: {train_txt_batches[batch_idx].shape}")

            optimizer.zero_grad()
            
            output_scores = model.forward(train_txt_batches[batch_idx])
            
            flat_labels = train_labels_batches[batch_idx].view(BATCH_SIZE * MAX_SENT_LEN)
            output_scores = output_scores.view(BATCH_SIZE * MAX_SENT_LEN, -1)
            print("output scores shape", output_scores.shape)
            batch_loss = loss_function(output_scores, flat_labels)
            
            predicted_labels = torch.argmax(output_scores, 1)         
            #predicted_labels = predicted_labels.view(BATCH_SIZE, MAX_SENT_LEN)

            print("train labels in a single batch: ", train_labels_batches[batch_idx].shape)
            print(f"predicted labels size: {predicted_labels.shape}")

            loss += batch_loss.item()
    
            batch_loss.backward()

            optimizer.step()

        dev_score = model.run_eval(dev_txt_subset, dev_label_subset)
        print('Loss: {:.2f}'.format(loss))
        print('Acc(dev): {:.2f}'.format(100*dev_score))
        print()
        
    
    # test_score = model.run_test(test_txt_batches, test_labels_batches)
    # # print('Acc(test): {:.2f}'.format(100 * test_score))

output_file = 'predictions.txt'
predict_and_save(model, test_txt_batches, output_file)