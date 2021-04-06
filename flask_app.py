from flask import Flask, request,render_template, g, session, redirect, url_for
import os
import re
import csv
from nltk import tokenize
from uuid import uuid4

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
import pickle

from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import cosine

from torch.autograd import Variable
if_linear_embedding = True
max_len_seq_lstm = 20
max_seq_len = 30

# from rq import Queue
# from worker import conn

# q = Queue(connection=conn)

class CONFIG:
    def __init__(self):
        self.UNK = "UNK"
        self.max_seq_len = max_seq_len
        self.padding_idx = 0
        self.padding_token = 'EOS'
cfg = CONFIG()

class TwoLevelLstmClassifier_MALCOM_setting(nn.Module):
    def __init__(self, embedding_dim, seq_hidden_dim, seqs_hidden_dim, vocab_size, label_size,
                 max_len_seq_lstm, max_len_sent):
        super(TwoLevelLstmClassifier_MALCOM_setting, self).__init__()

        self.embedding_dim = embedding_dim
        self.seq_hidden_dim = seq_hidden_dim
        self.seqs_hidden_dim = seqs_hidden_dim
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.max_len_seq_lstm = max_len_seq_lstm
        self.max_len_sent = max_len_sent
        self.emb = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.seq_lstm = nn.LSTM(embedding_dim, seq_hidden_dim)
        self.seqs_lstm = nn.LSTM(seq_hidden_dim, seqs_hidden_dim)
        self.linear = nn.Linear(seq_hidden_dim+seqs_hidden_dim, label_size)

    def forward(self, sequences):
        """
        :param sequences: a seq of sentence, shape [max_len_sent]*max_len_seq; i.e. [max_len_seq]*max_len_sent
        :return:
        """
        # sequences format: an article + multile comments.
        article_emb = self.emb(sequences[0])  # 1*max_len_sent*embed_dim
        comments_emb = self.emb(sequences[1]) # not at the same length, thus, we use
        # by change axis, or view-- how to decide the index ordering is correct
        _, (article_h_n, _) = self.seq_lstm(article_emb.permute(1, 0, 2))  # 1*1*hidden_emb
        _, (comments_h_n, _) = self.seq_lstm(comments_emb.permute(1, 0, 2)) # 1*n*hidden_emb
        _, (comments_h_n, _) = self.seqs_lstm(comments_h_n.permute(1, 0, 2))  # 1*1*hidden_emb_2
        # print(hn_seqs.shape) # (1, 1, hidden_dim)
        feature_vec = torch.cat((article_h_n.squeeze(0), comments_h_n.squeeze(0)), dim=1) # (1, all_feature_dimension)
        # print(feature_vec.shape)
        pred_label = self.linear(feature_vec)  # 1*label_size
        return pred_label

def tokens_to_tensor(tokens, dictionary, one_long = False, one_long_if_pad = False):
    """transform word tokens to Tensor"""
    ## tokens_to_tensor(tokenlized, self.test_data.word2idx_dict)
    # todo: github: fix the bugs , we should prepend not apprend??but in the mle, not sure?
    if one_long:
        tensor = []
        for word in tokens:
            try:
                tensor.append(int(dictionary[str(word)]))
            except KeyError:
                # print(f"we have key error in word: {word} and ignor it")
                tensor.append(int(dictionary[cfg.UNK]))
        if one_long_if_pad:
            # too short
            while len(tensor) < cfg.max_seq_len:
                # now
                # tensor = [cfg.padding_idx]*(cfg.max_seq_len-len(tensor)) + tensor
                # prev: error
                tensor.append(cfg.padding_idx)
            # too long
            tensor = tensor[: cfg.max_seq_len]
        return torch.LongTensor(tensor)

    if not one_long:
        global i
        tensor = []
        for sent in tokens:
            sent_ten = []
            # print(len(sent))
            for i, word in enumerate(sent):
                if word == cfg.padding_token:
                    break
                try:
                    sent_ten.append(int(dictionary[str(word)]))
                except KeyError: # KeyError
                    # print(sent)
                    # print(f"we have key error in word: {word}")
                    sent_ten.append(int(dictionary[cfg.UNK]))
            # # ==== approach 0 ====: by index
            # while i < cfg.max_seq_len - 1:
            #     sent_ten.append(cfg.padding_idx)
            #     i += 1
            # ==== approach 1 ====: by the length
            while len(sent_ten) < cfg.max_seq_len:
                # prev
                sent_ten.append(cfg.padding_idx)
                # now
                # sent_ten = [cfg.padding_idx] * (cfg.max_seq_len - len(sent_ten)) + sent_ten
            # print(f"the parameter is: {cfg.max_seq_len}")
            tensor.append(sent_ten[:cfg.max_seq_len])
        return torch.LongTensor(tensor)


dataset = "GOSSIP"
word2idx_dict = pickle.load(open(f"{dataset}_word2idx_dict.pkl", "rb"))
idx2word_dict = pickle.load(open(f"{dataset}_idx2word_dict.pkl", "rb"))

EMBEDDING_DIM = 256 * 2  # # LSTM_LAYERS = 1
num_label = 2
SEQ_HIDDEN_DIM = 128 * 2
SEQS_HIDDEN_DIM = 64 * 2

device = "cpu"

clf_seq = TwoLevelLstmClassifier_MALCOM_setting(EMBEDDING_DIM, SEQ_HIDDEN_DIM, SEQS_HIDDEN_DIM,
                               len(word2idx_dict), num_label,
                                max_len_seq_lstm=max_len_seq_lstm, max_len_sent=max_seq_len).to(device)


pretrained_clf_path = r"CLF_model100_epoch399.pt"


clf_seq.load_state_dict(torch.load(pretrained_clf_path, map_location='cpu'))



tweets = []
com_tweets = []
labels = []
texts = []
com_texts = []
glove_dir = "./glove.6B"
embeddings_index = {}
prediction_text = {}
wordss = {}
comm = {}
prob = {}
bleu = {}
one_relevancy_score = {}






data_dic = {}   

gossip_ids = []
gosscontents = []
gosscoms = []





def tensor_to_tokens(tensor, dictionary):
# """transform Tensor to word tokens"""
    tokens = []
    for sent in tensor:
        sent_token = []
        for word in sent.tolist():
            if word == cfg.padding_idx:
                break
            sent_token.append(dictionary[str(word)])
        tokens.append(sent_token)
    return tokens

def word_token_list2feature_vector(list_of_list_of_tokens, vocab_size):
    contexts_matrix = np.zeros((len(list_of_list_of_tokens), vocab_size))
    for i, context in enumerate(list_of_list_of_tokens):
        for word_id in context:
            contexts_matrix[i, word_id] += 1
    return contexts_matrix

def tokens_to_tensor(tokens, dictionary, one_long = False, one_long_if_pad = False):
    # """transform word tokens to Tensor"""
    ## tokens_to_tensor(tokenlized, self.test_data.word2idx_dict)
    # todo: github: fix the bugs , we should prepend not apprend??but in the mle, not sure?
    if one_long:
        tensor = []
        for word in tokens:
            try:
                tensor.append(int(dictionary[str(word)]))
            except KeyError:
                # print(f"we have key error in word: {word} and ignor it")
                tensor.append(int(dictionary[cfg.UNK]))
        if one_long_if_pad:
            # too short
            while len(tensor) < cfg.max_seq_len:
                # now
                # tensor = [cfg.padding_idx]*(cfg.max_seq_len-len(tensor)) + tensor
                # prev: error
                tensor.append(cfg.padding_idx)
            # too long
            tensor = tensor[: cfg.max_seq_len]
        return torch.LongTensor(tensor)

    if not one_long:
        global i
        tensor = []
        for sent in tokens:
            sent_ten = []
            # print(len(sent))
            for i, word in enumerate(sent):
                if word == cfg.padding_token:
                    break
                try:
                    sent_ten.append(int(dictionary[str(word)]))
                except KeyError: # KeyError
                    # print(sent)
                    # print(f"we have key error in word: {word}")
                    sent_ten.append(int(dictionary[cfg.UNK]))
            # # ==== approach 0 ====: by index
            # while i < cfg.max_seq_len - 1:
            #     sent_ten.append(cfg.padding_idx)
            #     i += 1
            # ==== approach 1 ====: by the length
            while len(sent_ten) < cfg.max_seq_len:
                # prev
                sent_ten.append(cfg.padding_idx)
                # now
                # sent_ten = [cfg.padding_idx] * (cfg.max_seq_len - len(sent_ten)) + sent_ten
            # print(f"the parameter is: {cfg.max_seq_len}")
            tensor.append(sent_ten[:cfg.max_seq_len])
        return torch.LongTensor(tensor)


tsv_file = open("mergedvals25noheader.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")


for row in read_tsv:
    gossip_ids.append(row[1])
    gosscontents.append(row[2])
    gosscoms.append(row[8])
    data_dic[row[1]] = {}
    data_dic[row[1]]['content'] = row[2]
    data_dic[row[1]]['com1'] = row[3]
    data_dic[row[1]]['com2'] = row[4]
    data_dic[row[1]]['com3'] = row[5]
    data_dic[row[1]]['com4'] = row[6]
    data_dic[row[1]]['com5'] = row[7]
    data_dic[row[1]]['comments'] = row[8]
tsv_file.close()



def get_vars(gosscontents,word2idx_dict):
    corpus = []
    for gosscontent in gosscontents:
        corpus += gosscontent
    for gosscom in gosscoms:
        corpus += gosscom

    reference = [ nltk.word_tokenize(sent.lower()) for sent in corpus]

    articles_tensor_format = []
    for articlee in gosscontents:
       
        articlee = nltk.word_tokenize(articlee.lower())
        articlee = tokens_to_tensor(articlee, word2idx_dict, one_long=True)
        articles_tensor_format.append(articlee.view(1, -1))
    article_token_list = [articlee.tolist() for articlee in articles_tensor_format] # torch.cat(self.contexts, dim=0).tolist()
    contexts_matrix = word_token_list2feature_vector(article_token_list, len(word2idx_dict))
    ldas = []
    for num_clusters in range(2, 20, 5):
        lda = LatentDirichletAllocation(n_components=num_clusters)
        lda.fit(contexts_matrix)
        ldas.append(lda)

    return corpus, reference, ldas

# corpus, reference, ldas = q.enqueue(get_vars(gosscontents,word2idx_dict), 'http://heroku.com')
corpus, reference, ldas = get_vars(gosscontents,word2idx_dict)



app = Flask(__name__)
app.secret_key = ".."


@app.route('/<username>', methods=['GET', 'POST'])
def home(username):
    global user
    user=username
    if user not in prediction_text:
        prediction_text[user] = {}
    if user not in wordss:
        wordss[user] = {}
    if user not in comm:
        comm[user] = {}
    if user not in prob:
        prob[user] = {}
    if user not in bleu:
        bleu[user] = {}
    if user not in one_relevancy_score:
        one_relevancy_score[user] = {}
    if user in session:
        return 'hello {}'.format(user)
    else:
        session[user] = username
        return 'login as {}'.format(username)


@app.route('/logout/<username>', methods=['GET', 'POST'])
def logout(username):
    with open("logout.txt", "a") as myfile:
        myfile.write(str(username))
        if username in prediction_text:
            myfile.write(str(prediction_text[username]))
            del prediction_text[username]
        if username in wordss:
            myfile.write(str(wordss[username]))
            del wordss[username]
        if username in comm:
            myfile.write(str(comm[username]))
            del comm[username]
        if username in prob:
            myfile.write(str(prob[username]))
            del prob[username]
        if username in bleu:
            myfile.write(str(bleu[username]))
            del bleu[username]
        if username in one_relevancy_score:
            myfile.write(str(one_relevancy_score[username]))
            del one_relevancy_score[username]
        myfile.write("\n")
    session.pop(username, None)
    return '{} logout!'.format(username)

@app.route('/')
def index():

    return render_template('index.html', gossip_ids=gossip_ids, gosscontents=gosscontents)


@app.route('/v_timestamp/<string:id>')
def v_timestamp(id):
    id = id
    tsv_file = open("mergedvals25noheader.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    
    comments = [data_dic[id]['comments']]


    
    art = data_dic[id]['content']

    org_article = nltk.word_tokenize(art)
    org_comment = [nltk.word_tokenize(each_comment)  for each_comment in comments]

    org_article = tokens_to_tensor(org_article, word2idx_dict, one_long=True).view(1, -1).to(device)
    org_comment = tokens_to_tensor(org_comment, word2idx_dict).to(device)

    org_article = F.one_hot(org_article, len(word2idx_dict)).float()
    org_comment = F.one_hot(org_comment, len(word2idx_dict)).float()
    org_ret = F.softmax(clf_seq([org_article, org_comment]))
    org_vals = org_ret.detach().numpy()[0]
    if org_vals[0]>org_vals[1]:
        org_class = "Not Misinformation"
        org_prob = org_vals[0]
    else:
        org_class = "Misinformation"
        org_prob = org_vals[1]

    com1 = data_dic[id]['com1']
    com2 = data_dic[id]['com2']
    com3 = data_dic[id]['com3']
    com4= data_dic[id]['com4']
    com5 = data_dic[id]['com5']





    return render_template('v_timestamp_new.html', org_class = org_class, org_prob = org_prob, art = art, id=id, comments = comments, 
        com1 = com1, com2 = com2, com3 = com3, com4 = com4, com5 = com5  
        )

@app.route("/", methods=["POST"])
def predict():
    id = request.form['abc']


    art = data_dic[id]['content']



    comments = [data_dic[id]['comments']]

    new_comm = request.form['search']


    
    candidate = nltk.word_tokenize(new_comm.lower())

    one_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    two_gram = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
    three_gram = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
    four_gram = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))

    bleue = (two_gram+three_gram+four_gram)/4



    generated_sent_in_token = nltk.word_tokenize(new_comm.lower())
    generated_sent_in_token = tokens_to_tensor(generated_sent_in_token, word2idx_dict, one_long=True).view(1, -1)
    generated_texts_in_feature_vectors = word_token_list2feature_vector(generated_sent_in_token.tolist(), len(word2idx_dict))

    target_article_in_token = nltk.word_tokenize(art.lower())
    target_article_in_token = tokens_to_tensor(target_article_in_token, word2idx_dict, one_long=True).view(1, -1)
    target_article_in_feature_vectors = word_token_list2feature_vector(target_article_in_token.tolist(), len(word2idx_dict))

    one_relevancy_scoree = 0
    for index, lda in enumerate(ldas):
        
        one_relevancy_scoree += cosine(lda.transform(generated_texts_in_feature_vectors.reshape(1, -1)),
                                      lda.transform(target_article_in_feature_vectors.reshape(1, -1)))
    one_relevancy_scoree= one_relevancy_scoree / (len(ldas))
    
    new_comment=[new_comm]

    comment = comments + new_comment



    article = nltk.word_tokenize(art)
    comment = [nltk.word_tokenize(each_comment)  for each_comment in comment]

    article = tokens_to_tensor(article, word2idx_dict, one_long=True).view(1, -1).to(device)
    comment = tokens_to_tensor(comment, word2idx_dict).to(device)

    # preprocessing before the embedding
    article = F.one_hot(article, len(word2idx_dict)).float()
    comment = F.one_hot(comment, len(word2idx_dict)).float()
    ret = F.softmax(clf_seq([article, comment]))
    class_vals = ret.detach().numpy()[0]
    

    if class_vals[0]>class_vals[1]:
        coms = "Not Misinformation"
        probab = class_vals[0]
    else:
        coms = "Misinformation"
        probab = class_vals[1]

    if id in prediction_text[session[user]]:
        prediction_text[session[user]][id].append(coms)
    else:
        prediction_text[session[user]][id] = [coms]
        
    if id in comm[session[user]]:
        comm[session[user]][id].append(new_comm)
    else:
        comm[session[user]][id] = [new_comm]


    if id in prob[session[user]]:
        prob[session[user]][id].append(probab)
    else:
        prob[session[user]][id] = [probab]

    if id in bleu[session[user]]:
        bleu[session[user]][id].append(bleue)
    else:
        bleu[session[user]][id] = [bleue]

    if id in one_relevancy_score[session[user]]:
        one_relevancy_score[session[user]][id].append(one_relevancy_scoree)
    else:
        one_relevancy_score[session[user]][id] = [one_relevancy_scoree]

    org_comment = [nltk.word_tokenize(each_comment)  for each_comment in comments]


    org_comment = tokens_to_tensor(org_comment, word2idx_dict).to(device)


    org_comment = F.one_hot(org_comment, len(word2idx_dict)).float()
    org_ret = F.softmax(clf_seq([article, org_comment]))
    org_vals = org_ret.detach().numpy()[0]
    if org_vals[0]>org_vals[1]:
        org_class = "Not Misinformation"
        org_prob = org_vals[0]
    else:
        org_class = "Misinformation"
        org_prob = org_vals[1]

    print(session[user])
    print(",")
    print(str(id))
    print(",")
    print(str(coms) )
    print(",")
    print(str(new_comm))
    print(",")
    print(str(probab)) 
    print("\n")
    com1 = data_dic[id]['com1']
    com2 = data_dic[id]['com2']
    com3 = data_dic[id]['com3']
    com4= data_dic[id]['com4']
    com5 = data_dic[id]['com5']


    return render_template('v_timestamp_new.html', prediction_text=prediction_text[session[user]], art = art, id=id, comments = comments, 
        com1 = com1, com2 = com2, com3 = com3, com4 = com4, com5 = com5 , bleu = bleu[session[user]], one_relevancy_score = one_relevancy_score[session[user]], 

        comm = comm[session[user]]
        ,

        prob=prob[session[user]],org_class = org_class, org_prob = org_prob
        
        )



if __name__ == "__main__":
    app.secret_key = ".."

    app.run(threaded=False)

