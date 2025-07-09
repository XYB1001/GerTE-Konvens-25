"""
Using BERT embeddings in addition to traditional feature-based models
"""
import pickle, os, sys
import statistics
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import KFold

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix, hstack

import torch
from transformers import BertTokenizerFast, BertModel
from sentence_transformers import SentenceTransformer

import feats
import utils
import eval

# set up DEVICE
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using DEVICE:', device)
os.environ['TOKENIZERS_PARALLELISM'] = "False"
BERT_MODEL_NAME = 'bert-base-german-cased'
BERT_MODEL_DIM = 768
SBERT_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

outdir = sys.argv[1]
Path(outdir).mkdir(parents=True, exist_ok=True)
print('Saving model output in', outdir)


def get_bert_embeddings_for_x(x, model, tokenizer, device, model_dim):
    """
    Get frozen BERT embeddings for each sentence in the list
    :param x: list(str), input sentences as sentence list
    :param model: pre-trained BERT model
    :param tokenizer: pre-trained BERT tokenizer
    :param device: DEVICE to compute on
    :return list of BERT embeddings as nd.array with shape (len(x), BERT_dim)
    """
    model.to(device)
    embeds = list()
    for sent in x:
        sent_encoded_simple = tokenizer(sent, truncation=True, padding=False)
        sent_encoded = {}
        # reformat to torch tensors
        for k, v in sent_encoded_simple.items():
            tensor_val = torch.tensor(v)
            # BERT expects encodings to be of shape (batch_size, input_size)
            # Since our input is not in batches, we need to unsqueeze
            if tensor_val.dim() == 1:
                sent_encoded[k] = torch.unsqueeze(tensor_val, 0).to(device)
            else:
                sent_encoded[k] = tensor_val.to(device)

        model.eval()
        with torch.no_grad():
            sent_embedding = model(
                input_ids=sent_encoded['input_ids'],
                token_type_ids=sent_encoded['token_type_ids'],
                attention_mask=sent_encoded['attention_mask']
            ).pooler_output
        embeds.append(sent_embedding.cpu().detach().numpy())

    return np.array(embeds).reshape(-1, model_dim)


def get_padded_sim(sim_scores, max_len):
    """
    Pad sim_scores vector to max_len with 0.0
    :param sim_scores nd.array of shape (1, -1)
    :param max_len int length to which to pad
    :return nd.array of shape (1, max_len)
    """
    # if needed, pad along axis 1 to max_len
    if sim_scores.shape[1] < max_len:
        to_pad = max_len - sim_scores.shape[1]
        sim_scores_padded = np.pad(sim_scores, ((0, 0), (0, to_pad)), mode='constant')
    else:
        sim_scores_padded = sim_scores
    return sim_scores_padded


if __name__ == '__main__':
    # label mapping with only the labels: article_pro, article_con, own, info_article_topic, other
    class_5_mapping = {
        'intro_topic': 'info_intro',
        'info_article': 'info_intro',
        'info_article_topic': 'info_intro',
        'meta': 'other',
        'off_topic': 'other'
    }

    lab2index = {
        'info_intro': 0,
        'article_pro': 1,
        'article_con': 2,
        'own': 3,
        'other': 4,
        #    'off_topic': 5,
        #    'meta': 6
    }
    index2lab = {v: k for k, v in lab2index.items()}

    data = utils.getall_pickled_data('.../data/full_cz_data.p')

    # label mapping
    data = utils.map_labels(data, class_5_mapping, topic_id=True)

    # Setting up frozen BERT model (not fine-tuned!)
    print('Setting up pre-trained, frozen BERT model')
    pretrained_bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
    pretrained_tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    # Setting up SBert model
    print('Setting up SBert model:', SBERT_MODEL_NAME)
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    # Getting SBert embeddings for article texts
    print('Extracting article embeds')
    articles = feats.get_all_articles_unprocessed(split_articles_by='sent', lower=False)
    max_article_len = max([len(texts) for tid, texts in articles.items()])
    tid2article_embeds = dict()
    for tid, texts in articles.items():
        a_embeddings = sbert_model.encode(texts)
        tid2article_embeds[tid] = a_embeddings

    # set up cross-val with KFold
    NUM_FOLDS = 5
    data = np.array(data, dtype=object)
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2025)

    # store eval by fold
    acc_s, prfs_s, prfs_s_byclass = [], [], []

    current_fold = 0
    for train_i, test_i in kf.split(data):
        current_fold += 1
        print('Fold', current_fold)
        train = data[train_i].tolist()
        test = data[test_i].tolist()

        # extraction of position features
        train_position_feats = feats.get_sent_position_in_text(train, option=1)
        test_position_feats = feats.get_sent_position_in_text(test, option=1)
        train_position_feats = csr_matrix(np.array(train_position_feats).reshape(-1, 4))
        test_position_feats = csr_matrix(np.array(test_position_feats).reshape(-1, 4))

        # form X and Y, map Y to indices
        xtrain, ytrain, tidtrain = utils.separate_x_y(train, lab2index=lab2index, topic_id=True)
        xtest, ytest, tidtest = utils.separate_x_y(test, lab2index=lab2index, topic_id=True)

        # Extracting BERT embeddings of sample text
        print('Extracting BERT embeds of training data')
        train_bert_embeds = get_bert_embeddings_for_x(
            x=xtrain,
            model=pretrained_bert_model,
            tokenizer=pretrained_tokenizer,
            device=device,
            model_dim=BERT_MODEL_DIM
        )
        print('Extracting BERT embeds of test data')
        test_bert_embeds = get_bert_embeddings_for_x(
            x=xtest,
            model=pretrained_bert_model,
            tokenizer=pretrained_tokenizer,
            device=device,
            model_dim=BERT_MODEL_DIM
        )
        train_bert_embeds = csr_matrix(train_bert_embeds)
        test_bert_embeds = csr_matrix(test_bert_embeds)

        # Article simil features using SBert
        train_au_feats = []
        for idx in range(len(xtrain)):
            x_embed = sbert_model.encode(xtrain[idx])
            sim = sbert_model.similarity(x_embed, tid2article_embeds[tidtrain[idx]]).numpy()
            train_au_feats.append(get_padded_sim(sim_scores=sim, max_len=max_article_len))
        train_au_feats = csr_matrix(np.vstack(train_au_feats))
        print('(Train) AU feats shape:', train_au_feats.shape)

        test_au_feats = []
        for idx in range(len(xtest)):
            x_embed = sbert_model.encode(xtest[idx])
            sim = sbert_model.similarity(x_embed, tid2article_embeds[tidtest[idx]]).numpy()
            test_au_feats.append(get_padded_sim(sim_scores=sim, max_len=max_article_len))
        test_au_feats = csr_matrix(np.vstack(test_au_feats))
        print('(Test) AU feats shape:', test_au_feats.shape)

        # Setting up feature-based model
        vectorizer = FeatureUnion([
            ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 2))),
            ('char', CountVectorizer(analyzer='char', ngram_range=(3, 4)))
        ])
        # model = SVC(kernel='linear')
        model = RandomForestClassifier()

        # training
        train_in = vectorizer.fit_transform(xtrain)
        # train_in = hstack([train_bert_embeds, train_position_feats])
        # train_in = hstack([train_bert_embeds, train_position_feats, train_au_feats])
        train_in = hstack([train_in, train_bert_embeds, train_position_feats, train_au_feats])
        # train_in = hstack([train_in, train_position_feats, train_au_feats])
        print('Train feature shape:', train_in.shape)
        model.fit(train_in, ytrain)

        # testing
        test_in = vectorizer.transform(xtest)
        # test_in = hstack([test_bert_embeds, test_position_feats])
        # test_in = hstack([test_bert_embeds, test_position_feats, test_au_feats])
        test_in = hstack([test_in, test_bert_embeds, test_position_feats, test_au_feats])
        print('Test feature shape:', test_in.shape)
        ypreds = model.predict(test_in)

        # Save output
        fo = open(outdir + '/fold{}.p'.format(str(current_fold)), 'wb')
        pickle.dump((ypreds, ytest), fo)
        fo.close()

        # Eval
        _, prfs_byclass = eval.evaluate_classification(ypreds, ytest, average=None)
        acc, prfs = eval.evaluate_classification(ypreds, ytest, average='weighted')
        prfs_s_byclass.append(prfs_byclass)
        acc_s.append(acc)
        prfs_s.append(prfs)

    print('Eval across {} folds:'.format(NUM_FOLDS))
    print('Accuracy', round(statistics.mean(acc_s), 3))
    print('Precision, Recall, F1')
    print(eval.print_avg_prfs(prfs_s))
    print('--- By class ---')
    print(eval.print_avg_prfs_by_class(prfs_s_byclass, num_folds=NUM_FOLDS, num_classes=5))
