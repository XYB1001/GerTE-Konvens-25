"""
Using BERT-based neural net with BERT-embeddings concatenated with position features
BERT is fine-tuned in the process
"""
import sys, pickle
import numpy as np
import statistics
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from sklearn.model_selection import KFold

import torch
from transformers import BertTokenizerFast, BertModel

import feats
import utils
import eval

outdir = sys.argv[1]
Path(outdir).mkdir(parents=True, exist_ok=True)
print('Saving model output in', outdir)

# set up DEVICE
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using DEVICE:', device)
BERT_MODEL_NAME = 'bert-base-german-cased'
BERT_MODEL_DIM = 768


class TaskDataset(Dataset):

    def __init__(self, encodings, context_feats, labels):
        self.encodings = encodings
        self.labels = labels
        self.context_feats = context_feats

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['context_feats'] = torch.tensor(self.context_feats[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Define Classification Model
class BERTContextModel(nn.Module):
    """
    BERT pooler output as sentence representation, concatenated with context feats
    """
    def __init__(self, bert_model, bert_out_size, context_size, target_size):
        super(BERTContextModel, self).__init__()
        self.bert_model = bert_model
        self.bert_out_size = bert_out_size
        self.context_size = context_size
        self.target_size = target_size
        # adding a linear layer on top of the representation of BERT pooler + context feats
        self.repre2out = nn.Linear(in_features=self.bert_out_size + self.context_size, out_features=target_size)

    def forward(self, input_ids, token_type_ids, attention_mask, context_feats):
        bert_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_out = bert_out.pooler_output
        repre = torch.cat([bert_out, context_feats], dim=1)
        out = self.repre2out(repre)
        # Apply log softmax. Log needed in conjunction with NLLLOSS loss function
        # Output logged probabs for each label
        log_probabs = F.log_softmax(out, dim=1)
        return log_probabs


# Define training
def train_step(dataloader, model, loss_function, optimiser):
    """
    Model training for one epoch. Model needs to be pushed to DEVICE (global var) already
    """
    model.train()
    total_epoch_loss = 0
    num_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        # clear gradients
        optimiser.zero_grad()
        # push necessary items onto DEVICE
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        context_feats = batch['context_feats'].to(device)
        labels = batch['labels'].to(device)

        # call forward pass
        log_probabs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            context_feats=context_feats
        )
        # get loss
        loss = loss_function(log_probabs, labels)
        total_epoch_loss += loss.item()

        # make train step
        loss.backward()
        optimiser.step()

    print('Train loss after current epoch:', total_epoch_loss / num_batches)


def validate_step(dataloader, model, loss_function):
    """
    Model training for one epoch. Model needs to be pushed to DEVICE (global var) already
    """
    model.eval()
    total_epoch_loss = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # push necessary items onto DEVICE
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            context_feats = batch['context_feats'].to(device)
            labels = batch['labels'].to(device)

            # call forward pass
            log_probabs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                context_feats=context_feats
            )
            # get loss
            loss = loss_function(log_probabs, labels)
            total_epoch_loss += loss.item()

    print('Val loss after current epoch:', total_epoch_loss / num_batches)


def inference_step(dataloader, model):
    """
    Model training for one epoch. Model needs to be pushed to DEVICE (global var) already
    """
    model.eval()
    all_preds, all_targets = [],[]
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # push necessary items onto DEVICE
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            context_feats = batch['context_feats'].to(device)

            # call forward pass
            log_probabs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                context_feats=context_feats
            )
            # Get prediction using argmax
            ypreds = torch.argmax(log_probabs, dim=1)
            all_preds.append(ypreds.cpu().detach().numpy())
            all_targets.append(batch['labels'].numpy())

        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_targets = np.concatenate(all_targets, axis=0).flatten()
        assert len(all_preds) == len(all_targets)

    return all_preds, all_targets


if __name__ == '__main__':
    # label mapping with only the labels: article_pro, article_con, own, info_article_topic, other
    class_5_mapping = {
        'intro_topic': 'info_intro',
        'info_article': 'info_intro',
        'info_article_topic': 'info_intro',
        'meta': 'other',
        'off_topic': 'other'
    }

    class_7_mapping = {
        'info_article_topic': 'info_intro',
        'intro_topic': 'info_intro',
        'info_article': 'info_intro'
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

    # Hyperparams
    CONTEXT_SIZE = 4
    BERT_OUT_SIZE = 768
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 16
    BERT_MODEL_NAME = 'bert-base-german-cased'
    TARGET_SIZE = 5
    NUM_EPOCHS = 4

    print('Current hyper params')
    print('Epochs:', NUM_EPOCHS)
    print('Batch size:', BATCH_SIZE)
    print('Learning rate', LEARNING_RATE)
    # print('Using pre-trained model', BERT_MODEL_NAME)

    data = utils.getall_pickled_data('.../data/full_cz_data.p')

    # label mapping
    data = utils.map_labels(data, class_5_mapping, topic_id=True)

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
        # val_position_feats = feats.get_sent_position_in_text(val)
        test_position_feats = feats.get_sent_position_in_text(test, option=1)
        train_context_labels = train_position_feats
        # val_context_labels = val_position_feats
        test_context_labels = test_position_feats

        # form X and Y, map Y to indices
        xtrain, ytrain, tidtrain = utils.separate_x_y(train, lab2index=lab2index, topic_id=True)
        xtest, ytest, tidtest = utils.separate_x_y(test, lab2index=lab2index, topic_id=True)

        print('len(train):', len(xtrain))
        # print('len(val):', len(xval))
        print('len(test):', len(xtest))

        # Extraction of BERT embedding features
        # re-init for each fold since model gets fine-tuned
        pretrained_bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
        pretrained_tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

        # Instantiate model
        model = BERTContextModel(
            bert_model=pretrained_bert_model,
            bert_out_size=BERT_OUT_SIZE,
            context_size=CONTEXT_SIZE,
            target_size=TARGET_SIZE
        )
        model.to(device)
        print('Model initialised on', device)

        # Set up PyTorch datasets
        Xtrain_encodings = pretrained_tokenizer(xtrain, truncation=True, padding=True)
        # Xval_encodings = pretrained_tokenizer(xval, truncation=True, padding=True)
        Xtest_encodings = pretrained_tokenizer(xtest, truncation=True, padding=True)

        train_data = TaskDataset(Xtrain_encodings, train_context_labels, ytrain)
        # val_data = TaskDataset(Xval_encodings, val_context_labels, yval)
        test_data = TaskDataset(Xtest_encodings, test_context_labels, ytest)

        # dataloaders
        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        # val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

        # set up loss function and optimiser
        loss_f = nn.NLLLoss()
        optimiser = AdamW(model.parameters(), lr=LEARNING_RATE)
        print('Using loss function {}, optimiser {}'.format(loss_f, optimiser))

        # Training
        print('=== Starting Training ===')
        for idx in range(NUM_EPOCHS):
            print('Epoch {}/{}'.format(idx + 1, NUM_EPOCHS))
            train_step(dataloader=train_dataloader, model=model, loss_function=loss_f, optimiser=optimiser)
            # validate_step(dataloader=test_dataloader, model=model, loss_function=loss_f)
        print('=== Training Finished ===')
        print('=== Predicting on Test ===')
        ypreds, ygolds = inference_step(dataloader=test_dataloader, model=model)
        ypreds = ypreds.tolist()
        # ygolds = ygolds.tolist()

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
