"""
Using traditional feature-based models
"""
import pickle, sys
import statistics
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix, hstack

from pathlib import Path

from sklearn.model_selection import KFold

import feats
import utils
import eval

outdir = sys.argv[1]
Path(outdir).mkdir(parents=True, exist_ok=True)
print('Saving model output in', outdir)


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

    lab2index ={
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

        # count label distribution
        print('Train:', utils.count_labels(ytrain))
        print('Test:', utils.count_labels(ytest))

        # training
        vectorizer = FeatureUnion([
             ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 2))),
             ('char', CountVectorizer(analyzer='char', ngram_range=(3, 4)))
        ])
        # vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        # model = SVC(kernel='linear')
        # model = SVC()
        model = RandomForestClassifier()
        # model = LogisticRegression()
        print('Model:', model)

        train_in = vectorizer.fit_transform(xtrain)
        train_in = hstack([train_in, train_position_feats])
        print('Train feat size:', train_in.shape)
        model.fit(train_in, ytrain)

        # testing
        test_in = vectorizer.transform(xtest)
        test_in = hstack([test_in, test_position_feats])
        print('Test feat size:', test_in.shape)
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
    # labels = [v for v in index2lab.values()]
    # print(labels)
    # confmat = confusion_matrix(ytest, ypreds)
    # print(confmat)
    # # print()
    # # eval.print_errors(xtest, ypreds, ytest, index2lab)








