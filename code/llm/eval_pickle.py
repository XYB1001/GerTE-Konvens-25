import pickle
import helper
import sys
import pprint
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class_5_mapping = {
        'intro_topic': 'info_intro',
        'info_article': 'info_intro',
        'info_article_topic': 'info_intro',
        'meta': 'other',
        'off_topic': 'other'
    }

llm2labs = {
        "_einleitung": "info_intro",
        "_pro_aus_lesetext": "article_pro",
        "_con_aus_lesetext": "article_con",
        "_eigen": "own",
        "_sonstiges": "other"
    }


def map_gold_to_5cl(label, mapping):
    """ Map a single label from gold annotation to 5cl label set """
    return mapping[label] if label in mapping else label


def read_pred_gold(llm_out_file, gold_data_file):
    """ Read in pickled files of LLM output and index aligned gold data file """
    fi = open(llm_out_file, 'rb')
    llm_out = pickle.load(fi)
    fi.close()

    fi = open(gold_data_file, 'rb')
    gold_data = pickle.load(fi)
    fi.close()
    return llm_out, gold_data


def show_text_pred_gold(llm_out_file, gold_data_file):
    """ Side-by-side display of text input with gold and llm-predicted labels """
    llm_out, gold_data = read_pred_gold(llm_out_file, gold_data_file)

    assert len(gold_data) == len(llm_out), "Unequal num of total essay samples!"

    for i in range(len(gold_data)):
        essay_data, pred_labels = gold_data[i], helper.extract_lab_from_llm_out(llm_out[i])
        try:
            assert len(essay_data) == len(pred_labels)
        except AssertionError:
            print("Unequal num of labels in essay {}".format(i))
            print('_' * 50 + '\n')
            continue
        for idx in range(len(essay_data)):
            # map gold label to 5 classes
            gold_lab = map_gold_to_5cl(essay_data[idx][1], class_5_mapping)
            # llm_output has already been mapped to 5cl through helper.extract_lab_from_llm_out
            print('\t'.join([essay_data[idx][0], gold_lab, pred_labels[idx]]))
        print('_' * 50 + '\n')
    return


def evaluate_pred_gold(llm_out_file, gold_data_file, average="weighted"):
    """
    Classification evaluation of LLM output
    """
    llm_out, gold_data = read_pred_gold(llm_out_file, gold_data_file)
    assert len(gold_data) == len(llm_out), "Unequal num of total essay samples!"

    model_preds = []
    golds = []
    num_bad_output = 0
    for i in range(len(gold_data)):
        essay_data, pred_labels = gold_data[i], helper.extract_lab_from_llm_out(llm_out[i])
        # if unequal num of labels in essay, i.e. llm-output is bad, skip
        try:
            assert len(essay_data) == len(pred_labels)
        except AssertionError:
            print("Unequal num of labels in essay {}".format(i))
            num_bad_output += 1
            continue
        # map gold labels to 5 classes, llm-out labels have already been mapped by helper function
        gold_labels = [map_gold_to_5cl(lab, class_5_mapping) for text, lab, tid in essay_data]

        model_preds += pred_labels
        golds += gold_labels

    np.set_printoptions(suppress=True)

    # standard classification evaluation
    if average is not None:
        acc, prfs = evaluate_classification(model_preds, golds, average=average)
        # acc, prfs = evaluate_classification(model_preds, golds, average=None)
        print('No. of bad outputs:', num_bad_output)
        print('Accuracy:', acc)
        print('PRFS:')
        print(prfs)
    else:
        acc, prfs = evaluate_classification(model_preds, golds, average=None)
        print('No. of bad outputs:', num_bad_output)
        print('Accuracy:', acc)
        print('PRFS:')
        # pprint.pp(np.round(prfs, decimals=2))
        print(np.round(prfs, decimals=3))
    return


def evaluate_classification(Ypred, Ygold, average='macro'):
    """
    Classification eval on test set
    Ypred, Ygold must be list of the same length
    if averaged: only return accuracy and macro prec/rec/f1/support across all classes
    else: return accuracy and more detailed prec/rec etc. for each class + conf_mat
    """
    acc = accuracy_score(y_true=Ygold, y_pred=Ypred)
    if average is not None:
        prfs = precision_recall_fscore_support(y_true=Ygold, y_pred=Ypred, average=average)
    else:
        prfs = precision_recall_fscore_support(y_true=Ygold, y_pred=Ypred)
    return acc, prfs


def print_avg_prfs(prfs_list):
    """
    print average prec, rec, f1 across k folds
    prfs_list = list(tuple) where each tuple is the prfs output for one fold
    return: [precs, recs, f1s]
    """
    prfs = np.array(prfs_list)
    assert prfs.shape == (len(prfs_list), 4)
    # disregard the last col, i.e. the support values
    prf = prfs[:,:3].astype(float)
    metrics = np.mean(prf, axis=0)
    metrics = np.around(metrics, decimals=3).tolist()
    return metrics


def main():
    raw_out_file = sys.argv[1]
    gold_data_file = sys.argv[2]

    # show_text_pred_gold(llm_out_file=raw_out_file, gold_data_file=gold_data_file)
    evaluate_pred_gold(llm_out_file=raw_out_file, gold_data_file=gold_data_file, average=None)


if __name__ == '__main__':
    main()
