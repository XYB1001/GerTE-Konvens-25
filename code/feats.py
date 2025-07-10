"""
Functions related to the extraction of manual features
"""

# import spacy

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

import utils


# locations of source article files
climate = '../resc/SourceArticles/climate.txt'
lateschool = '../resc/SourceArticles/lateschool.txt'
twitter = '../resc/SourceArticles/twitter.txt'


def get_sent_position_in_text(data, option=1):
    """
    Looping through set of texts
    For each sent in text, extract its relative position in the text in terms of which quarter it occurs in

    :param data list(list(sentence, cz_label))
    :param option ways of capturing position. See code
    :return list(list) for each sent return the one-hot-encoding of the sentence's position like [0,1,0,0]
    """

    positions = []
    for text in data:
        total = len(text)
        split1, split2, split3 = int(0.25 * total), int(0.5 * total), int(0.75 * total)
        first = text[:split1]
        second = text[split1:split2]
        third = text[split2:split3]
        fourth = text[split3:]
        if option == 1:
            p1 = [[1, 0, 0, 0] for _ in range(len(first))]
            p2 = [[0, 1, 0, 0] for _ in range(len(second))]
            p3 = [[0, 0, 1, 0] for _ in range(len(third))]
            p4 = [[0, 0, 0, 1] for _ in range(len(fourth))]
        else:
            p1 = [1 for _ in range(len(first))]
            p2 = [2 for _ in range(len(second))]
            p3 = [3 for _ in range(len(third))]
            p4 = [4 for _ in range(len(fourth))]
        positions = positions + p1 + p2 + p3 + p4
    return positions


# def spacy_process(essay, lemma=False, exclude_stopwords=True):
#     """
#     Using Spacy to process an essay
#     (deprecated...)
#     """
#     nlp_de = spacy.load("de_core_news_sm", disable=["tagger", "attribute_ruler", "parser", "ner"])
#     stop_words = stopwords.words('german')
#     processed = []
#     for sample in essay:
#         mod_tokens = []
#         for tok in nlp_de(sample.lower()):
#             if exclude_stopwords:
#                 if tok.text in stop_words:
#                     continue
#             tok_mod = tok.lemma_ if lemma else tok.text
#             mod_tokens.append(tok_mod)
#         processed.append(' '.join(mod_tokens))
#     return processed


####### Concerning source articles #############

def read_source_article(filepath, split_by='para', lower=False, remove_punc=False):
    """
    Read in each soure article and turn it into a sequence of paragraphs/ sentences
    :param split_by options 'para' for paragraph or 'sent' for sentences or 'no_split' for leaving article
    as a single string
    """
    with open(filepath, 'r') as fi:
        # skip the first 5 lines which are meta info
        data = fi.readlines()[5:]
        # Split the whole text into docs by paragraph breaks / empty lines
        data = ['%%%' if l == '\n' else l for l in data]
        data = [l.replace('\n','') for l in data]
        docs = ' '.join(data).split(' %%% ')
        if split_by == 'sent':
            sent_docs = []
            for d in docs:
                sents = sent_tokenize(d, language="german")
                if remove_punc:
                    for s in sents:
                        sent_docs.append(utils.remove_punctuation(s, lower=lower))
                else:
                    sent_docs += sents
            docs = sent_docs
        if split_by == 'no_split':
            if remove_punc:
                docs = [utils.remove_punctuation(d, lower=lower) for d in docs]
            docs = ' '.join(docs)
        else:
            if remove_punc:
                docs = [utils.remove_punctuation(d, lower=lower) for d in docs]

        return docs


def get_all_articles_unprocessed(split_articles_by='para', lower=True, remove_punc=False):
    """
    Read in all articles, store in dictionary form
    :param split_articles_by 'para' or 'sent'
    return: {article_number: list(str))}
    """
    articles = dict()
    articles['1'] = read_source_article(twitter, split_by=split_articles_by, lower=lower, remove_punc=remove_punc)
    articles['2'] = read_source_article(lateschool, split_by=split_articles_by, lower=lower, remove_punc=remove_punc)
    articles['3'] = read_source_article(climate, split_by=split_articles_by, lower=lower, remove_punc=remove_punc)
    return articles


if __name__ == '__main__':
    docs = read_source_article(lateschool, split_by='no_split', lower=False)
    print(docs)
