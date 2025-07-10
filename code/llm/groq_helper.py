"""
Helper functions to support LLM-based classification
"""
import re


def read_source_articles():
    """
    Read in all soure articles
    :return dict(topic_id:article_text)
    """
    tid2article_file = {'1':'twitter.txt', '2':'lateschool.txt', '3':'climate.txt'}
    tid2article = {}
    for i in ['1', '2', '3']:
        with open('../../resc/SourceArticles/' + tid2article_file[i], 'r') as fi:
            # skip the first 5 lines which are meta info
            data = fi.readlines()[5:]
            # Split the whole text into docs by paragraph breaks / empty lines
            data = ['%%%' if l == '\n' else l for l in data]
            data = [l.replace('\n','') for l in data]
            docs = ' '.join(data).split(' %%% ')
        tid2article[i] = ' '.join(docs)

    return tid2article


def process_demo_essay(essay_data):
    """
    Process a single essay to be used as demonstration in one-shot prompting scenario
    Args:
        essay_data: a single essay as list((sentence, label, tid))
    Returns:
        demo_essay: str Essay as a single string to be incorporated into prompt
        demo_out: str Desired LLM out values as a single string to be incorporated into prompt
    """
    # process example sentences for one-shot-learning
    labs2llm = {
        "info_intro": "_Einleitung",
        "intro_topic": "_Einleitung",
        "info_article": "_Einleitung",
        "info_article_topic": "_Einleitung",
        "article_pro": "_Pro_aus_Lesetext",
        "article_con": "_Con_aus_Lesetext",
        "own": "_Eigen",
        "other": "_Sonstiges",
        "meta": "_Sonstiges",
        "off_topic": "_Sonstiges"
    }
    demo_essay = ""
    demo_out = ""
    for idx in range(len(essay_data)):
        demo_sent = str(idx + 1) + ': ' + essay_data[idx][0] + '\n'
        gold_lab = labs2llm[essay_data[idx][1]]
        demo_lab = str(idx + 1) + ': ' + gold_lab + '\n'
        demo_essay += demo_sent
        demo_out += demo_lab
    return demo_essay, demo_out


def extract_lab_from_llm_out(llm_out):
    """
    Extract content zone labels from the LLM output text
    :param llm_out:str LLM output text for a given essay
    :return: list() extracted list of labels
    """
    llm2labs = {
        "_einleitung": "info_intro",
        "_pro_aus_lesetext": "article_pro",
        "_con_aus_lesetext": "article_con",
        "_eigen": "own",
        "_sonstiges": "other"
    }
    verdicts = llm_out.split('\n')
    # use regex to search for label match in LLM output
    rpattern = r'^\d{1,2}:\s?.*(_Einleitung|_[Pp]ro_aus_[Ll]esetext|_[Cc]on_aus_[Ll]esetext|_[Ee]igen|_[Ss]onstiges)'
    labs = []
    for v in verdicts:
        # disregard possible LLM output lines that do NOT contain label predictions
        if not re.search(rpattern, v):
            continue
        else:
            lab = re.search(rpattern, v).group(1)
            try:
                lab = llm2labs[lab.lower()]
            except KeyError:
                print('Invalid label in LLM output:', lab)
            labs.append(lab)
    return labs


if __name__ == '__main__':
    tid2articles = read_source_articles()
    for i, text in tid2articles.items():
        print(i + '\n' + text)
