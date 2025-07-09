import sys
import os
import random
import pickle
import time
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import groq_helper
import prompt_templates

# Load env variables (.env)
load_dotenv()

# Global Vars
FREE_ACCOUNT = True
GROQ_API_KEY = os.getenv('GROQ_API_KEY_FREE') if FREE_ACCOUNT else os.getenv('GROQ_API_KEY')
# LLM_MODEL_NAME = "llama-3.3-70b-versatile"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
LLM_OUT_NAME = "llama8_oneshot22"
NUM_ESSAYS_UNIT = 5
SLEEP_MIN = 3
print('Experiment params:')
print('Using free account:', FREE_ACCOUNT)
print('LLM model used:', LLM_MODEL_NAME)
print('Saving LLM output to dir:', LLM_OUT_NAME)
print(f'Sleeping for {SLEEP_MIN} minutes every {NUM_ESSAYS_UNIT} essays to avoid reaching rate limit')
print()


def load_data(data_path, random_sample_size=None):
    """ Load data """
    fi = open(data_path, 'rb')
    data = pickle.load(fi)
    fi.close()

    if random_sample_size is not None:
        data = random.sample(data, random_sample_size)
    return data


def process_essays(list_essays):
    """
    Process essay
    """
    essay_texts, labels, tids = [], [], []
    for essay in list_essays:
        # tid is the same for the whole essay
        tid = essay[0][2]
        target_essay = ""
        labs = []
        for i, sent_lab_tid in enumerate(essay):
            sent, lab, _ = sent_lab_tid
            labs.append(lab)
            target_sent = str(i + 1) + ': ' + sent + '\n'
            target_essay += target_sent

        essay_texts.append(target_essay)
        labels.append(labs)
        tids.append(tid)

    return essay_texts, labels, tids


def gen_prompt_zero(target_essay_processed, article=None):
    """
    Generate prompt to feed to LLM in a zero-shot prompt scenario
    """
    # System prompt
    if article is None:
        system_prompt = prompt_templates.system_prompt
    else:
        system_prompt = prompt_templates.system_prompt_article.format(article)

    # User prompt
    article_reference = "einen" if article is None else "den besagten"
    user_p_background = prompt_templates.user_prompt_background.format(article_reference)
    user_p_command = prompt_templates.user_prompt_command.format(target_essay_processed)
    user_prompt = user_p_background + user_p_command
    return system_prompt, user_prompt


def gen_prompt_one(target_essay_processed, demo_essay_processed):
    """
    Generate prompt to feed to LLM in one-shot scenario
    Return:
        system_prompt:str System prompt for the whole chat turn
        user_prompt:str User prompt including the TARGET essay
        user_prompt_demo:str User prompt including the DEMO essay
    """
    # System prompt
    system_prompt = prompt_templates.system_prompt

    # User prompt
    article_reference = "einen"
    user_p_background = prompt_templates.user_prompt_background.format(article_reference)
    user_p_command_target = prompt_templates.user_prompt_command.format(target_essay_processed)
    user_p_command_demo = prompt_templates.user_prompt_command.format(demo_essay_processed)

    user_prompt = user_p_background + user_p_command_target
    user_prompt_demo = user_p_background + user_p_command_demo
    return system_prompt, user_prompt, user_prompt_demo


def get_llm_completion_zero(groq_client, model_name, user_prompt, system_prompt="", temperature=1, max_out_tokens=512):
    """
    Call to Groq API to get model out in zero_setting
    Args
    """
    completion = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=temperature,
        max_completion_tokens=max_out_tokens,
        top_p=1,
        stream=False,
        stop=None,
    )
    out_text = completion.choices[0].message.content
    return out_text


def get_llm_completion_one(groq_client, model_name, user_prompt, user_prompt_demo, assistant_demo_output,
                           system_prompt="", temperature=1, max_out_tokens=512):
    """
    Call to Groq API to get model out in zero_setting
    Args:
        user_prompt:str General task instruction + TARGET essay
        user_prompt_demo:str task instruction + DEMO essay
    """
    completion = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt_demo
            },
            {
                "role": "assistant",
                "content": assistant_demo_output
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=temperature,
        max_completion_tokens=max_out_tokens,
        top_p=1,
        stream=False,
        stop=None,
    )
    out_text = completion.choices[0].message.content
    return out_text


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


def main():
    # load data
    data_path = sys.argv[1]
    # data = load_data(data_path, random_sample_size=5)
    # data = load_data(data_path)
    data = load_data(data_path)
    print('Loading {} samples from {}'.format(len(data), data_path))

    # Demo essay, manually chosen
    demo_id = 22
    demo_essay = data.pop(demo_id)
    print(f'Using essay with index {demo_id} as demonstration. Removing it from dataset to test on.')
    print('Remaining dataset has {} samples'.format(len(data)))
    # process demo essay
    demo_essay_prompt, demo_essay_out = groq_helper.process_demo_essay(demo_essay)

    # process (test) data
    essay_texts, labels, tids = process_essays(data)

    class_5_mapping = {
        'intro_topic': 'info_intro',
        'info_article': 'info_intro',
        'info_article_topic': 'info_intro',
        'meta': 'other',
        'off_topic': 'other'
    }

    # get source articles
    # tid2article_text = groq_helper.read_source_articles()

    # set up Groq client
    client = Groq(api_key=GROQ_API_KEY)
    print('Groq client set up')

    # iteratively prompt LLM with each essay from (test) data:
    llm_raw_outputs, llm_bad_outputs = [], []
    model_preds, golds = [], []
    num_bad_output = 0
    process_count = 1
    for idx in range(len(essay_texts)):
        # print('Entering processing')
        if process_count % NUM_ESSAYS_UNIT == 0:
            print('Finished prompting {} essays'.format(process_count))
            # sleep for ... min to prevent rate limit error
            print(f'Sleeping for {SLEEP_MIN} min to prevent rate limit error')
            time.sleep(SLEEP_MIN * 60)
        essay = essay_texts[idx]
        gold_labs = labels[idx]
        # map labels to 5 classes
        gold_labs = [class_5_mapping[lab] if lab in class_5_mapping else lab for lab in gold_labs]
        # tid = tids[idx]

        # get prompt
        # sys_prompt, usr_prompt = gen_prompt_zero(target_essay_processed=essay, article=tid2article_text[tid])
        sys_prompt, usr_prompt, usr_prompt_demo = gen_prompt_one(
            target_essay_processed=essay,
            demo_essay_processed=demo_essay_prompt
        )

        # prompt LLM - zero shot
        # llm_outtext = get_llm_completion_zero(
        #     groq_client=client,
        #     model_name=LLM_MODEL_NAME,
        #     system_prompt=sys_prompt,
        #     user_prompt=usr_prompt
        # )

        # prompt LLM - one shot
        llm_outtext = get_llm_completion_one(
            groq_client=client,
            model_name=LLM_MODEL_NAME,
            user_prompt=usr_prompt,
            user_prompt_demo=usr_prompt_demo,
            assistant_demo_output=demo_essay_out
        )

        # parse output
        llm_raw_outputs.append(llm_outtext)
        pred_essay_labs = groq_helper.extract_lab_from_llm_out(llm_out=llm_outtext)
        if len(gold_labs) != len(pred_essay_labs):
            print('Bad output found')
            num_bad_output += 1
            llm_bad_outputs.append([essay, llm_outtext])
        else:
            model_preds += pred_essay_labs
            golds += gold_labs

        # increase counter
        process_count += 1

    # standard classification evaluation
    acc, prfs = evaluate_classification(model_preds, golds, average="weighted")
    print('Accuracy:', acc)
    print('PRFS:')
    print(prfs)

    # store LLM out
    fo = open(f'out_tmp/raw/{LLM_OUT_NAME}.p', 'wb')
    pickle.dump(llm_raw_outputs, fo)
    fo.close()
    print('Created raw out file')

    fo = open(f'out_tmp/bad/{LLM_OUT_NAME}.p', 'wb')
    pickle.dump(llm_bad_outputs, fo)
    fo.close()
    print('Created bad out file')


if __name__ == '__main__':
    main()
