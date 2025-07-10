# GerTE (KONVENS 2025)


This is the repo for the code and dataset of the content zone prediction experiment on German source-dependent essays (<ins>T</ins>extgebundene <ins>E</ins>rörterung) (GerTE) experiment, to be presented at KONVENS 2025. The accompanying paper, titled *Predicting Functional Content Zones in German Source-Dependent Argumentative Essays: Experiments on a Novel Dataset* will be published as part of the Proceedings of KONVENS 2025.

## About

The dataset consists of 117 short argumentative essays written in response to 3 different news articles. The essays are therefore source-dependent, and each essay deals with the discussion topic of one article. Each essay has been segmented into sentences, and each sentence is labelled with exactly 1 out of 7 possible content zone labels. Details on the data collection and annotation process are provided in the paper.

The sentence-segmented and content-zone labelled essays are provided in `data/gerte_full.tsv`. It is a tab-separated file with the following header lines:

|Item | Description |
| ------------- | ------------- |
| `essay_id`  |  ID of the essay |
| `sent_id` | ID of the sentence within the essay |
| `sent_text`| Sentence text |
| `sent_label`| Content zone label for the sentence |
| `topic_id` | ID of the topic that the essay deals with |

The three topics and their IDs are as follows. The respective news articles that served as source text are linked from the discussion topcis:

| Topic ID | Discussion Topic | 
| ------------- | ------------- |
| 1 | [Should social media like Twitter/X be integrated into schools for learning?](https://www.zeit.de/digital/internet/2011-06/twitter-unterricht/komplettansicht) |
| 2 | [Should school start later in the morning than 8 am?](https://www.aerztezeitung.de/Panorama/Ist-es-vernuenftig-die-Schule-um-8-zu-beginnen-402238.html) |
| 3 | [Should climate change be taught in a school subject of its own?](https://www.zeit.de/gesellschaft/schule/2020-01/klimawandel-schulfach-bildung-unterricht-konkurrenz) |

## Some Dataset Statistics

### Topic Distribution

| Topic | Number of Essays |
| ------------- | ------------- |
| 1 | 50 |
| 2 | 50 |
| 3 | 17 |

### Content zone distribution based on 7 classes

| Label | Number of sentence instances |
| ------------- | ------------- |
| own | 551 |
| article_pro | 460 |
| info_intro | 281 |
| article_con | 243 |
| off_topic | 77 |
| other | 71 |
| meta | 30 |


## License

TBD

## Reference

TBD
