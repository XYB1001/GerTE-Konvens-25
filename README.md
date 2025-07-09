# GerArgSimilarity

[![DOI](https://zenodo.org/badge/486148603.svg)](https://zenodo.org/badge/latestdoi/486148603)

This is the repo for the dataset of the German argument similarity prediction (GerArgSimilarity) experiment, to be presented at LREC 2022. The accompanying paper, titled *Argument Similarity Assessment in German for Intelligent Tutoring: Crowdsourced Dataset and First Experiments* will be published as part of the Proceedings of LREC 2022.

## About

The dataset consists of 2940 argumentative pairs of text snippets in German which are annotated for argument similarity on a scale from 0  (*no similarity*) to 4 (*high similarity*). The snippets are written based on three openly available news articles, which cover three discussion topics. Please refer to our paper for details. The first snippet in each pair is the candidate snippet, the second the reference snippet.

Each text pair sample has been independently annotated by two annotators using the scores [0,1,2,3,4]. A third annotation which is the average between the two annotations is also provided and is the annotation that has been used in our experiments. The header line refers to the following:

|Item | Description |
| ------------- | ------------- |
| `topic_id`  |  ID of the topic / news article |
| `ref_id` | ID of the reference snippet in the pair |
| `s1`| candidate text snippet |
| `s2`| reference text snippet |
| `anno_1` | first annotation |
| `anno_2` | second annotation |
| `anno_average` | averaged annotation, which ranges from 0 to 4 in steps of 0.5 instead of 1 |


## Some Dataset Statistics

### Topic Distribution

| Topic | Number of pairs |
| ------------- | ------------- |
| 1 | 1164 |
| 2 | 1165 |
| 3 | 611 |

### Similarity score distribution based on `anno_average`

| Score | Number of pairs |
| ------------- | ------------- |
| 0.0 | 1839 |
| 0.5 | 255 |
| 1.0 | 169 |
| 1.5 | 141 |
| 2.0 | 249 |
| 2.5 | 133 |
| 3.0 | 87 |
| 3.5 | 29 |
| 4.0 | 38 |


## License

CC-BY-SA-4.0

## Reference

If you use this dataset please cite the following:
<pre>
@inproceedings{bai2022argument,
  title={Argument Similarity Assessment in German for Intelligent Tutoring: Crowdsourced Dataset and First Experiments},
  author={Bai, Xiaoyu and Stede, Manfred},
  booktitle={Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  pages={2177--2187},
  year={2022}
}
</pre>
