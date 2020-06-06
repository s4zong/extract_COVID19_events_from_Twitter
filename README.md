## Extracting COVID-19 Events from Twitter

This repo contains the annotated corpus and code for paper ``Extracting COVID-19 Events from Twitter".

### Our annotated corpus

*2020.6.4: We will be releasing the annotated corpus within this week.*

#### About our corpus

In this work, we aim at extracting 5 types of events from Twitter: (1) tested positive, (2) tested negative, (3) can not test, (4) death and (5) cure and prevention. The following table provides statistics for our current corpus.

| Event Type 	| # of Annotated Tweets 	| # of Slots
|-----	|-----	|-----|
| Tested positive  |    2500 	|   9  	| 
| Tested negative |    1200  	|   8  	|
| Can not test   	|    1200 	|   5	|
|  Death 	|  1300   	|  6	|
|   Cure & prevention  	|    1300 	|  3	|

#### Corpus format

All annotated tweets are stored in .jsonl file under data folder. Our annotated corpus is released in the following format.


- 'id': It represents tweet id. Due to users' privacy concerns and Twitter's terms of service, we are only able to release the Tweet IDs. There are many tool you could use to download the raw tweet, for example [Twarc](https://github.com/DocNow/twarc).
- 'candidate_chunks_char_offsets': This field contains the character offsets of all the candidate choices we present to crowdsourcing workers during annotation.
- 'annotation': It contains the consensus annotation from 7 annotators. 'part1' and 'part2' denote two steps in our annotation process: (1) specific events identification and (2) slot filling. Please refer to Section 2 of our paper for detailed explaination of our annotation procedure.


### Models Training and Results
To predict the structured information (slots) within a tweet, we setup a binary classification task, where given the tweet `t` and candidate slot `s` the classifier `f` has to predict whether the slot correctly answers the question about the tweet or not `f(t,s) -> 0,1`.
We experiment with Logistic Regression baseline and BERT-based classifier.
- Logistic Regression baseline: masks the candidate slot `s` in the tweet `t` with a special symbol `<Q_TOKEN>` and then makes the binary prediction for each slot filling task using word n-gram features (n = 1,2,3). Model code at `model/logistic_regression_baseline.py`.
- BERT-based classifier: Encloses the candidate slot `s` in the tweet `t` inside special entity markers start and end markers, `<E>` and `</E>` respectively. The BERT hidden representation of the entity start marker `<E>` is used to predict the final label for each task. We also share the BERT model across slot-filling task in each event type (since multiple slots within each event are related to each other). Model code at `model/multitask_bert_entity_classifier.py`.

To recreate all the Logistic Regression experiments results in the paper run `python automate_logistic_regression_baseline_experiments.py`
To recreate all the BERT classifier experiments results in the paper run `python automate_multitask_bert_entity_classifier_experiments.py`
Both `automate_...` scripts will first preprocess the data files, then train the classifiers if they haven't and finally consolidate all the results into a single CSV file.
For Logistic Regression the final results will be saved at `results/all_experiments_lr_baseline_results.tsv` and for BERT classifier the results will be saved at `results/all_experiments_multitask_bert_entity_classifier_fixed_results.tsv`

#### Dependencies and their versions
- `sklearn`
- `scipy==1.4.1`
- `transformers==2.9.0`
- `tqdm`
- `torch==1.5.0`

