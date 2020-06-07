## Extracting COVID-19 Events from Twitter

This repo contains the annotated corpus and code for paper ``[Extracting COVID-19 Events from Twitter](https://arxiv.org/abs/2006.02567)".

### Our annotated corpus

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

```angular2
{'id': '1238504197319995397',
 'candidate_chunks_offsets':
    [[0, 19], [27, 52], [42, 65], [101, 112],
     [0, 9], [13, 19], [27, 36], [42, 52],
     [56, 65], [89, 91], [96, 112], [117, 121]],
 'annotation':
     {'part1.Response': ['yes'],
      'part2-age.Response': ['Not Specified'],
      'part2-close_contact.Response': ['Not Specified'],
      'part2-employer.Response': ['Not Specified'],
      'part2-gender.Response': ['Not Specified'],
      'part2-name.Response': [[101, 112], [0, 9]],
      'part2-recent_travel.Response': ['Not Specified'],
      'part2-relation.Response': ['Not Specified'],
      'part2-when.Response': [[13, 19]],
      'part2-where.Response': [[56, 65]]}
}
```

- 'id': It contains the tweet id. Due to users' privacy concerns and Twitter's terms of service, we are only able to release tweet ids. We suggest you download tweets from Twitter API, for example using [Tweepy](https://www.tweepy.org/).
- 'candidate_chunks_offsets': This field contains the character offsets for candidate choices we present to crowdsourcing workers during annotation. Please note that there might be slight differences for tweets obtained using different methods, character offsets we provide are calculated based on the 'full_text' field of tweet obtained from Twitter API (in the following way).
```angular2
a_single_tweet = api.get_status(id='id_for_tweet', tweet_mode='extended')
tweet_text_we_use = a_single_tweet['full_text']
```
- 'annotation': It contains our annotation for the tweet. 'part1' and 'part2' denote two steps in our annotation process: (1) specific events identification and (2) slot filling. Please refer to Section 2 of our paper for detailed explanation of our annotation procedure. Also note there might be more than one candidate chunks that could answer a specific question.


### Models Training and Results

*2020.6.7: In our current code release, we calculate character offsets from indices of tokenized chunks, rather than directly using character offsets as inputs. We are still working on changing the input format of our code.*

To predict the structured information (slots) within a tweet, we setup a binary classification task, where given the tweet `t` and candidate slot `s` the classifier `f` has to predict whether the slot correctly answers the question about the tweet or not `f(t,s) -> 0,1`.  <br />
We experiment with Logistic Regression baseline and BERT-based classifier.  <br />
- Logistic Regression baseline: masks the candidate slot `s` in the tweet `t` with a special symbol `<Q_TOKEN>` and then makes the binary prediction for each slot filling task using word n-gram features (n = 1,2,3). Model code at `model/logistic_regression_baseline.py`.
- BERT-based classifier: Encloses the candidate slot `s` in the tweet `t` inside special entity markers start and end markers, `<E>` and `</E>` respectively. The BERT hidden representation of the entity start marker `<E>` is used to predict the final label for each task. We also share the BERT model across slot-filling task in each event type (since multiple slots within each event are related to each other). Model code at `model/multitask_bert_entity_classifier.py`.

To recreate all the Logistic Regression experiments results in the paper run `python automate_logistic_regression_baseline_experiments.py`  <br />
To recreate all the BERT classifier experiments results in the paper run `python automate_multitask_bert_entity_classifier_experiments.py`  <br />
Both `automate_...` scripts will first preprocess the data files, then train the classifiers if they haven't and finally consolidate all the results into a single TSV file. For Logistic Regression the final results will be saved at `results/all_experiments_lr_baseline_results.tsv` and for BERT classifier the results will be saved at `results/all_experiments_multitask_bert_entity_classifier_fixed_results.tsv`  <br />

#### Dependencies and their versions
- `sklearn`
- `scipy==1.4.1`
- `transformers==2.9.0`
- `tqdm`
- `torch==1.5.0`

### Cite
TODO add citation

