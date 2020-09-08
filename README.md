## Extracting COVID-19 Events from Twitter

This repo contains the annotated corpus and code for paper ``[Extracting COVID-19 Events from Twitter](https://arxiv.org/abs/2006.02567)".

```
@misc{zong2020extracting,
    title={Extracting COVID-19 Events from Twitter},
    author={Shi Zong and Ashutosh Baheti and Wei Xu and Alan Ritter},
    year={2020},
    eprint={2006.02567},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### Shared task

We are organizing a shared task on COVID-19 event extraction from Twitter by using our annotated corpus at [EMNLP 2020 Workshop on User-generated Text](http://noisy-text.github.io/2020/extract_covid19_event-shared_task.html). The system description papers will be peer-reviews and published as part of the EMNLP 2020 Workshop Proceedings (ACL Anthology).

Check `shared_task` folder for the provided baseline models and evaluation scripts for the shared task.

If interested in participating, please [register](https://forms.gle/zUvxLoDohbDmBzuh6) here. For any questions related to the shared task, please contact us at [wnut.sharedtask.covid19extract@gmail.com](mailto:wnut.sharedtask.covid19extract@gmail.com).

2020-09-06: We have started our evaluation period. Check detailed information at https://github.com/viczong/extract_COVID19_events_from_Twitter/tree/master/shared_task#evaluation-period-sep-7-2020---sep-11-2020.

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

All annotated tweets are stored in .jsonl file under `data` folder. Our annotated corpus is released in the following format.

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

- 'id': It contains the tweet id. Due to users' privacy concerns and Twitter's terms of service, we are only able to release tweet ids.
- 'candidate_chunks_offsets': This field contains the character offsets for candidate choices we present to crowdsourcing workers during annotation. Please note that there might be slight differences for tweets obtained using different methods, character offsets we provide are calculated based on the 'full_text' field of tweet obtained from Twitter API (in the following way).
```angular2
a_single_tweet = api.get_status(id='id_for_tweet', tweet_mode='extended')
tweet_text_we_use = a_single_tweet['full_text']
```
- 'annotation': It contains our annotation for the tweet. 'part1' and 'part2' denote two steps in our annotation process: (1) specific events identification and (2) slot filling. Please refer to Section 2 of our paper for detailed explanation of our annotation procedure. Also note there might be more than one candidate chunks that could answer a specific question.

### Download tweets and preprocessing

#### Download tweets

We provide a script to download tweets by using tweepy. Prepare your Twitter API keys and tokens, and then run

```angular2
python download_data.py --API_key your_API_key
                        --API_secret_key your_API_secret_key
                        --access_token your_access_token
                        --access_token_secret your_access_token_secret 
```

Please allow the script to run for a while. The downloaded tweets will be under `data` folder, named `downloaded_tweets.jsonl`.

#### Tweets parsing and pre-processing

We use [Twitter tagging tool](https://github.com/aritter/twitter_nlp) for tokenization.

We suggest using tagging tool in following way, which reads in json line format files and directly appends 'tags' field into the original file. Please make sure there is a 'text' field for each line (we have already added this field if you use our `download_data.py` script). Please use `python2` to run this tagging tool.

```angular2
cat PATH_TO_downloaded_tweets.jsonl | python2 python/ner/extractEntities2_json.py --pos --chunk
                                    > PATH_TO_downloaded_tweets-tagging.jsonl
```

Once you get the tagging file, store it under `data` folder, named `downloaded_tweets-tagging.jsonl`. Then run the following command

```angular2
python load_data.py
```

This script will add tweet text and tags into original annotations.

### Models training and results

To predict the structured information (slots) within a tweet, we setup a binary classification task, where given the tweet `t` and candidate slot `s` the classifier `f` has to predict whether the slot correctly answers the question about the tweet or not `f(t,s) -> 0,1`.  <br />
We experiment with Logistic Regression baseline and BERT-based classifier.  <br />
- Logistic Regression baseline: masks the candidate slot `s` in the tweet `t` with a special symbol `<Q_TOKEN>` and then makes the binary prediction for each slot filling task using word n-gram features (n = 1,2,3). Model code at `model/logistic_regression_baseline.py`.
- BERT-based classifier: Encloses the candidate slot `s` in the tweet `t` inside special entity markers start and end markers, `<E>` and `</E>` respectively. The BERT hidden representation of the entity start marker `<E>` is used to predict the final label for each task. We also share the BERT model across slot-filling task in each event type (since multiple slots within each event are related to each other). Model code at `model/multitask_bert_entity_classifier.py`.

To recreate all the Logistic Regression experiments results in the paper run 
```angular2
python automate_logistic_regression_baseline_experiments.py
```

To recreate all the BERT classifier experiments results in the paper run
```angular2
python automate_multitask_bert_entity_classifier_experiments.py
```

Both `automate_...` scripts will first preprocess the data files, then train the classifiers if they haven't and finally consolidate all the results into a single TSV file. For Logistic Regression the final results will be saved at `results/all_experiments_lr_baseline_results.tsv` and for BERT classifier the results will be saved at `results/all_experiments_multitask_bert_entity_classifier_fixed_results.tsv`  <br />

#### Dependencies and their versions
- `sklearn`
- `scipy==1.4.1`
- `transformers==2.9.0`
- `tqdm`
- `torch==1.5.0`
