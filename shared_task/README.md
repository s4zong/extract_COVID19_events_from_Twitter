
# Baseline Model and Evaluations for W-NUT 2020 Shared Task

Check the shared task official website at: [http://noisy-text.github.io/2020/extract_covid19_event-shared_task.html](http://noisy-text.github.io/2020/extract_covid19_event-shared_task.html).

**Note: We also summarize all questions with corresponding candidate choices [here](https://docs.google.com/document/d/1OWFTXOZpoXNrDULq6PFXvIGarSZwpU-uLQRuV4wrJwI/edit?usp=sharing).**

### Download tweets

We provide a script to download tweets by using tweepy. Prepare your Twitter API keys and tokens, and then under `extract_COVID19_events_from_Twitter` folder run

```angular2
python download_data.py --API_key your_API_key
                        --API_secret_key your_API_secret_key
                        --access_token your_access_token
                        --access_token_secret your_access_token_secret 
```

Please allow the script to run for a while. The downloaded tweets will be under `data` folder, named `downloaded_tweets.jsonl`.

### Tweets parsing and pre-processing

We use [Twitter tagging tool](https://github.com/aritter/twitter_nlp) for tokenization.

We suggest using tagging tool in following way, which reads in json line format files and directly appends 'tags' field into the original file. Please make sure there is a 'text' field for each line (we have already added this field if you use our `download_data.py` script). Please use `python2` to run this tagging tool.

```angular2
cat PATH_TO_downloaded_tweets.jsonl | python2 python/ner/extractEntities2_json.py --pos --chunk 
                                    > PATH_TO_downloaded_tweets-tagging.jsonl
```

Once you get the tagging results, store it under `data` folder, named `downloaded_tweets-tagging.jsonl`. Then run the following command

```angular2
python load_data.py
```

This script will add tweet text and tags into original annotations.

### Run baseline

We provide a logistic regression baseline for our task. You could directly run this baseline by:

```angular2
python automate_logistic_regression_baseline_experiments.py
```

### Evaluation

We will only evaluate the system outputs for slot filling questions (part2.xxx.response). System outputs should be organized in the following format.

```angular2
[{'id': '1238504197319995397',
  'predicted_annotation': {'part2-age.Response': ['Not Specified'],
                           'part2-close_contact.Response': ['Not Specified'],
                           'part2-employer.Response': ['Not Specified'],
                           'part2-gender.Response': ['Not Specified'],
                           'part2-name.Response': ['Rita Wilson', 'Tom Hanks'],
                           'part2-recent_travel.Response': ['Not Specified'],
                           'part2-relation.Response': ['Not Specified'],
                           'part2-when.Response': ['Friday'],
                           'part2-where.Response': ['Australia']}}]
```

We will use `eval.py` to evaluate the system performance. System predictions will be compared against the golden annotations. Evaluations will only be done for chunks other than "Not Specified". 

```angular2
python evaluation.py -p PATH_TO_YOUR_PREDICTION.jsonl
                     -g PATH_TO_GOLDEN_ANNOS.jsonl
```

The golden annotation file will follow the same format as specified above. Instead of having `predicted_annotation` field, it will have a `golden_annotation` field.


