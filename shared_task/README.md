
# Baseline Model and Evaluations for W-NUT 2020 Shared Task

Check the shared task official website at: [http://noisy-text.github.io/2020/extract_covid19_event-shared_task.html](http://noisy-text.github.io/2020/extract_covid19_event-shared_task.html).

**Note: We also summarize all questions with corresponding candidate choices [here](https://docs.google.com/document/d/1OWFTXOZpoXNrDULq6PFXvIGarSZwpU-uLQRuV4wrJwI/edit?usp=sharing).**

## Data Preparation

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

## Run baseline

We provide a logistic regression baseline for our task. You could directly run this baseline by:

```angular2
python automate_logistic_regression_baseline_experiments.py
```

## Evaluation Period (Sep. 7, 2020 - Sep. 11, 2020)

**Deadline: Sep. 11, 2020 11:59pm AoE (https://www.timeanddate.com/time/zones/aoe)**

We will only evaluate the system outputs for slot filling questions (part2.XXX.Response). Please read the following instructions carefully for how to correctly format your system outputs (Basically we ask you to follow the same format as training data to organize your prediction outputs).

Please see e-mail announcements (Dropbox link for the test data has been sent on Sep. 6, 2020) for downloading the data. We prepare 500 tweets for each event category.

### Results submission link

Please upload your results at https://forms.gle/8tDfzMQp7mxQmFvk7. We will only allow ONE submission for each team. If multiple runs submitted by accident, please email us at wnut.sharedtask.covid19extract@gmail.com to specify which one run you want us to include in the official evaluation. 

### Prediction format for each tweet

Specifically, for each tweet:

1. It should contain keys "id" and "predicted_annotation", and "predicted_annotation" contains your prediction results.
2. Your prediction results shall be stored within a list for each slot.
3. Do NOT use character offsets, directly extract the corresponding text from tweet contents.

A sample output should look like:

```angular2
{'id': '1238504197319995397',
 'predicted_annotation': {'part2-age.Response': ['Not Specified'],
                          'part2-close_contact.Response': ['Not Specified'],
                          'part2-employer.Response': ['Not Specified'],
                          'part2-gender.Response': ['Not Specified'],
                          'part2-name.Response': ['Rita Wilson', 'Tom Hanks'],
                          'part2-recent_travel.Response': ['Not Specified'],
                          'part2-relation.Response': ['Not Specified'],
                          'part2-when.Response': ['Friday'],
                          'part2-where.Response': ['Australia']}}
```

**PLEASE READ:**

0. It doesn't matter if your predictions are lowercased or uppercased.
1. For `name` slot and `close_contact` slot for ALL event types, you shall replace "I" (or any variations of "I", for example "I'm") with "AUTHOR OF THE TWEET". In other words, only use "AUTHOR OF THE TWEET" if it is the answer for the `name` slot.
2. For `opinion` slot in cure & prevention category, you shall only have two labels: "effective" and "not_effective" ("no_opinion", "no_cure" and "not_effective" will be merged into "not_effective").
3. For `relation` slot and `symptoms` slot, you shall only have two labels: "yes" and "not specified" ("no" and "not specified" will be merged into "not specified").
4. The following slots will be excluded in the final evaluation, as too few annotations are collected in the test set.

- Tested Positive: No slots will be excluded
- Tested Negative: "how long" slots will be excluded
- Can Not Test: No slots will be excluded
- Death: "symptom" slot will be excluded
- Cure: No slots will be excluded

### Prediction format for each category

You are required to generate a SEPARATE .jsonl file for each category. Please name your file by using `TEAM_NAME-CATEGORL_NAME.jsonl` (for `CAGEGORY_NAME` please use one of the following: "positive", "negative", "can_not_test", "death" and "cure").

For example, if the team name is OSU_NLP, then the submission folder shall be organized as follows:

```angular2
OSU_NLP/
     ├─ OSU_NLP-positive.jsonl
     ├─ OSU_NLP-negative.jsonl
     ├─ OSU_NLP-can_not_test.jsonl
     ├─ OSU_NLP-death.jsonl
     ├─ OSU_NLP-cure.jsonl
```

Once you have files for all 5 categories, pack all model predictions in a SINGLE .zip file. The name of your .zip file should be `TEAM_NAME.zip`. In above case, it shall be `OSU_NLP.zip`.

We provide a format checker `format_checker.py` to help you check if your predictions have met our format requirements. You could run our format checker script by

```angular2
python format_checker.py -f PATH_TO_YOUR_PREDICTION_FOLDER/
```

### Evaluation script

We will use `eval.py` to evaluate the system performance. System predictions will be compared against the golden annotations. Evaluations will only be done for chunks other than "Not Specified". 

```angular2
python evaluation.py -p PATH_TO_YOUR_PREDICTION.jsonl
                     -g PATH_TO_GOLDEN_ANNOS.jsonl
```

The golden annotation file will follow the same format as specified above. Instead of having `predicted_annotation` field, it will have a `golden_annotation` field.
