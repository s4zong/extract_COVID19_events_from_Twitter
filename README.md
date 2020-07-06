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

- 'id': It contains the tweet id. Due to users' privacy concerns and Twitter's terms of service, we are only able to release tweet ids. We suggest you download tweets from Twitter API, for example using [Tweepy](https://www.tweepy.org/). We also provide a script `download_data.py` for downloading tweets (instructions in `shared_task/README.md`).
- 'candidate_chunks_offsets': This field contains the character offsets for candidate choices we present to crowdsourcing workers during annotation. Please note that there might be slight differences for tweets obtained using different methods, character offsets we provide are calculated based on the 'full_text' field of tweet obtained from Twitter API (in the following way).
```angular2
a_single_tweet = api.get_status(id='id_for_tweet', tweet_mode='extended')
tweet_text_we_use = a_single_tweet['full_text']
```
- 'annotation': It contains our annotation for the tweet. 'part1' and 'part2' denote two steps in our annotation process: (1) specific events identification and (2) slot filling. Please refer to Section 2 of our paper for detailed explanation of our annotation procedure. Also note there might be more than one candidate chunks that could answer a specific question.


### Models Training and Results

*2020.7.6: In our current code release, we calculate character offsets from indices of tokenized chunks, rather than directly using character offsets as inputs. Please expect a full code release this week.*