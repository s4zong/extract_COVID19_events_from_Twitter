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