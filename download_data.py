
## download tweets from Twitter API

import argparse
import tweepy
import glob
import json


parser = argparse.ArgumentParser()
parser.add_argument("--API_key", help="Your API key", type=str,
                    required=True)
parser.add_argument("--API_secret_key", help="your API secret key", type=str, required=True)
parser.add_argument("--access_token", help="your Access token", type=str,
                    required=True)
parser.add_argument("--access_token_secret", help="your Access secret token",
                    type=str, required=True)
args = parser.parse_args()


def read_json_line(path):
    output = []
    with open(path, 'r') as f:
        for line in f:
            output.append(json.loads(line))
    return output

def write_json_line(data, path):

    with open(path, 'w') as f:
        for i in data:
            f.write("%s\n" % json.dumps(i))

    return None

def acquire_from_twitter_api(input_data):

    ## tweepy
    auth = tweepy.OAuthHandler(args.API_key, args.API_secret_key)
    auth.set_access_token(args.access_token, args.access_token_secret)

    api = tweepy.API(auth, parser=tweepy.parsers.JSONParser(), wait_on_rate_limit=True)

    tweets_by_API = []
    wrong_ones = []
    for idx, i in enumerate(input_data):
        if idx % 500 == 0:
            print('[I] number of ids processed:', idx)
        try:
            tweets_by_API.append(api.get_status(i['id'], tweet_mode='extended'))
        except tweepy.TweepError as e:
            wrong_ones.append([i, e])

    return tweets_by_API, wrong_ones


if __name__ == '__main__':

    ## read all files and get tweet ids
    all_files = glob.glob('./data/*.jsonl')
    # print(all_files)
    all_ids = [j for i in all_files for j in read_json_line(i)]

    ## get from twitter api
    print('===== download starts =====')
    all_returned_tweets, not_collect_ones = acquire_from_twitter_api(all_ids)
    print('[I] number of tweets collected:', len(all_returned_tweets))
    print('===== download finishes =====')

    ## store file
    write_json_line(all_returned_tweets, './data/downloaded_tweets.jsonl')