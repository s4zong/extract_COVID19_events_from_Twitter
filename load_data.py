
### load data.py

from model.utils import *

downloaded_ones = read_json_line('./data/downloaded_tweets-tagging.jsonl')
downloaded_ones_dict = {}
for each_line in downloaded_ones:
    downloaded_ones_dict[each_line['id_str']] = each_line

file_names = ['positive', 'negative', 'can_not_test', 'death', 'cure_and_prevention']

## generate all files
for each_category in file_names:
    input_file = read_json_line('./data/'+each_category+'.jsonl')
    output_file = []
    for each_line in input_file:
        if each_line['id'] in downloaded_ones_dict:
            each_line['text'] = downloaded_ones_dict[each_line['id']]['full_text']
            each_line['tags'] = downloaded_ones_dict[each_line['id']]['tags']
            output_file.append(each_line)
    print('[I] number of tweets', len(output_file), 'for', each_category)
    write_json_line(output_file, './data/'+each_category+'-add_text.jsonl')