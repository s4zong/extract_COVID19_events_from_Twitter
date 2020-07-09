# We will read the jsonl data file and convert it into a format that can be used by the logistic regression classifier
# jsonl file is basically json object on each line

import argparse
import os
import re
import json
import pickle
import logging
from utils import log_list, print_list, save_in_pickle

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

Q_TOKEN = "<Q_TARGET>"
URL_TOKEN = "<URL>"
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_file", help="Path to the data file", type=str, required=True)
parser.add_argument("-s", "--save_file", help="Path to the Instances, Statistics and Header pickle save file", type=str, required=True)
args = parser.parse_args()

def writeJSONLine(data, path):

    with open(path, 'w') as f:
        for i in data:
            f.write("%s\n" % json.dumps(i))

    return None

def read_jsonl_datafile(data_file):
	data_instances = []
	with open(data_file, "r") as reader:
		for line in reader:
			line = line.strip()
			if line:
				data_instances.append(json.loads(line))
	return data_instances

def get_label_for_key_from_annotation(key, annotation, candidate_chunk):
	tagged_chunks = annotation[key]
	label = 0
	if tagged_chunks:
		# if key is "name", "who_cure", and "I" is a gold chunk then add "AUTHOR OF THE TWEET" as a gold chunk
		if key in ["name", "who_cure", "close_contact", "opinion"] and ("I" in tagged_chunks or "i" in tagged_chunks):
			tagged_chunks.append("AUTHOR OF THE TWEET")

		for tagged_chunk in tagged_chunks:
			if tagged_chunk == candidate_chunk:
				label = 1
				break
	return label, tagged_chunks

def get_tagged_label_for_key_from_annotation(key, annotation):
	tagged_chunks = annotation[key]
	return tagged_chunks

def get_label_from_tagged_label(tagged_label):
	if tagged_label == "Not Specified":
		return 0
	elif tagged_label == "Yes":
		return 1
	elif tagged_label == "Male":
		return 1
	elif tagged_label == "Female":
		return 1
	elif tagged_label.startswith("no_cure"):
		return 0
	elif tagged_label.startswith("not_effective"):
		return 0
	elif tagged_label.startswith("no_opinion"):
		return 0
	elif tagged_label.startswith("effective"):
		return 1
	else:
		print(f"Unknown tagged_label {tagged_label}")
		exit()

def find_text_to_tweet_tokens_mapping(text, tweet_tokens):
	# NOTE: tweet_tokens is a list of strings where each element is a token
	current_tok = 0
	current_tok_c_pos = 0
	n_toks = len(tweet_tokens)
	tweet_toks_c_mapping = [list()]
	for c_pos, c in enumerate(text):
		if c.isspace():
			# Just ignore
			continue
		if current_tok_c_pos == len(tweet_tokens[current_tok]):
			# Change current tok and reset c_pos
			current_tok += 1
			current_tok_c_pos = 0
			tweet_toks_c_mapping.append(list())
		# print(text)
		# print(tweet_tokens)
		# print(c_pos, f";{c};")
		# print(current_tok, current_tok_c_pos, tweet_tokens[current_tok][current_tok_c_pos])
		if c == tweet_tokens[current_tok][current_tok_c_pos]:
			# Add mapping
			tweet_toks_c_mapping[current_tok].append(c_pos)
			current_tok_c_pos += 1
		else:
			# Something wrong. This shouldn't happen
			print("Wrong mapping:")
			print(text)
			print(tweet_tokens)
			print(c_pos, f"{text[c_pos-1]};{c};{text[c_pos+1]}")
			print(current_tok, current_tok_c_pos, f";{tweet_tokens[current_tok][current_tok_c_pos]};")
			exit()

	# Check if reached end
	assert len(tweet_tokens)-1 == current_tok and len(tweet_tokens[current_tok]) == current_tok_c_pos
	return tweet_toks_c_mapping

def get_tweet_tokens_from_tags(tags):
	tokens = [e.rsplit("/", 3)[0] for e in tags.split()]
	return ' '.join(tokens)

def make_instances_from_dataset(dataset):
	# Create instances for all each task.
	# we will store instances for each task separately in a dictionary
	task_instances_dict = dict()

	# All the questions with interesting annotations start with prefix "part2-" and end with suffix ".Response"
	# Extract all the interesting questions' annotation keys and their corresponding question-tags
	question_keys_and_tags = list()		# list of tuples of the format (<tag>, <dict-key>)
	# Extract the keys and tags from first annotation in the dataset
	dummy_annotation = dataset[0]['annotation']
	for key in dummy_annotation.keys():
		if key.startswith("part2-") and key.endswith(".Response"):
			question_tag = key.replace("part2-", "").replace(".Response", "")
			question_keys_and_tags.append((question_tag, key))
	# Sort the question keys to have a fixed ordering
	question_keys_and_tags.sort(key=lambda tup: tup[0])
	# print(question_keys_and_tags)
	# exit()
	question_tags = [question_tag for question_tag, question_key in question_keys_and_tags]
	question_keys = [question_key for question_tag, question_key in question_keys_and_tags]
	if "gender" in question_tags:
		# Update the question_keys_and_tags
		gender_index = question_tags.index("gender")
		question_tags[gender_index] = "gender_female"
		question_tags.insert(gender_index, "gender_male")
		question_keys.insert(gender_index, question_keys[gender_index])
		question_keys_and_tags = list(zip(question_tags, question_keys))

	task_instances_dict= {question_tag: list() for question_tag, question_key in question_keys_and_tags}
	# Dictionary to store total statistics of labels for each question_tag
	gold_labels_stats = {question_tag: dict() for question_tag, question_key in question_keys_and_tags}
	# Dictionary to store unique tweets for each gold tag within each question tag
	gold_labels_unique_tweets = {question_tag: dict() for question_tag, question_key in question_keys_and_tags}
	skipped_chunks = 0
	ignore_ones = []
	for annotated_data in dataset:
		# We will take one annotation and generate who instances based on the chunks
		id = annotated_data['id']
		annotation = annotated_data['annotation']
		text = annotated_data['text'].strip()
		candidate_chunks_offsets = annotated_data['candidate_chunks_offsets']
		candidate_chunks_from_text = [text[c[0]:c[1]] for c in candidate_chunks_offsets]
		# print(annotated_data)
		tags = annotated_data['tags']
		tweet_tokens = get_tweet_tokens_from_tags(tags)
		# logging.info(f"Text:{text}")
		# logging.info(f"Tokenized text:{tweet_tokens}")
		# Checking if the tokenized tweet is equal to the tweet without spaces
		try:
			assert re.sub("\s+", "", text) == re.sub("\s+", "", tweet_tokens)
		except AssertionError:
			logging.error(f"Tweet and tokenized tweets don't match in id:{id}")
			text_without_spaces = re.sub('\s+', '', text)
			logging.error(f"Tweets without spaces: {text_without_spaces}")
			tweet_tokens_without_spaces = re.sub('\s+', '', tweet_tokens)
			logging.error(f"Tokens without spaces: {tweet_tokens_without_spaces}")
			exit()
		tweet_tokens = tweet_tokens.split()
		tweet_tokens_char_mapping = find_text_to_tweet_tokens_mapping(text, tweet_tokens)
		# NOTE: tweet_tokens_char_mapping is a list of list where each inner list maps every tweet_token's char to the original text char position
		# print(tweet_tokens_char_mapping)
		# Get candidate_chunk's offsets in terms of tweet_tokens
		candidate_chunks_offsets_from_tweet_tokens = list()
		ignore_flags = list()
		for chunk_start_idx, chunk_end_idx in candidate_chunks_offsets:
			ignore_flag = False
			# Find the tweet_token id in which the chunk_start_idx belongs
			chunk_start_token_idx = None
			chunk_end_token_idx = None
			for token_idx, tweet_token_char_mapping in enumerate(tweet_tokens_char_mapping):
				if chunk_start_idx in tweet_token_char_mapping:
					# if chunk start offset is the last character of the token
					chunk_start_token_idx = token_idx
				if (chunk_end_idx-1) in tweet_token_char_mapping:
					# if chunk end offset is the last character of the token
					chunk_end_token_idx = token_idx+1
			if chunk_start_token_idx == None or chunk_end_token_idx == None:
				logging.error(f"Tweet id:{id}\nCouldn't find chunk tokens for chunk offsets [{chunk_start_idx}, {chunk_end_idx}]:{text[chunk_start_idx:chunk_end_idx]};")  
				logging.error(f"Found chunk start and end token idx [{chunk_start_token_idx}, {chunk_end_token_idx}]")  
				# logging.error(f"tweet_tokens_char_mapping len {len(tweet_tokens_char_mapping)}= {tweet_tokens_char_mapping}")
				logging.error(f"Ignoring this chunk")
				ignore_flag = True
				#### DOCUMENT IGNORE THINGS
				ignore_info = {}
				ignore_info['id'] = id
				ignore_info['problem_chunk_id'] = [chunk_start_idx, chunk_end_idx]
				ignore_info['problem_chunk_text'] = text[chunk_start_idx:chunk_end_idx]
				ignore_info['whole_mapping'] = []
				for i in tweet_tokens_char_mapping:
					if len(i) == 1:
						ignore_info['whole_mapping'].append([i, text[i[0]]])
					else:
						ignore_info['whole_mapping'].append([i, text[i[0]:i[-1]+1]])
				ignore_info['flag'] = 'NOT_FIND_ERROR'
				ignore_ones.append(ignore_info)
			ignore_flags.append(ignore_flag)
			candidate_chunks_offsets_from_tweet_tokens.append((chunk_start_token_idx, chunk_end_token_idx))
		candidate_chunks_from_tokens = [' '.join(tweet_tokens[c[0]:c[1]]) for c in candidate_chunks_offsets_from_tweet_tokens]

		# TODO: Verify if the candidate_chunks from tokens and from text are the same
		"""
		for chunk_text, chunk_token, ignore_flag in zip(candidate_chunks_from_text, candidate_chunks_from_tokens, ignore_flags):
			if ignore_flag:
				continue
			try:
				assert re.sub("\s+", "", chunk_text) == re.sub("\s+", "", chunk_token)
			except AssertionError:
				logging.error(f"Chunk and text is not matching the chunk in tokenized tweet")
				chunk_text_without_spaces = re.sub("\s+", "", chunk_text)
				chunk_token_without_spaces = re.sub("\s+", "", chunk_token)
				logging.error(f"Chunk from text without spaces: {chunk_text_without_spaces}")
				logging.error(f"Chunk from tokens without spaces: {chunk_token_without_spaces}")
				exit()
		"""
		# Update the list
		candidate_chunks_from_text = [e for ignore_flag, e in zip(ignore_flags, candidate_chunks_from_text) if not ignore_flag]
		candidate_chunks_from_tokens = [e for ignore_flag, e in zip(ignore_flags, candidate_chunks_from_tokens) if not ignore_flag ]
		candidate_chunks_offsets = [e for ignore_flag, e in zip(ignore_flags, candidate_chunks_offsets) if not ignore_flag ]
		candidate_chunks_offsets_from_tweet_tokens = [e for ignore_flag, e in zip(ignore_flags, candidate_chunks_offsets_from_tweet_tokens) if not ignore_flag ]

		# print_list(candidate_chunks_from_text)
		# print_list(candidate_chunks_from_tokens)
		# Convert annotation to token_idxs from tweet_char_offsets
		# First get the mapping from chunk_char_offsets to chunk_token_idxs
		chunk_char_offsets_to_token_idxs_mapping = {(offset[0],offset[1]):(c[0],c[1]) for offset, c in zip(candidate_chunks_offsets, candidate_chunks_offsets_from_tweet_tokens)}
		# print(chunk_char_offsets_to_token_idxs_mapping)
		annotation_tweet_tokens = dict()
		for key, value in annotation.items():
			# print(key, value)
			if value == "NO_CONSENSUS":
				new_assignments = ["Not Specified"]
			else:
				new_assignments = list()
				for assignment in value:
					if type(assignment) == list:
						# get the candidate_chunk from tweet_tokens
						# print(chunk_char_offsets_to_token_idxs_mapping.keys())
						gold_chunk_token_idxs = chunk_char_offsets_to_token_idxs_mapping[tuple(assignment)]
						new_assignment = ' '.join(tweet_tokens[gold_chunk_token_idxs[0]:gold_chunk_token_idxs[1]])
						new_assignments.append(new_assignment)
					else:
						new_assignments.append(assignment)
			annotation_tweet_tokens[key] = new_assignments

		# print(annotation)
		# print(annotation_tweet_tokens)
		# print(question_keys_and_tags)
		# exit()
		# change the URLs to special URL tag
		# tweet_tokens = [URL_TOKEN if e.startswith("http") or 'twitter.com' in e or e.startswith('www.') else e for e in tweet_tokens]
		final_tweet_tokens = [URL_TOKEN if e.startswith("http") or 'twitter.com' in e or e.startswith('www.') else e for e in tweet_tokens]
		final_candidate_chunks_with_token_id = [(f"{c[0]}_{c[1]}", ' '.join(tweet_tokens[c[0]:c[1]]), c) for c in candidate_chunks_offsets_from_tweet_tokens]
		
		for question_tag, question_key in question_keys_and_tags:
			
			if question_tag in ["name", "close_contact", "who_cure", "opinion"]:
				# add "AUTHOR OF THE TWEET" as a candidate chunk
				final_candidate_chunks_with_token_id.append(["author_chunk", "AUTHOR OF THE TWEET", [0,0]])
				# print(final_candidate_chunks_with_token_id)
				# exit()
			elif question_tag in ["where", "recent_travel"]:
				# add "NEAR AUTHOR OF THE TWEET" as a candidate chunk
				final_candidate_chunks_with_token_id.append(["near_author_chunk", "AUTHOR OF THE TWEET", [0,0]])

			# If there are more then one candidate slot with the same candidate chunk then simply keep the first occurrence. Remove the rest.
			current_candidate_chunks = set()
			for candidate_chunk_with_id in final_candidate_chunks_with_token_id:
				candidate_chunk_id = candidate_chunk_with_id[0]
				candidate_chunk = candidate_chunk_with_id[1]


				if candidate_chunk.lower() == 'coronavirus':
					continue

				chunk_start_id = candidate_chunk_with_id[2][0]
				chunk_start_text_id = tweet_tokens_char_mapping[chunk_start_id][0]
				chunk_end_id = candidate_chunk_with_id[2][1]
				# print(len(tweet_tokens_char_mapping), len(tweet_tokens), chunk_start_id, chunk_end_id)
				chunk_end_text_id = tweet_tokens_char_mapping[chunk_end_id-1][-1]+1
				
				if candidate_chunk == "AUTHOR OF THE TWEET":
					# No need to verify or fix this candidate_chunk
					# print("VERIFY if chunk coming here!")
					# exit()
					pass
				else:
					if chunk_end_id > len(tweet_tokens):
						# Incorrect chunk end id. Skip this chunk
						continue
					candidate_chunk = ' '.join(final_tweet_tokens[chunk_start_id:chunk_end_id])

				if candidate_chunk in current_candidate_chunks:
					# Skip this chunk. Already processed before
					skipped_chunks += 1
					continue
				else:
					# Add to the known list and keep going
					current_candidate_chunks.add(candidate_chunk)
				# assert candidate_chunk == text[chunk_start_text_id:chunk_end_text_id+1]

				# Find gold labels for the current question and candidate chunk
				if question_tag in ["relation", "gender_male", "gender_female", "believe", "binary-relation", "binary-symptoms", "symptoms", "opinion"]:
					# If the question is a yes/no question. It is for the name candidate chunk
					special_tagged_chunks = get_tagged_label_for_key_from_annotation(question_key, annotation_tweet_tokens)
					try:
						assert len(special_tagged_chunks) == 1
					except AssertionError:
						logging.error(f"for question_tag {question_tag} the special_tagged_chunks = {special_tagged_chunks}")
						exit()
					tagged_label = special_tagged_chunks[0]
					if tagged_label == "No":
						tagged_label = "Not Specified"
					if question_tag in ["gender_male", "gender_female"]:
						gender = "Male" if question_tag == "gender_male" else "Female"
						if gender == tagged_label:
							special_question_label = get_label_from_tagged_label(tagged_label)
						else:
							special_question_label = 0
					else:
						special_question_label = get_label_from_tagged_label(tagged_label)

					if question_tag == "opinion":
						# question_label, tagged_chunks = get_label_for_key_from_annotation("part2-who_cure.Response", annotation_tweet_tokens, candidate_chunk)
						tagged_chunks = []
						if candidate_chunk == "AUTHOR OF THE TWEET":
							question_label = 1
							tagged_chunks.append("AUTHOR OF THE TWEET")
						else:
							question_label = 0
					else:
						question_label, tagged_chunks = get_label_for_key_from_annotation("part2-name.Response", annotation_tweet_tokens, candidate_chunk)
					question_label = question_label & special_question_label
					if question_label == 0:
						tagged_chunks = []
				else:
					question_label, tagged_chunks = get_label_for_key_from_annotation(question_key, annotation_tweet_tokens, candidate_chunk)
					# if question_tag == "close_contact" and question_label == 1:
					# 	print(candidate_chunk, annotation_tweet_tokens[question_key], question_label)

				# Add instance
				tokenized_tweet = ' '.join(final_tweet_tokens)
				# text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
				task_instances_dict[question_tag].append((text, candidate_chunk, candidate_chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, ' '.join(final_tweet_tokens[:chunk_start_id] + [Q_TOKEN] + final_tweet_tokens[chunk_end_id:]), tagged_chunks, question_label))
				# Update statistics for data analysis
				# if (tagged_chunks and len(tagged_chunks) == 1 and tagged_chunks[0] == "Not Specified") or question_label == 0:
				gold_labels_stats[question_tag].setdefault(question_label, 0)
				gold_labels_stats[question_tag][question_label] += 1
				gold_labels_unique_tweets[question_tag].setdefault(question_label, set())
				gold_labels_unique_tweets[question_tag][question_label].add(tokenized_tweet)
	logging.info(f"Total skipped chunks:{skipped_chunks}\t n_question tags:{len(question_keys_and_tags)}")

	# Convert gold_labels_unique_tweets from set of tweets to counts

	for question_tag, question_key in question_keys_and_tags:
		label_unique_tweets = gold_labels_unique_tweets[question_tag]
		label_unique_tweets_counts = dict()
		for label, tweets in label_unique_tweets.items():
			label_unique_tweets_counts[label] = len(tweets)
		gold_labels_unique_tweets[question_tag] = label_unique_tweets_counts

	# Log the label-wise total statistics
	logging.info("Gold label instances statistics:")
	log_list(gold_labels_stats.items())
	logging.info("Gold label tweets statistics:")
	log_list(gold_labels_unique_tweets.items())
	tag_statistics = (gold_labels_stats, gold_labels_unique_tweets)

	# TODO: return instances header to save in pickle for later
	# TODO: Think of somehow saving the data statistics somewhere. Maybe save that in pickle as well
	question_tag_gold_chunks = [qt + "_gold_chunks" for qt in question_tags]
	question_tag_gold_labels = [qt + "_label" for qt in question_tags]
	return task_instances_dict, tag_statistics, question_keys_and_tags

def main():
	logging.info(f"Reading annotations from {args.data_file} file...")
	dataset = read_jsonl_datafile(args.data_file)
	logging.info(f"Total annotations:{len(dataset)}")
	logging.info(f"Creating labeled data instances from annotations...")
	print(dataset[0].keys())
	task_instances_dict, tag_statistics, question_keys_and_tags = make_instances_from_dataset(dataset)
	# Save in pickle file
	logging.info(f"Saving all the instances, statistics and labels in {args.save_file}")
	save_in_pickle((task_instances_dict, tag_statistics, question_keys_and_tags), args.save_file)

if __name__ == '__main__':
	main()