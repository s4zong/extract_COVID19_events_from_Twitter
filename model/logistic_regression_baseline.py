# We will implement an ngram feature based logistic regression baseline for different tasks
# We will first split the instances into train and test
# Then we will create the n-gram features based on train split
# Using the feature we will create sparse feature matrix for train and test split
# Then we will train the logistic regresison classifier on the train split and evaluate it on the test split

import argparse
import os
import re
import string
import collections

import numpy as np
from collections import Counter
import pickle

from sklearn.linear_model import LogisticRegression, Ridge
from scipy.sparse import *
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import log_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, extract_instances_for_current_subtask, split_instances_in_train_dev_test, log_data_statistics, save_in_json, get_raw_scores, get_TP_FP_FN

Q_TOKEN = "<Q_TARGET>"
URL_TOKEN = "<URL>"
RANDOM_SEED = 901
Ns_for_NGRAMS = [1,2,3]

import random
random.seed(RANDOM_SEED)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_file", help="Path to the pickle file that contains the training instances", type=str, required=True)
parser.add_argument("-t", "--task", help="Event for which we want to train the baseline", type=str, required=True)
parser.add_argument("-st", "--sub_task", help="slot name of question for which we want to train the baseline", type=str, required=True)
parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all the model results", type=str, required=True)
args = parser.parse_args()

import logging
# Ref: https://stackoverflow.com/a/49202811/4535284
for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)
# Also add the stream handler so that it logs on STD out as well
# Ref: https://stackoverflow.com/a/46098711/4535284
make_dir_if_not_exists(args.output_dir)
logfile = os.path.join(args.output_dir, "output.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

def get_ngrams_dict_from_text(n, text):
	split_text = text.split()
	ngrams = dict()
	# print(text)
	for i in range(0, len(split_text) + 1 - n):
		current_ngram = ' '.join(split_text[i:i+n])
		ngrams.setdefault(current_ngram, 0)
		ngrams[current_ngram] += 1
		# print(split_text[i:i+n])
	return ngrams

def create_ngram_features_from(data):
	# Extract all the 1,2,3-grams from the text part of the data
	
	# We will store all the ngrams in a dict of dict structure
	# Outer dict will have key = n and value as dictionary of ngram counts
	# initialize the structure
	ngrams = dict()
	for n in Ns_for_NGRAMS:
		ngrams[n] = dict()

	# Extract ngrams from all the text fields in the data and populate the dictionary
	for text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, gold_chunk, label in data:
		for n in Ns_for_NGRAMS:
			current_text_ngrams = get_ngrams_dict_from_text(n, tokenized_tweet_with_masked_chunk)
			for current_ngram, count in current_text_ngrams.items():
				ngrams[n].setdefault(current_ngram, 0)
				ngrams[n][current_ngram] += 1
	
	# Print the sizes of different ngrams
	# for n in Ns_for_NGRAMS:
	# 	print(n, len(ngrams[n]))

	# Merge all ngrams into a serialized feature dictionary
	feature2i = dict()		# dict with key as ngram and value as its feature_index
	i2feature = list()		# list of all the features where index is the feature_index
	index = 0
	for n in Ns_for_NGRAMS:
		for ngram in sorted(ngrams[n].keys()):
			feature2i[ngram] = index
			i2feature.append(ngram)
			index += 1
	return feature2i, i2feature

def convert_data_to_feature_vector_and_labels(data, feature2i):
	# We will take the data and the feature2i dict and create a sparse matrix of features along with a list of their corresponding labels
	# sparse feature matrix will be of the type csr_matrix from scipy.sparse. ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
	
	"""
	scipy.sparse.csr_matrix initialization example
	>>> row = np.array([0, 0, 1, 2, 2, 2])
	>>> col = np.array([0, 2, 2, 0, 1, 2])
	>>> data = np.array([1, 2, 3, 4, 5, 6])
	>>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
	array([[1, 0, 2],
		   [0, 0, 3],
		   [4, 5, 6]])
	"""
	row = list()
	col = list()
	values = list()
	labels = list()
	found_count = 0
	# text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
	for row_id, (text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, gold_chunk, label) in enumerate(data):
		labels.append(label)
		for n in Ns_for_NGRAMS:
			text_ngrams_dict = get_ngrams_dict_from_text(n, tokenized_tweet_with_masked_chunk)
			# if any of the ngrams in the current text's ngrams_dict matches with the features then add it to the row, col, data
			for current_text_ngram, count in text_ngrams_dict.items():
				if current_text_ngram in feature2i:
					found_count += 1
					row.append(row_id)
					col.append(feature2i[current_text_ngram])
					values.append(count)
	# print("Total ngrams found:", found_count)
	row = np.array(row)
	col = np.array(col)
	values = np.array(values)
	# Create feature csr matrix
	feature_matrix = csr_matrix((values, (row, col)), shape=(len(data), len(feature2i)))
	return feature_matrix, labels

def main():
	task_instances_dict, tag_statistics, question_keys_and_tags = load_from_pickle(args.data_file)
	data = extract_instances_for_current_subtask(task_instances_dict, args.sub_task)
	logging.info(f"Task dataset for task: {args.task} loaded from {args.data_file}.")
	
	model_config = dict()
	results = dict()

	# Split the data into train, dev and test and shuffle the train segment
	train_data, dev_data, test_data = split_instances_in_train_dev_test(data)
	random.shuffle(train_data)		# shuffle happens in-place
	logging.info("Train Data:")
	total_train_size, pos_train_size, neg_train_size = log_data_statistics(train_data)
	logging.info("Dev Data:")
	total_dev_size, pos_dev_size, neg_dev_size = log_data_statistics(dev_data)
	logging.info("Test Data:")
	total_test_size, pos_test_size, neg_test_size = log_data_statistics(test_data)
	logging.info("\n")
	model_config["train_data"] = {"size":total_train_size, "pos":pos_train_size, "neg":neg_train_size}
	model_config["dev_data"] = {"size":total_dev_size, "pos":pos_dev_size, "neg":neg_dev_size}
	model_config["test_data"] = {"size":total_test_size, "pos":pos_test_size, "neg":neg_test_size}
	
	# Extract n-gram features from the train data
	# Returned ngrams will be dict of dict
	# TODO: update the feature extractor
	feature2i, i2feature = create_ngram_features_from(train_data)
	logging.info(f"Total number of features extracted from train = {len(feature2i)}, {len(i2feature)}")
	model_config["features"] = {"size": len(feature2i)}

	# Extract Feature vectors and labels from train and test data
	train_X, train_Y = convert_data_to_feature_vector_and_labels(train_data, feature2i)
	dev_X, dev_Y = convert_data_to_feature_vector_and_labels(dev_data, feature2i)
	test_X, test_Y = convert_data_to_feature_vector_and_labels(test_data, feature2i)
	logging.info(f"Train Data Features = {train_X.shape} and Labels = {len(train_Y)}")
	logging.info(f"Dev Data Features = {dev_X.shape} and Labels = {len(dev_Y)}")
	logging.info(f"Test Data Features = {test_X.shape} and Labels = {len(test_Y)}")
	model_config["train_data"]["features_shape"] = train_X.shape
	model_config["train_data"]["labels_shape"] = len(train_Y)
	model_config["dev_data"]["features_shape"] = dev_X.shape
	model_config["dev_data"]["labels_shape"] = len(dev_Y)
	model_config["test_data"]["features_shape"] = test_X.shape
	model_config["test_data"]["labels_shape"] = len(test_Y)
	
	# Train logistic regression classifier
	logging.info("Training the Logistic Regression classifier")
	lr = LogisticRegression(solver='lbfgs', max_iter=1000)
	lr.fit(train_X, train_Y)
	model_config["model"] = "LogisticRegression(solver='lbfgs')"
	
	# Find best threshold based on dev set performance
	thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

	dev_prediction_probs = lr.predict_proba(dev_X)[:, 1]
	dev_t_F1_P_Rs = list()
	best_threshold_based_on_F1 = 0.5
	best_dev_F1 = 0.0
	for t in thresholds:
		dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN = get_TP_FP_FN(dev_data, dev_prediction_probs, THRESHOLD=t)
		dev_t_F1_P_Rs.append((t, dev_F1, dev_P, dev_R, dev_TP + dev_FN, dev_TP, dev_FP, dev_FN))
		if dev_F1 > best_dev_F1:
			best_threshold_based_on_F1 = t
			best_dev_F1 = dev_F1
	log_list(dev_t_F1_P_Rs)
	logging.info(f"Best Threshold: {best_threshold_based_on_F1}\t Best dev F1: {best_dev_F1}")
	# Save the best dev threshold and dev_F1 in results dict
	results["best_dev_threshold"] = best_threshold_based_on_F1
	results["best_dev_F1"] = best_dev_F1
	results["dev_t_F1_P_Rs"] = dev_t_F1_P_Rs
	# y_pred = (clf.predict_proba(X_test)[:,1] >= 0.3).astype(bool)

	# Test 
	logging.info("Testing the trained classifier")
	predictions = lr.predict(test_X)
	probs = lr.predict_proba(test_X)
	test_Y_prediction_probs = probs[:, 1]

	cm = metrics.confusion_matrix(test_Y, predictions)
	classification_report = metrics.classification_report(test_Y, predictions, output_dict=True)
	logging.info(cm)
	logging.info(metrics.classification_report(test_Y, predictions))
	results["CM"] = cm.tolist()			# Storing it as list of lists instead of numpy.ndarray
	results["Classification Report"] = classification_report
	
	# SQuAD style EM and F1 evaluation for all test cases and for positive test cases (i.e. for cases where annotators had a gold annotation)
	EM_score, F1_score, total = get_raw_scores(test_data, test_Y_prediction_probs)
	logging.info("Word overlap based SQuAD evaluation style metrics:")
	logging.info(f"Total number of cases: {total}")
	logging.info(f"EM_score: {EM_score}")
	logging.info(f"F1_score: {F1_score}")
	results["SQuAD_EM"] = EM_score
	results["SQuAD_F1"] = F1_score
	results["SQuAD_total"] = total
	pos_EM_score, pos_F1_score, pos_total = get_raw_scores(test_data, test_Y_prediction_probs, positive_only=True)
	logging.info(f"Total number of Positive cases: {pos_total}")
	logging.info(f"Pos. EM_score: {pos_EM_score}")
	logging.info(f"Pos. F1_score: {pos_F1_score}")
	results["SQuAD_Pos. EM"] = pos_EM_score
	results["SQuAD_Pos. F1"] = pos_F1_score
	results["SQuAD_Pos. EM_F1_total"] = pos_total

	# New evaluation suggested by Alan
	F1, P, R, TP, FP, FN = get_TP_FP_FN(test_data, test_Y_prediction_probs, THRESHOLD=best_threshold_based_on_F1)
	logging.info("New evaluation scores:")
	logging.info(f"F1: {F1}")
	logging.info(f"Precision: {P}")
	logging.info(f"Recall: {R}")
	logging.info(f"True Positive: {TP}")
	logging.info(f"False Positive: {FP}")
	logging.info(f"False Negative: {FN}")
	results["F1"] = F1
	results["P"] = P
	results["R"] = R
	results["TP"] = TP
	results["FP"] = FP
	results["FN"] = FN
	N = TP + FN
	results["N"] = N

	# # Top predictions in the Test case
	sorted_prediction_ids = np.argsort(-test_Y_prediction_probs)
	# K = 30
	# logging.info("Top {} predictions:".format(K))
	# for i in range(K):
	# 	instance_id = sorted_prediction_ids[i]
	# 	# text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
	# 	list_to_print = [test_data[instance_id][0], test_data[instance_id][6], test_data[instance_id][1], str(test_Y_prediction_probs[instance_id]), str(test_Y[instance_id]), str(test_data[instance_id][-1]), str(test_data[instance_id][-2])]
	# 	logging.info("\t".join(list_to_print))
	#
	# # Top feature analysis
	# coefs=lr.coef_[0]
	# K = 10
	# sorted_feature_ids = np.argsort(-coefs)
	# logging.info("Top {} features:".format(K))
	# for i in range(K):
	# 	feature_id = sorted_feature_ids[i]
	# 	logging.info(f"{i2feature[feature_id]}\t{coefs[feature_id]}")
	#
	# # Plot the precision recall curve
	# save_figure_file = os.path.join(args.output_dir, "Precision Recall Curve.png")
	# logging.info(f"Saving precision recall curve at {save_figure_file}")
	# disp = plot_precision_recall_curve(lr, test_X, test_Y)
	# disp.ax_.set_title('2-class Precision-Recall curve')
	# disp.ax_.figure.savefig(save_figure_file)

	# Save the model and features in pickle file
	model_and_features_save_file = os.path.join(args.output_dir, "model_and_features.pkl")
	logging.info(f"Saving LR model and features at {model_and_features_save_file}")
	save_in_pickle((lr, feature2i, i2feature), model_and_features_save_file)

	# Save model_config and results
	model_config_file = os.path.join(args.output_dir, "model_config.json")
	results_file = os.path.join(args.output_dir, "results.json")
	logging.info(f"Saving model config at {model_config_file}")
	save_in_json(model_config, model_config_file)
	logging.info(f"Saving results at {results_file}")
	save_in_json(results, results_file)


if __name__ == '__main__':

	main()
