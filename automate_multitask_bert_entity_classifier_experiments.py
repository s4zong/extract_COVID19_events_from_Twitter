# We will automate all the logistic regression baseline experiments for different event types and their subtasks
# For each Event type we will first run the data_preprocessing and 
# then run the logistic regression classifier for each subtask that has few (non-zero) positive examples
# We will save all the different classifier models, configs and results in separate directories
# Finally when all the codes have finished we will aggregate all the results and save the final metrics in csv file

from model.utils import make_dir_if_not_exists, load_from_pickle, load_from_json, MIN_POS_SAMPLES_THRESHOLD
import os
import json
import time
import csv
import subprocess

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

task_type_to_datapath_dict = {
								"tested_positive": ("./data/positive-add_text.jsonl", "./data/test_positive.pkl"),
								"tested_negative": ("./data/negative-add_text.jsonl", "./data/test_negative.pkl"),
								"can_not_test": ("./data/can_not_test-add_text.jsonl", "./data/can_not_test.pkl"),
								"death": ("./data/death-add_text.jsonl", "./data/death.pkl"),
								"cure": ("./data/cure_and_prevention-add_text.jsonl", "./data/cure_and_prevention.pkl"),
								}

# REDO_DATA_FLAG = True
REDO_DATA_FLAG = False
REDO_FLAG = True
RETRAIN_FLAG = True
# REDO_FLAG = False

# We will save all the tasks and subtask's results and model configs in this dictionary
all_task_results_and_model_configs = dict()
# We will save the list of question_tags AKA subtasks for each event AKA task in this dict
all_task_question_tags = dict()
for taskname, (data_in_file, processed_out_file) in task_type_to_datapath_dict.items():
	if not os.path.exists(processed_out_file) or REDO_DATA_FLAG:
		data_preprocessing_cmd = f"python model/data_preprocessing.py -d {data_in_file} -s {processed_out_file}"
		logging.info(data_preprocessing_cmd)
		os.system(data_preprocessing_cmd)
	else:
		logging.info(f"Preprocessed data for task {taskname} already exists at {processed_out_file}")

	# Read the data statistics
	task_instances_dict, tag_statistics, question_keys_and_tags = load_from_pickle(processed_out_file)

	# We will store the list of subtasks for which we train the classifier
	tested_tasks = list()
	logging.info(f"Training Mutlitask BERT Entity Classifier model on {processed_out_file}")
	# output_dir = os.path.join("results", "multitask_bert_entity_classifier", taskname)
	# NOTE: After fixing the USER and URL tags
	output_dir = os.path.join("results", "multitask_bert_entity_classifier_fixed", taskname)
	make_dir_if_not_exists(output_dir)
	results_file = os.path.join(output_dir, "results.json")
	model_config_file = os.path.join(output_dir, "model_config.json")
	if not os.path.exists(results_file) or REDO_FLAG:
		# Execute the Bert entity classifier train and test only if the results file doesn't exists
		# multitask_bert_cmd = f"python model/multitask_bert_entitity_classifier.py -d {processed_out_file} -t {taskname} -o {output_dir} -s saved_models/multitask_bert_entity_classifier/{taskname}_8_epoch_32_batch_multitask_bert_model"
		# After fixing the USER and URL tags
		multitask_bert_cmd = f"python model/multitask_bert_entitity_classifier.py -d {processed_out_file} -t {taskname} -o {output_dir} -s saved_models/multitask_bert_entity_classifier_fixed/{taskname}_8_epoch_32_batch_multitask_bert_model"
		if RETRAIN_FLAG:
			multitask_bert_cmd += " -r"
		logging.info(f"Running: {multitask_bert_cmd}")
		try:
			retcode = subprocess.call(multitask_bert_cmd, shell=True)
			# os.system(multitask_bert_cmd)
		except KeyboardInterrupt:
			exit()
	#  Read the results from the results json file
	results = load_from_json(results_file)
	model_config = load_from_json(model_config_file)
	# We will save the classifier results and model config for each subtask in this dictionary
	all_subtasks_results_and_model_configs = dict()
	for key in results:
		if key not in ["best_dev_threshold", "best_dev_F1s", "dev_t_F1_P_Rs"]:
			tested_tasks.append(key)
			results[key]["best_dev_threshold"] = results["best_dev_threshold"][key]
			results[key]["best_dev_F1"] = results["best_dev_F1s"][key]
			results[key]["dev_t_F1_P_Rs"] = results["dev_t_F1_P_Rs"][key]
			all_subtasks_results_and_model_configs[key] = results[key], model_config
	all_task_results_and_model_configs[taskname] = all_subtasks_results_and_model_configs
	all_task_question_tags[taskname] = tested_tasks

# Read the results for each task and save them in csv file
# results_tsv_save_file = os.path.join("results", "all_experiments_multitask_bert_entity_classifier_results.tsv")
# NOTE: After fixing the USER and URL tags
results_tsv_save_file = os.path.join("results", "all_experiments_multitask_bert_entity_classifier_fixed_results.tsv")
with open(results_tsv_save_file, "w") as tsv_out:
	writer = csv.writer(tsv_out, delimiter='\t')
	header = ["Event", "Sub-task", "Train Data (size, pos., neg.)", "Dev Data (size, pos., neg.)", "Test Data (size, pos., neg.)", "model name", "accuracy", "CM", "pos. F1", "SQuAD_total", "SQuAD_EM", "SQuAD_F1", "SQuAD_Pos. EM_F1_total", "SQuAD_Pos. EM", "SQuAD_Pos. F1", "dev_threshold", "dev_N", "dev_F1", "dev_P", "dev_R", "dev_TP", "dev_FP", "dev_FN", "N", "F1", "P", "R", "TP", "FP", "FN"]
	writer.writerow(header)
	for taskname, question_tags in all_task_question_tags.items():
		current_task_results_and_model_configs = all_task_results_and_model_configs[taskname]
		for question_tag in question_tags:
			results, model_config = current_task_results_and_model_configs[question_tag]
			# Extract results
			classification_report = results["Classification Report"]
			positive_f1_classification_report = classification_report['1']['f1-score']
			accuracy = classification_report['accuracy']
			CM = results["CM"]
			# SQuAD results
			total_EM = results["SQuAD_EM"]
			total_F1 = results["SQuAD_F1"]
			total_tweets = results["SQuAD_total"]
			pos_EM = results["SQuAD_Pos. EM"]
			pos_F1 = results["SQuAD_Pos. F1"]
			total_pos_tweets = results["SQuAD_Pos. EM_F1_total"]
			# Best threshold and dev F1
			best_dev_threshold = results["best_dev_threshold"]
			best_dev_F1 = results["best_dev_F1"]
			dev_t_F1_P_Rs = results["dev_t_F1_P_Rs"]
			best_dev_threshold_index = int(best_dev_threshold * 10) - 1
			# Each entry in dev_t_F1_P_Rs is of the format t, dev_F1, dev_P, dev_R, dev_TP + dev_FN, dev_TP, dev_FP, dev_FN
			t, dev_F1, dev_P, dev_R, dev_N, dev_TP, dev_FP, dev_FN = dev_t_F1_P_Rs[best_dev_threshold_index]
			# Alan's metrics
			F1 = results["F1"]
			P = results["P"]
			R = results["R"]
			TP = results["TP"]
			FP = results["FP"]
			FN = results["FN"]
			N = results["N"]
			# Extract model config
			model_name = model_config["model"]
			train_data = (model_config["train_data"]["size"], model_config["train_data"]["pos"], model_config["train_data"]["neg"])
			dev_data = (model_config["dev_data"]["size"], model_config["dev_data"]["pos"], model_config["dev_data"]["neg"])
			test_data = (model_config["test_data"]["size"], model_config["test_data"]["pos"], model_config["test_data"]["neg"])
			
			row = [taskname, question_tag, train_data, dev_data, test_data, model_name, accuracy, CM, positive_f1_classification_report, total_tweets, total_EM, total_F1, total_pos_tweets, pos_EM, pos_F1, best_dev_threshold, dev_N, dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN, N, F1, P, R, TP, FP, FN]
			writer.writerow(row)




		

