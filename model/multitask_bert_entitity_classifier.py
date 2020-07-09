from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch

RANDOM_SEED = 901
import random
random.seed(RANDOM_SEED)

import numpy as np
from collections import Counter
import pickle
from pprint import pprint

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from tqdm import tqdm
import argparse
import time
import datetime
import string
import re
import collections

from utils import log_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, get_multitask_instances_for_valid_tasks, split_multitask_instances_in_train_dev_test, log_data_statistics, save_in_json, get_raw_scores, get_TP_FP_FN

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data_file", help="Path to the pickle file that contains the training instances", type=str, required=True)
parser.add_argument("-t", "--task", help="Event for which we want to train the baseline", type=str, required=True)
parser.add_argument("-s", "--save_directory", help="Path to the directory where we will save model and the tokenizer", type=str, required=True)
parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all the model results", type=str, required=True)
parser.add_argument("-rt", "--retrain", help="Flag that will indicate if the model needs to be retrained or loaded from the existing save_directory", action="store_true")
parser.add_argument("-bs", "--batch_size", help="Train batch size for BERT model", type=int, default=32)
parser.add_argument("-e", "--n_epochs", help="Number of epochs", type=int, default=8)
args = parser.parse_args()

import logging
# Ref: https://stackoverflow.com/a/49202811/4535284
for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)
# Also add the stream handler so that it logs on STD out as well
# Ref: https://stackoverflow.com/a/46098711/4535284
make_dir_if_not_exists(args.output_dir)
if args.retrain:
	logfile = os.path.join(args.output_dir, "train_output.log")
else:
	logfile = os.path.join(args.output_dir, "output.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

Q_TOKEN = "<Q_TARGET>"
URL_TOKEN = "<URL>"
RANDOM_SEED = 901
torch.manual_seed(RANDOM_SEED)
POSSIBLE_BATCH_SIZE = 8

# export CUDA_VISIBLE_DEVICES=3
if torch.cuda.is_available():
	device = torch.device("cuda")
	logging.info(f"Using GPU{torch.cuda.get_device_name(0)} to train")
else:
	device = torch.device("cpu")
	logging.info(f"Using CPU to train")

def make_dir_if_not_exists(directory):
	if not os.path.exists(directory):
		logging.info("Creating new directory: {}".format(directory))
		os.makedirs(directory)

def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))
	
	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))

class MultiTaskBertForCovidEntityClassification(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

		self.subtasks = config.subtasks
		# We will create a dictionary of classifiers based on the number of subtasks
		self.classifiers = {subtask: nn.Linear(config.hidden_size, config.num_labels) for subtask in self.subtasks}
		# self.classifier = nn.Linear(config.hidden_size, config.num_labels)

		self.init_weights()

	def forward(
		self,
		input_ids,
		entity_start_positions,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
			Labels for computing the sequence classification/regression loss.
			Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
			If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
			If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

	Returns:
		:obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
		loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
			Classification (or regression if config.num_labels==1) loss.
		logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
			Classification (or regression if config.num_labels==1) scores (before SoftMax).
		hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
			Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
			of shape :obj:`(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
			Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
			:obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
			heads.

	Examples::

		from transformers import BertTokenizer, BertForSequenceClassification
		import torch

		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

		input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
		labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
		outputs = model(input_ids, labels=labels)

		loss, logits = outputs[:2]

		"""

		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)
		# DEBUG:
		# print("BERT model outputs shape", outputs[0].shape, outputs[1].shape)
		# print(entity_start_positions[:, 0], entity_start_positions[:, 1])
		
		# OLD CODE:
		# pooled_output = outputs[1]

		# NOTE: outputs[0] has all the hidden dimensions for the entire sequence
		# We will extract the embeddings indexed with entity_start_positions
		pooled_output = outputs[0][entity_start_positions[:, 0], entity_start_positions[:, 1], :]
		
		# DEBUG:
		# print(pooled_output.shape)
		# exit()

		pooled_output = self.dropout(pooled_output)
		# Get logits for each subtask
		# logits = self.classifier(pooled_output)
		logits = {subtask: self.classifiers[subtask](pooled_output) for subtask in self.subtasks}

		outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

		if labels is not None:
			loss_fct = nn.CrossEntropyLoss()
			
			# DEBUG:
			# print(f"Logits:{logits.view(-1, self.num_labels)}, \t, Labels:{labels.view(-1)}")
			for i, subtask in enumerate(self.subtasks):
				# print(labels[subtask].is_cuda)
				if i == 0:
					loss = loss_fct(logits[subtask].view(-1, self.num_labels), labels[subtask].view(-1))
				else:
					loss += loss_fct(logits[subtask].view(-1, self.num_labels), labels[subtask].view(-1))
			outputs = (loss,) + outputs

		return outputs  # (loss), logits, (hidden_states), (attentions)

class COVID19TaskDataset(Dataset):
	"""COVID19TaskDataset is a generic dataset class which will read data related to different questions"""
	def __init__(self, instances):
		super(COVID19TaskDataset, self).__init__()
		self.instances = instances
		self.nsamples = len(self.instances)

	def __getitem__(self, index):
		return self.instances[index]

	def __len__(self):
		return self.nsamples

class TokenizeCollator():
	def __init__(self, tokenizer, subtasks, entity_start_token_id):
		self.tokenizer = tokenizer
		self.subtasks = subtasks
		self.entity_start_token_id = entity_start_token_id

	def fix_user_mentions_in_tokenized_tweet(self, tokenized_tweet):
		return ' '.join(["@USER" if word.startswith("@") else word for word in tokenized_tweet.split()])

	def __call__(self, batch):
		all_bert_model_input_texts = list()
		gold_labels = {subtask: list() for subtask in self.subtasks}
		# text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
		for text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, subtask_labels_dict in batch:
			tokenized_tweet_with_masked_chunk = self.fix_user_mentions_in_tokenized_tweet(tokenized_tweet_with_masked_chunk)
			if chunk in ["AUTHOR OF THE TWEET", "NEAR AUTHOR OF THE TWEET"]:
				# First element of the text will be considered as AUTHOR OF THE TWEET or NEAR AUTHOR OF THE TWEET
				bert_model_input_text = tokenized_tweet_with_masked_chunk.replace(Q_TOKEN, "<E> </E>")
				# print(tokenized_tweet_with_masked_chunk)
				# print(bert_model_input_text)
				# exit()
			else:
				bert_model_input_text = tokenized_tweet_with_masked_chunk.replace(Q_TOKEN, "<E> " + chunk + " </E>")
			all_bert_model_input_texts.append(bert_model_input_text)
			# Add subtask labels in the gold_labels dictionary
			for subtask in self.subtasks:
				gold_labels[subtask].append(subtask_labels_dict[subtask][1])
		# Tokenize
		all_bert_model_inputs_tokenized = self.tokenizer.batch_encode_plus(all_bert_model_input_texts, pad_to_max_length=True, return_tensors="pt")
		input_ids, token_type_ids, attention_mask = all_bert_model_inputs_tokenized['input_ids'], all_bert_model_inputs_tokenized['token_type_ids'], all_bert_model_inputs_tokenized['attention_mask']
		# print(input_ids.type())
		# print(input_ids.size())
		# print(input_ids)
		# First extract the indices of <E> token in each sentence and save it in the batch
		entity_start_positions = (input_ids == self.entity_start_token_id).nonzero()
		# Also extract the gold labels
		labels = {subtask: torch.LongTensor(subtask_gold_labels) for subtask, subtask_gold_labels in gold_labels.items()}
		# print(len(batch))
		if entity_start_positions.size(0) == 0:
			# Send entity_start_positions to [CLS]'s position i.e. 0
			entity_start_positions = torch.zeros(input_ids.size(0), 2).long()
		# print(entity_start_positions)
		# print(input_ids.size(), labels.size())

		# Verify that the number of labels for each subtask is equal to the number of instances
		for subtask in self.subtasks:
			try:
				assert input_ids.size(0) == labels[subtask].size(0)
			except AssertionError:
				logging.error(f"Error Bad batch: Incorrect number of labels given for the batch of size: {len(batch)}")
				logging.error(f"{subtask}, {labels[subtask]}, {labels[subtask].size(0)}")
				exit()
		# assert input_ids.size(0) == labels.size(0)
		return {"input_ids": input_ids, "entity_start_positions": entity_start_positions, "gold_labels": labels, "batch_data": batch}


"""
def _glue_convert_examples_to_features(
	examples: List[InputExample],
	tokenizer: PreTrainedTokenizer,
	max_length: Optional[int] = None,
	task=None,
	label_list=None,
	output_mode=None,
):
	if max_length is None:
		max_length = tokenizer.max_len

	if task is not None:
		processor = glue_processors[task]()
		if label_list is None:
			label_list = processor.get_labels()
			logging.info("Using label list %s for task %s" % (label_list, task))
		if output_mode is None:
			output_mode = glue_output_modes[task]
			logging.info("Using output mode %s for task %s" % (output_mode, task))

	label_map = {label: i for i, label in enumerate(label_list)}

	def label_from_example(example: InputExample) -> Union[int, float]:
		if output_mode == "classification":
			return label_map[example.label]
		elif output_mode == "regression":
			return float(example.label)
		raise KeyError(output_mode)

	labels = [label_from_example(example) for example in examples]

	batch_encoding = tokenizer.batch_encode_plus(
		[(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
	)

	features = []
	for i in range(len(examples)):
		inputs = {k: batch_encoding[k][i] for k in batch_encoding}

		feature = InputFeatures(**inputs, label=labels[i])
		features.append(feature)

	for i, example in enumerate(examples[:5]):
		logging.info("*** Example ***")
		logging.info("guid: %s" % (example.guid))
		logging.info("features: %s" % features[i])

	return features
"""

def make_predictions_on_dataset(dataloader, model, device, dataset_name, dev_flag = False):
	# Create tqdm progressbar
	if dev_flag:
		pbar = dataloader
	else:
		logging.info(f"Predicting on the dataset {dataset_name}")
		pbar = tqdm(dataloader)
	# Setting model to eval for predictions
	# NOTE: assuming that model is already in the given device
	model.eval()
	subtasks = model.subtasks
	all_predictions = {subtask: list() for subtask in subtasks}
	all_prediction_scores = {subtask: list() for subtask in subtasks}
	all_labels = {subtask: list() for subtask in subtasks}
	softmax_func = nn.Softmax(dim=1)
	with torch.no_grad():
		for step, batch in enumerate(pbar):
			# Create testing instance for model
			input_dict = {"input_ids": batch["input_ids"].to(device), "entity_start_positions": batch["entity_start_positions"].to(device)}
			labels = batch["gold_labels"]
			logits = model(**input_dict)[0]

			# DEBUG:
			# print(labels)
			# print(logits)
			# print(type(logits))

			# Apply softmax on each subtask's logits			
			# softmax_logits = softmax_func(logits)
			softmax_logits = {subtask: softmax_func(logits[subtask]) for subtask in subtasks}
			
			# DEBUG:
			# print(softmax_logits)
			
			# Extract predicted labels and predicted scores for each subtask
			# _, predicted_labels = softmax_logits.max(1)
			# prediction_scores = softmax_logits[:, 1]
			# prediction_scores = prediction_scores.cpu().tolist()
			# predicted_labels = predicted_labels.cpu().tolist()
			predicted_labels = {subtask: None for subtask in subtasks}
			prediction_scores = {subtask: None for subtask in subtasks}
			for subtask in subtasks:
				_, predicted_label = softmax_logits[subtask].max(1)
				prediction_score = softmax_logits[subtask][:, 1]
				prediction_scores[subtask] = prediction_score.cpu().tolist()
				predicted_labels[subtask] = predicted_label.cpu().tolist()
				# Save all the predictions and labels in lists
				all_predictions[subtask].extend(predicted_labels[subtask])
				all_prediction_scores[subtask].extend(prediction_scores[subtask])
				all_labels[subtask].extend(labels[subtask])
			# DEBUG:
			# print(prediction_scores)
			# print(predicted_labels)
	return all_predictions, all_prediction_scores, all_labels

def plot_train_loss(loss_trajectory_per_epoch, trajectory_file):
	plt.cla()
	plt.clf()

	fig, ax = plt.subplots()
	x = [epoch * len(loss_trajectory) + j + 1 for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory) ]
	x_ticks = [ "(" + str(epoch + 1) + "," + str(j + 1) + ")" for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory) ]
	full_train_trajectory = [avg_loss for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory)]
	ax.plot(x, full_train_trajectory)

	ax.set(xlabel='Epoch, Step', ylabel='Loss',
			title='Train loss trajectory')
	step_size = 100
	ax.xaxis.set_ticks(range(0, len(x_ticks), step_size), x_ticks[::step_size])
	ax.grid()

	fig.savefig(trajectory_file)

def split_data_based_on_subtasks(data, subtasks):
	# We will split the data into data_instances based on subtask_labels
	subtasks_data = {subtask: list() for subtask in subtasks}
	for text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, subtask_labels_dict in data:
		for subtask in subtasks:
			subtasks_data[subtask].append((text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, subtask_labels_dict[subtask][0], subtask_labels_dict[subtask][1]))
	return subtasks_data

def log_multitask_data_statistics(data, subtasks):
	logging.info(f"Total instances in the data = {len(data)}")
	# print positive and negative counts for each subtask
	# print(len(data[0]))
	pos_counts = {subtask: sum(subtask_labels_dict[subtask][1] for _,_,_,_,_,_,_,subtask_labels_dict in data) for subtask in subtasks}
	# Log for each subtask
	neg_counts = dict()
	for subtask in subtasks:
		neg_counts[subtask] = len(data) - pos_counts[subtask]
		logging.info(f"Subtask:{subtask:>15}\tPositive labels = {pos_counts[subtask]}\tNegative labels = {neg_counts[subtask]}")
	return len(data), pos_counts, neg_counts

def main():
	# Read all the data instances
	task_instances_dict, tag_statistics, question_keys_and_tags = load_from_pickle(args.data_file)
	data, subtasks_list = get_multitask_instances_for_valid_tasks(task_instances_dict, tag_statistics)

	if args.retrain:
		logging.info("Creating and training the model from 'bert-base-cased' ")
		# Create the save_directory if not exists
		make_dir_if_not_exists(args.save_directory)

		# Initialize tokenizer and model with pretrained weights
		tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
		config = BertConfig.from_pretrained('bert-base-cased')
		config.subtasks = subtasks_list
		# print(config)
		model = MultiTaskBertForCovidEntityClassification.from_pretrained('bert-base-cased', config=config)

		# Add new tokens in tokenizer
		new_special_tokens_dict = {"additional_special_tokens": ["<E>", "</E>", "<URL>", "@USER"]}
		# new_special_tokens_dict = {"additional_special_tokens": ["<E>", "</E>"]}
		tokenizer.add_special_tokens(new_special_tokens_dict)
		
		# Add the new embeddings in the weights
		print("Embeddings type:", model.bert.embeddings.word_embeddings.weight.data.type())
		print("Embeddings shape:", model.bert.embeddings.word_embeddings.weight.data.size())
		embedding_size = model.bert.embeddings.word_embeddings.weight.size(1)
		new_embeddings = torch.FloatTensor(len(new_special_tokens_dict["additional_special_tokens"]), embedding_size).uniform_(-0.1, 0.1)
		# new_embeddings = torch.FloatTensor(2, embedding_size).uniform_(-0.1, 0.1)
		print("new_embeddings shape:", new_embeddings.size())
		new_embedding_weight = torch.cat((model.bert.embeddings.word_embeddings.weight.data,new_embeddings), 0)
		model.bert.embeddings.word_embeddings.weight.data = new_embedding_weight
		print("Embeddings shape:", model.bert.embeddings.word_embeddings.weight.data.size())
		# Update model config vocab size
		model.config.vocab_size = model.config.vocab_size + len(new_special_tokens_dict["additional_special_tokens"])
	else:
		# Load the tokenizer and model from the save_directory
		tokenizer = BertTokenizer.from_pretrained(args.save_directory)
		model = MultiTaskBertForCovidEntityClassification.from_pretrained(args.save_directory)
		# print(model.state_dict().keys())
		# TODO save and load the subtask classifier weights separately
		# Load from individual state dicts
		for subtask in model.subtasks:
			model.classifiers[subtask].load_state_dict(torch.load(os.path.join(args.save_directory, f"{subtask}_classifier.bin")))
		# print(model.config)
		# exit()
	model.to(device)
	# Explicitly move the classifiers to device
	for subtask, classifier in model.classifiers.items():
		classifier.to(device)
	entity_start_token_id = tokenizer.convert_tokens_to_ids(["<E>"])[0]
	
	logging.info(f"Task dataset for task: {args.task} loaded from {args.data_file}.")
	
	model_config = dict()
	results = dict()

	# Split the data into train, dev and test and shuffle the train segment
	train_data, dev_data, test_data = split_multitask_instances_in_train_dev_test(data)
	random.shuffle(train_data)		# shuffle happens in-place
	logging.info("Train Data:")
	total_train_size, pos_subtasks_train_size, neg_subtasks_train_size = log_multitask_data_statistics(train_data, model.subtasks)
	logging.info("Dev Data:")
	total_dev_size, pos_subtasks_dev_size, neg_subtasks_dev_size = log_multitask_data_statistics(dev_data, model.subtasks)
	logging.info("Test Data:")
	total_test_size, pos_subtasks_test_size, neg_subtasks_test_size = log_multitask_data_statistics(test_data, model.subtasks)
	logging.info("\n")
	model_config["train_data"] = {"size":total_train_size, "pos":pos_subtasks_train_size, "neg":neg_subtasks_train_size}
	model_config["dev_data"] = {"size":total_dev_size, "pos":pos_subtasks_dev_size, "neg":neg_subtasks_dev_size}
	model_config["test_data"] = {"size":total_test_size, "pos":pos_subtasks_test_size, "neg":neg_subtasks_test_size}
	
	# Extract subtasks data for dev and test
	dev_subtasks_data = split_data_based_on_subtasks(dev_data, model.subtasks)
	test_subtasks_data = split_data_based_on_subtasks(test_data, model.subtasks)

	# Load the instances into pytorch dataset
	train_dataset = COVID19TaskDataset(train_data)
	dev_dataset = COVID19TaskDataset(dev_data)
	test_dataset = COVID19TaskDataset(test_data)
	logging.info("Loaded the datasets into Pytorch datasets")

	tokenize_collator = TokenizeCollator(tokenizer, model.subtasks, entity_start_token_id)
	train_dataloader = DataLoader(train_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=tokenize_collator)
	dev_dataloader = DataLoader(dev_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=tokenize_collator)
	test_dataloader = DataLoader(test_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=tokenize_collator)
	logging.info("Created train and test dataloaders with batch aggregation")

	# Only retrain if needed
	if args.retrain:
		##################################################################################################
		# NOTE: Training Tutorial Reference
		# https://mccormickml.com/2019/07/22/BERT-fine-tuning/#41-bertforsequenceclassification	
		##################################################################################################

		# Create an optimizer training schedule for the BERT text classification model
		# NOTE: AdamW is a class from the huggingface library (as opposed to pytorch) 
		# I believe the 'W' stands for 'Weight Decay fix"
		# Recommended Schedule for BERT fine-tuning as per the paper
		# Batch size: 16, 32
		# Learning rate (Adam): 5e-5, 3e-5, 2e-5
		# Number of epochs: 2, 3, 4
		optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
		logging.info("Created model optimizer")
		# Number of training epochs. The BERT authors recommend between 2 and 4. 
		# We chose to run for 4, but we'll see later that this may be over-fitting the
		# training data.
		epochs = args.n_epochs

		# Total number of training steps is [number of batches] x [number of epochs]. 
		# (Note that this is not the same as the number of training samples).
		total_steps = len(train_dataloader) * epochs

		# Create the learning rate scheduler.
		# NOTE: num_warmup_steps = 0 is the Default value in run_glue.py
		scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
		# We'll store a number of quantities such as training and validation loss, 
		# validation accuracy, and timings.
		training_stats = []

		logging.info(f"Initiating training loop for {args.n_epochs} epochs...")
		# Measure the total training time for the whole run.
		total_start_time = time.time()

		# Find the accumulation steps
		accumulation_steps = args.batch_size/POSSIBLE_BATCH_SIZE

		# Loss trajectory for epochs
		epoch_train_loss = list()
		# Dev validation trajectory
		dev_subtasks_validation_statistics = {subtask: list() for subtask in model.subtasks}
		for epoch in range(epochs):
			pbar = tqdm(train_dataloader)
			logging.info(f"Initiating Epoch {epoch+1}:")
			# Reset the total loss for each epoch.
			total_train_loss = 0
			train_loss_trajectory = list()

			# Reset timer for each epoch
			start_time = time.time()
			model.train()

			dev_log_frequency = 5
			n_steps = len(train_dataloader)
			dev_steps = int(n_steps / dev_log_frequency)
			for step, batch in enumerate(pbar):
				# Upload labels of each subtask to device
				for subtask in model.subtasks:
					subtask_labels = batch["gold_labels"][subtask]
					subtask_labels = subtask_labels.to(device)
					# print("HAHAHAHAH:", subtask_labels.is_cuda)
					batch["gold_labels"][subtask] = subtask_labels
					# print("HAHAHAHAH:", batch["gold_labels"][subtask].is_cuda)
				# Forward
				input_dict = {"input_ids": batch["input_ids"].to(device), "entity_start_positions": batch["entity_start_positions"].to(device), "labels": batch["gold_labels"]}

				input_ids = batch["input_ids"]
				entity_start_positions = batch["entity_start_positions"]
				gold_labels = batch["gold_labels"]
				batch_data = batch["batch_data"]
				loss, logits = model(**input_dict)
				# loss = loss / accumulation_steps
				# Accumulate loss
				total_train_loss += loss.item()

				# Backward: compute gradients
				loss.backward()
				
				if (step + 1) % accumulation_steps == 0:
					
					# Calculate elapsed time in minutes and print loss on the tqdm bar
					elapsed = format_time(time.time() - start_time)
					avg_train_loss = total_train_loss/(step+1)
					# keep track of changing avg_train_loss
					train_loss_trajectory.append(avg_train_loss)
					pbar.set_description(f"Epoch:{epoch+1}|Batch:{step}/{len(train_dataloader)}|Time:{elapsed}|Avg. Loss:{avg_train_loss:.4f}|Loss:{loss.item():.4f}")

					# Clip the norm of the gradients to 1.0.
					# This is to help prevent the "exploding gradients" problem.
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

					# Update parameters
					optimizer.step()

					# Clean the model's previous gradients
					model.zero_grad()                           # Reset gradients tensors

					# Update the learning rate.
					scheduler.step()
					pbar.update()
				if (step + 1) % dev_steps == 0:
					# Perform validation with the model and log the performance
					logging.info("Running Validation...")
					# Put the model in evaluation mode--the dropout layers behave differently
					# during evaluation.
					model.eval()
					dev_predicted_labels, dev_prediction_scores, dev_gold_labels = make_predictions_on_dataset(dev_dataloader, model, device, args.task + "_dev", True)
					for subtask in model.subtasks:
						dev_subtask_data = dev_subtasks_data[subtask]
						dev_subtask_prediction_scores = dev_prediction_scores[subtask]
						dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN = get_TP_FP_FN(dev_subtask_data, dev_subtask_prediction_scores)
						logging.info(f"Subtask:{subtask:>15}\tN={dev_TP + dev_FN}\tF1={dev_F1}\tP={dev_P}\tR={dev_R}\tTP={dev_TP}\tFP={dev_FP}\tFN={dev_FN}")
						dev_subtasks_validation_statistics[subtask].append((epoch + 1, step + 1, dev_TP + dev_FN, dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN))

					# logging.info("DEBUG:Validation on Test")
					# dev_predicted_labels, dev_prediction_scores, dev_gold_labels = make_predictions_on_dataset(test_dataloader, model, device, args.task + "_dev", True)
					# for subtask in model.subtasks:
					# 	dev_subtask_data = test_subtasks_data[subtask]
					# 	dev_subtask_prediction_scores = dev_prediction_scores[subtask]
					# 	dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN = get_TP_FP_FN(dev_subtask_data, dev_subtask_prediction_scores)
					# 	logging.info(f"Subtask:{subtask:>15}\tN={dev_TP + dev_FN}\tF1={dev_F1}\tP={dev_P}\tR={dev_R}\tTP={dev_TP}\tFP={dev_FP}\tFN={dev_FN}")
					# 	dev_subtasks_validation_statistics[subtask].append((epoch + 1, step + 1, dev_TP + dev_FN, dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN))
					# Put the model back in train setting
					model.train()

			# Calculate the average loss over all of the batches.
			avg_train_loss = total_train_loss / len(train_dataloader)
			
			training_time = format_time(time.time() - start_time)

			# Record all statistics from this epoch.
			training_stats.append({
					'epoch': epoch + 1,
					'Training Loss': avg_train_loss,
					'Training Time': training_time})

			# Save the loss trajectory
			epoch_train_loss.append(train_loss_trajectory)

		logging.info(f"Training complete with total Train time:{format_time(time.time()- total_start_time)}")
		log_list(training_stats)
		
		# Save the model and the Tokenizer here:
		logging.info(f"Saving the model and tokenizer in {args.save_directory}")
		model.save_pretrained(args.save_directory)
		# Save each subtask classifiers weights to individual state dicts
		for subtask, classifier in model.classifiers.items():
			classifier_save_file = os.path.join(args.save_directory, f"{subtask}_classifier.bin")
			logging.info(f"Saving the model's {subtask} classifier weights at {classifier_save_file}")
			torch.save(classifier.state_dict(), classifier_save_file)
		tokenizer.save_pretrained(args.save_directory)

		# Plot the train loss trajectory in a plot
		train_loss_trajectory_plot_file = os.path.join(args.output_dir, "train_loss_trajectory.png")
		logging.info(f"Saving the Train loss trajectory at {train_loss_trajectory_plot_file}")
		plot_train_loss(epoch_train_loss, train_loss_trajectory_plot_file)

		# TODO: Plot the validation performance
		# Save dev_subtasks_validation_statistics
	else:
		logging.info("No training needed. Directly going to evaluation!")

	# Save the model name in the model_config file
	model_config["model"] = "MultiTaskBertForCovidEntityClassification"
	model_config["epochs"] = args.n_epochs

	# Find best threshold for each subtask based on dev set performance
	thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	test_predicted_labels, test_prediction_scores, test_gold_labels = make_predictions_on_dataset(test_dataloader, model, device, args.task, True)
	dev_predicted_labels, dev_prediction_scores, dev_gold_labels = make_predictions_on_dataset(dev_dataloader, model, device, args.task + "_dev", True)

	best_test_thresholds = {subtask: 0.5 for subtask in model.subtasks}
	best_dev_thresholds = {subtask: 0.5 for subtask in model.subtasks}
	best_test_F1s = {subtask: 0.0 for subtask in model.subtasks}
	best_dev_F1s = {subtask: 0.0 for subtask in model.subtasks}
	test_subtasks_t_F1_P_Rs = {subtask: list() for subtask in model.subtasks}
	dev_subtasks_t_F1_P_Rs = {subtask: list() for subtask in model.subtasks}
	# for subtask in model.subtasks:
	# 	test_subtask_data = test_subtasks_data[subtask]
	# 	test_subtask_prediction_scores = test_prediction_scores[subtask]
	# 	for t in thresholds:
	# 		test_F1, test_P, test_R, test_TP, test_FP, test_FN = get_TP_FP_FN(test_subtask_data, test_subtask_prediction_scores, THRESHOLD=t)
	# 		test_subtasks_t_F1_P_Rs[subtask].append((t, test_F1, test_P, test_R, test_TP + test_FN, test_TP, test_FP, test_FN))
	# 		if test_F1 > best_test_F1s[subtask]:
	# 			best_test_thresholds[subtask] = t
	# 			best_test_F1s[subtask] = test_F1

	# 	logging.info(f"Subtask:{subtask:>15}")
	# 	log_list(test_subtasks_t_F1_P_Rs[subtask])
	# 	logging.info(f"Best Test Threshold for subtask: {best_test_thresholds[subtask]}\t Best test F1: {best_test_F1s[subtask]}")

	for subtask in model.subtasks:	
		dev_subtask_data = dev_subtasks_data[subtask]
		dev_subtask_prediction_scores = dev_prediction_scores[subtask]
		for t in thresholds:
			dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN = get_TP_FP_FN(dev_subtask_data, dev_subtask_prediction_scores, THRESHOLD=t)
			dev_subtasks_t_F1_P_Rs[subtask].append((t, dev_F1, dev_P, dev_R, dev_TP + dev_FN, dev_TP, dev_FP, dev_FN))
			if dev_F1 > best_dev_F1s[subtask]:
				best_dev_thresholds[subtask] = t
				best_dev_F1s[subtask] = dev_F1

		logging.info(f"Subtask:{subtask:>15}")
		log_list(dev_subtasks_t_F1_P_Rs[subtask])
		logging.info(f"Best Dev Threshold for subtask: {best_dev_thresholds[subtask]}\t Best dev F1: {best_dev_F1s[subtask]}")

	# Save the best dev threshold and dev_F1 in results dict
	results["best_dev_threshold"] = best_dev_thresholds
	results["best_dev_F1s"] = best_dev_F1s
	results["dev_t_F1_P_Rs"] = dev_subtasks_t_F1_P_Rs

	# Evaluate on Test
	logging.info("Testing on test dataset")
	# test_predicted_labels, test_prediction_scores, test_gold_labels = make_predictions_on_dataset(test_dataloader, model, device, args.task)

	predicted_labels, prediction_scores, gold_labels = make_predictions_on_dataset(test_dataloader, model, device, args.task)
	
	# Test 
	for subtask in model.subtasks:
		logging.info(f"Testing the trained classifier on subtask: {subtask}")
		# print(len(test_dataloader))
		# print(len(prediction_scores[subtask]))
		# print(len(test_subtasks_data[subtask]))
		results[subtask] = dict()
		cm = metrics.confusion_matrix(gold_labels[subtask], predicted_labels[subtask])
		classification_report = metrics.classification_report(gold_labels[subtask], predicted_labels[subtask], output_dict=True)
		logging.info(cm)
		logging.info(metrics.classification_report(gold_labels[subtask], predicted_labels[subtask]))
		results[subtask]["CM"] = cm.tolist()			# Storing it as list of lists instead of numpy.ndarray
		results[subtask]["Classification Report"] = classification_report

		# SQuAD style EM and F1 evaluation for all test cases and for positive test cases (i.e. for cases where annotators had a gold annotation)
		EM_score, F1_score, total = get_raw_scores(test_subtasks_data[subtask], prediction_scores[subtask])
		logging.info("Word overlap based SQuAD evaluation style metrics:")
		logging.info(f"Total number of cases: {total}")
		logging.info(f"EM_score: {EM_score}")
		logging.info(f"F1_score: {F1_score}")
		results[subtask]["SQuAD_EM"] = EM_score
		results[subtask]["SQuAD_F1"] = F1_score
		results[subtask]["SQuAD_total"] = total
		pos_EM_score, pos_F1_score, pos_total = get_raw_scores(test_subtasks_data[subtask], prediction_scores[subtask], positive_only=True)
		logging.info(f"Total number of Positive cases: {pos_total}")
		logging.info(f"Pos. EM_score: {pos_EM_score}")
		logging.info(f"Pos. F1_score: {pos_F1_score}")
		results[subtask]["SQuAD_Pos. EM"] = pos_EM_score
		results[subtask]["SQuAD_Pos. F1"] = pos_F1_score
		results[subtask]["SQuAD_Pos. EM_F1_total"] = pos_total

		# New evaluation suggested by Alan
		F1, P, R, TP, FP, FN = get_TP_FP_FN(test_subtasks_data[subtask], prediction_scores[subtask], THRESHOLD=best_dev_thresholds[subtask])
		logging.info("New evaluation scores:")
		logging.info(f"F1: {F1}")
		logging.info(f"Precision: {P}")
		logging.info(f"Recall: {R}")
		logging.info(f"True Positive: {TP}")
		logging.info(f"False Positive: {FP}")
		logging.info(f"False Negative: {FN}")
		results[subtask]["F1"] = F1
		results[subtask]["P"] = P
		results[subtask]["R"] = R
		results[subtask]["TP"] = TP
		results[subtask]["FP"] = FP
		results[subtask]["FN"] = FN
		N = TP + FN
		results[subtask]["N"] = N

		# # Top predictions in the Test case
		# prediction_scores[subtask] = np.array(prediction_scores[subtask])
		# sorted_prediction_ids = np.argsort(-prediction_scores[subtask])
		# K = 200
		# logging.info("Top {} predictions:".format(K))
		# logging.info("\t".join(["Tweet", "BERT model input", "candidate chunk", "prediction score", "predicted label", "gold label", "gold chunks"]))
		# for i in range(K):
		# 	instance_id = sorted_prediction_ids[i]
		# 	# text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
		# 	tweet = test_subtasks_data[subtask][instance_id][0].replace("\n", " ")
		# 	chunk = test_subtasks_data[subtask][instance_id][1]
		# 	tokenized_tweet_with_masked_chunk = test_subtasks_data[subtask][instance_id][6]
		# 	if chunk in ["AUTHOR OF THE TWEET", "NEAR AUTHOR OF THE TWEET"]:
		# 		# First element of the text will be considered as AUTHOR OF THE TWEET or NEAR AUTHOR OF THE TWEET
		# 		bert_model_input_text = tokenized_tweet_with_masked_chunk.replace(Q_TOKEN, "<E> </E>")
		# 		# print(tokenized_tweet_with_masked_chunk)
		# 		# print(bert_model_input_text)
		# 		# exit()
		# 	else:
		# 		bert_model_input_text = tokenized_tweet_with_masked_chunk.replace(Q_TOKEN, "<E> " + chunk + " </E>")
		# 	list_to_print = [tweet, bert_model_input_text, chunk, str(prediction_scores[subtask][instance_id]), str(predicted_labels[subtask][instance_id]), str(test_subtasks_data[subtask][instance_id][-1]), str(test_subtasks_data[subtask][instance_id][-2])]
		# 	logging.info("\t".join(list_to_print))
	
	# Save model_config and results
	model_config_file = os.path.join(args.output_dir, "model_config.json")
	results_file = os.path.join(args.output_dir, "results.json")
	logging.info(f"Saving model config at {model_config_file}")
	save_in_json(model_config, model_config_file)
	logging.info(f"Saving results at {results_file}")
	save_in_json(results, results_file)

	# if args.save_figure_file:
	# 	# Plot the precision recall curve
	# 	disp = plot_precision_recall_curve(lr, test_X, test_Y)
	# 	disp.ax_.set_title('2-class Precision-Recall curve')
	# 	disp.ax_.figure.savefig(args.save_figure_file)
if __name__ == '__main__':
	main()