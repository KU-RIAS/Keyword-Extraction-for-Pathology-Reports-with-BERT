# coding: utf-8
import argparse

parser = argparse.ArgumentParser(description='Keyword Extraction for Pathology Reports with BERT')
parser.add_argument('--data', type=str, help='Data', default='sample.csv')
parser.add_argument('--maxlen', type=int, help='Max Length', default=128)
parser.add_argument('--bs', type=int, help='Batch Size', default=16)
parser.add_argument('--lr', type=float, help='Learning Rate', default=2e-5)
parser.add_argument('--epoch', type=int, help='Epochs', default=30)
args = parser.parse_args()

# parameters
filename_read = args.data
MAX_LEN = args.maxlen
batch_size_ = args.bs
learning_rate = args.lr
epochs = args.epoch

import csv
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification
from Preprocess import pre_process_str, multisplit, pre_process, tokenizing
from Postprocess import tensor_to_str, untokenizing, get_keyword, merging

def flat_accuracy(preds, labels, masks):
    mask_flat = masks.flatten()
    pred_flat = np.argmax(preds, axis=2).flatten()[mask_flat==1]
    labels_flat = labels.flatten()[mask_flat==1]
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Read Data
reader = csv.reader(open(filename_read, 'r', encoding='utf-8'))
data=[]
for line in reader:
    data.append(line)

reports_raw = [x[0] for x in data[1:]]
keywords_raw = [x[1:] for x in data[1:]]

# Pre-processing
reports, labels, text_raw, labels_raw = pre_process(reports_raw, keywords_raw)

# Labels
lab2idx = {'OR':0, 'PR':1, 'PA':2, 'XX':3}

# Word-piece tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenizing
tokenized_texts, corresponding_labels = tokenizing(reports, labels, tokenizer)
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
labs = pad_sequences([[lab2idx.get(l) for l in lab] for lab in corresponding_labels],
                     maxlen=MAX_LEN, value=lab2idx["XX"], padding="post",
                     dtype="long", truncating="post")
attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

# Split training/test sets
tr_inputs, val_inputs, tr_labs, val_labs = train_test_split(input_ids, labs, random_state=12345, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=12345, test_size=0.1)
tr_text, val_text, tr_keywords, val_keywords = train_test_split([x for x in text_raw], [x for x in labels_raw],
                                             random_state=12345, test_size=0.1)
# Unique datasets
nptr_keywords = np.array([[pre_process_str(x[0]), pre_process_str(x[1]), pre_process_str(x[2])] for x in tr_keywords])
npts_keywords = np.array([[pre_process_str(x[0]), pre_process_str(x[1]), pre_process_str(x[2])] for x in val_keywords])
print("Training Unique: {} {} {}".format(np.unique(nptr_keywords[:,0]).shape[0], np.unique(nptr_keywords[:,1]).shape[0], np.unique(nptr_keywords[:,2]).shape[0]))
print("Test Unique: {} {} {}".format(np.unique(npts_keywords[:,0]).shape[0], np.unique(npts_keywords[:,1]).shape[0], np.unique(npts_keywords[:,2]).shape[0]))

# Load Data
tr_inputs = torch.tensor(tr_inputs).to(torch.long)
val_inputs = torch.tensor(val_inputs).to(torch.long)
tr_labs = torch.tensor(tr_labs).to(torch.long)
val_labs = torch.tensor(val_labs).to(torch.long)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
train_data = TensorDataset(tr_inputs, tr_masks, tr_labs)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size_)
valid_data = TensorDataset(val_inputs, val_masks, val_labs)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size_)

# model
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(lab2idx))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.cuda() if device.type=='cuda' else model.cpu()

# Fine-tuning parameters
FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

train_loss, valid_loss = [], []
valid_accuracy = []
# Exact Matching
EM = []
max_grad_norm = 1.0
for iter, _ in tqdm(enumerate(range(epochs))):
    # training
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss.backward()
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        model.zero_grad()
    print("Train loss: {}".format(tr_loss / nb_tr_steps))
    train_loss.append(tr_loss / nb_tr_steps)
    # evaluation
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    predictions, true_labels = [], []
    matching = []
    for ii, batch in enumerate(valid_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        masks = b_input_mask.to('cpu').numpy()
        for jj, _ in enumerate(label_ids):
            pred = logits[jj]
            mask = masks[jj]
            if np.sum(mask)==0: continue
            pred_lab = np.argmax(pred, axis=1)
            pred_lab_ = pred_lab[mask==1]
            predictions.append(pred_lab_)
            true_lab = val_labs[ii * batch_size_ + jj].to('cpu').numpy()
            true_lab_ = true_lab[mask==1]
            true_labels.append(true_lab_)
            token_text = tensor_to_str(b_input_ids[jj], tokenizer)
            tkns = token_text[mask==1]
            pred_token, mpred = merging(tkns, pred)
            pred_label = np.argmax(mpred, axis=1)
            specimen = get_keyword(pred_label, pred_token, 0)
            procedure = get_keyword(pred_label, pred_token, 1)
            pathology = get_keyword(pred_label, pred_token, 2)
            true_specimen = get_keyword(true_lab, token_text, 0)
            true_procedure = get_keyword(true_lab, token_text, 1)
            true_pathology = get_keyword(true_lab, token_text, 2)
            matching.append([int(specimen==true_specimen), int(procedure==true_procedure), int(pathology==true_pathology)])
        tmp_eval_accuracy = flat_accuracy(logits, label_ids, masks)
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    valid_loss.append(eval_loss)
    valid_accuracy.append(eval_accuracy / nb_eval_steps)
    matching = np.array(matching)
    EM.append(list(np.average(matching, axis=0)))
    print("Loss: {}".format(eval_loss))
    print("Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    print("Exact Matching: {}".format(list(np.average(matching, axis=0))))
