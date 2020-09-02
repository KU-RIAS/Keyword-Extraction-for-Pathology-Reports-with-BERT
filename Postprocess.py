# coding: utf-8
import numpy as np

def tensor_to_str(tensor, tokenizer):
    tensor_ids = tensor.to('cpu').numpy()
    text = np.array(tokenizer.convert_ids_to_tokens(tensor_ids))
    return text

def untokenizing(tokenized_text):
    report = ""
    for ss in tokenized_text:
        if not ss.startswith('##'):
            report += ' ' + ss
        else:
            report += ss[2:]
    return report.strip()

def get_keyword(prediction, textlist, tag):
    prediction = np.array(prediction)
    textlist = np.array(textlist)
    idx = prediction==tag
    keyword_token = textlist[idx]
    keyword = untokenizing(keyword_token)
    return keyword

def merging(tokenized_text, pred):
    tokens = []
    proba = []
    for i, ss in enumerate(tokenized_text):
        if not ss.startswith('##'):
            tokens.append(ss)
            proba.append(pred[i,:])
        else:
            tokens[-1] += ss[2:]
            proba[-1] += list(pred[i,:])
    proba = np.array(proba)
    return tokens, proba
