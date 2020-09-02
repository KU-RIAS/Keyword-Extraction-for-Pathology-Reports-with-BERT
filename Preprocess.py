# coding: utf-8
import re
import csv
import numpy as np

def pre_process_str(text):
    text = text.lower()
    text = re.sub("<!--?.*?-->", "", text)
    text = re.sub("([^a-zA-Z0-9\u3131-\u3163\uac00-\ud7a3\n])+", " ", text)
    return text

def multisplit(text):
    s_text = text.split('\n')
    s_text_ = [x.strip() for x in s_text]
    m_text = '\n'.join(s_text_)
    m_text_ = re.sub(r'\n\n+', '\n\n', m_text)
    ss = m_text_.split('\n\n')
    ss = [x for x in ss if x]
    return ss

def pre_process(reports_raw, keywords_raw):
    labels, labels_raw = [], []
    text, text_raw = [], []
    for i, keywords in enumerate(keywords_raw):
        reports = multisplit(reports_raw[i])
        for j, report in enumerate(reports):
            report_pp = pre_process_str(report)
            text_raw.append(report)
            text.append(report_pp)
            token_s = report_pp.split()
            z = ['XX'] * len(token_s)
            labels_raw.append(keywords[j*3:(j+1)*3])
            for k in range(3):
                if k == 0: label = 'OR'
                elif k == 1: label = 'PR'
                elif k == 2: label = 'PA'
                keyword = pre_process_str(keywords[j * 3 + k])
                if keyword == '': break
                keyword_list = keyword.split()
                if keyword in report_pp:
                    idxl = len(keyword_list)
                    for ii in range(len(token_s) - idxl + 1):
                        tf = True
                        for jj in range(idxl):
                            tf = tf and (token_s[ii + jj] == keyword_list[jj])
                        if tf: break
                    for jj in range(idxl):
                        z[ii + jj] = label
                        token_s[ii + jj] = ['##']
                else:
                    for _, kk in enumerate(keyword_list):
                        try:
                            idx = token_s.index(kk)
                        except ValueError:
                            break
                        z[idx] = label
                        token_s[idx] = '##'
            labels.append(z)
    return text, labels, text_raw, labels_raw

def tokenizing(reports, label, tokenizer):
    tokenized_texts, corresponding_labels = [], []
    for i,s in enumerate(reports):
        tokenized_text, corresponding_label = [], []
        s_ = s.split()
        for j,ss in enumerate(s_):
            token=tokenizer.tokenize(ss)
            if len(token)==1:
                tokenized_text.append(token[0])
                corresponding_label.append(label[i][j])
            elif len(token)>1:
                 for token_ in token:
                    tokenized_text.append(token_)
                    corresponding_label.append(label[i][j])
        tokenized_texts.append(tokenized_text)
        corresponding_labels.append(corresponding_label)
    return tokenized_texts, corresponding_labels