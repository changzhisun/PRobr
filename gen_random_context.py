#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 21/01/05 15:02:12

@author: Changzhi Sun
"""
import sys
import numpy as np
import random
import os
import json

def read_jsonl(input_file):
    """Reads a tab separated value file."""
    records = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            records.append(json.loads(line))
        return records

def count_lines(input_file):
    ct = 0
    with open(input_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            ct += 1
    return ct

def count_question_num(input_file):
    ct = 0
    with open(input_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            ct += len(json.loads(line)['questions'])
    return ct

if __name__ == "__main__":
    percent = 0.1
    in_data_dir = "./data/depth-5"
    out_data_dir = "./data/depth-5-random-context-%d" % (percent * 100)
    random.seed(42)
    num_lines = count_lines(os.path.join(in_data_dir, "train.jsonl"))
    permutation = list(range(num_lines))
    random.shuffle(permutation)
    permutation = permutation[: int(percent * num_lines)]
    permutation_set = set(permutation)
    print(len(permutation_set))

    if os.path.exists(out_data_dir):
        os.system("rm -rf %s" % out_data_dir)
    os.system("mkdir %s" % out_data_dir)
    os.system("cp %s/*.jsonl %s" % (in_data_dir, out_data_dir))
    os.system("rm %s/train.jsonl" % (out_data_dir))

    fin = open(os.path.join(in_data_dir, "train.jsonl"), "r", encoding="utf-8-sig")
    fout = open(os.path.join(out_data_dir, "train.jsonl"), "w", encoding="utf-8-sig")
    i = 0
    for line in fin:
        record = json.loads(line)
        for question in record['questions']:
            if i in permutation_set:
                question['masked'] = False
            else:
                question['masked'] = True
        i += 1
        print(json.dumps(record), file=fout)
    fin.close()
    fout.close()
