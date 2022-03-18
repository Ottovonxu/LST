import collections
import os
import sys
import math
import random
import numpy as np
import numpy.random
import scipy as sp
import scipy.stats

from clsh import pyLSH
import torch

class LSH:
    def __init__(self, func_, MAX_SIZE_, K_, L_, threads_=4):
        self.func = func_
        self.K = K_
        self.L = L_
        self.lsh_ = pyLSH(MAX_SIZE_, self.K, self.L, threads_)

    def insert(self, item_id, item):
        fp = self.func.hash(item).cpu().numpy()
        self.lsh_.insert(fp, item_id)

    def insert_multi(self, items, N):
        fp = self.func.hash(items).cpu().numpy()
        self.lsh_.insert_multi(fp, N)

    def insert_multi_label(self, items, labels, N):
        fp = self.func.hash(items).cpu().numpy()
        labels_cpu = labels.int().cpu().numpy()
        self.lsh_.insert_multi_label(fp, labels_cpu, N)

    def union(self, item):
        fp = self.func.hash(item).cpu().numpy()
        return self.lsh_.query(fp)

    def query_multi(self, items, N):
        fp = self.func.hash(items, transpose=True).cpu().numpy().copy(order='C')
        return self.lsh_.query_multi(fp, N)

    def query_multi_label(self, items, N, c):
        fp = self.func.hash(items, transpose=True).cpu().numpy()
        return self.lsh_.query_multi_label(fp, N, c)

    def query_batch(self, items, result, N, SIZE):
        fp = self.func.hash(items).cpu().numpy()
        self.lsh_.query_batch(fp, result, N, SIZE)

    def clear(self):
        self.lsh_.clear()

    def count(self):
        self.lsh_.count()
