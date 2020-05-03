#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2020 gmlyytt@outlook.com, Inc. All Rights Reserved
#
########################################################################

"""
File: user_cf_v2.py
Author: liyang(gmlyytt@outlook.com)
Date: 2020/04/19 19:34:10
"""

import os
import random
import math
from operator import itemgetter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import numpy as np

class UserCF:
    def __init__(self, user_item_pair_capacity=100000):
        self.n_sim_user = 20
        self.n_rec_movie = 10

        self.train_set = {}
        self.test_set = {}

        self.user_sim_matrix = {}
        self.movie_count = 0

        self.user_item_pair_max_num = user_item_pair_capacity # train+test

    def load_train_test_set(self, file_path, pivot=0.75):
        if not os.path.exists(file_path):
            print("no exists file")
            exit(-1)

        fn = open(file_path, "r")
        user_item_pair_count = 0
        reach_user_item_pair_max_num = False
        user_ctnr = set()
        item_ctnr = set()
        for line in fn:
            if reach_user_item_pair_max_num:
                break
            data = line.strip().split(":")
            if len(data) < 2:
                continue
            user = data[0]
            user_ctnr.add(user)
            item_timestamp_list = data[1].split(";")
            for item_timestamp in item_timestamp_list:
                item_timestamp_elem_list = item_timestamp.split(",")
                if len(item_timestamp_elem_list) < 3:
                    continue
                item = item_timestamp_elem_list[0]
                item_ctnr.add(item)
                if random.random() < pivot:
                    self.train_set.setdefault(user, {})
                    self.train_set[user][item] = 1.0
                else:
                    self.test_set.setdefault(user, {})
                    self.test_set[user][item] = 1.0
                user_item_pair_count += 1
                if user_item_pair_count >= self.user_item_pair_max_num:
                    reach_user_item_pair_max_num = True
                    break

    def calc_user_sim(self):
        movie_popular = {}
        for _, movies in self.train_set.items():
            for movie in movies:
                if movie not in movie_popular:
                    movie_popular.setdefault(movie, 0)
                movie_popular[movie] += 1

        movie_user = {}
        for user, movies in self.train_set.items():
            for movie in movies:
                if movie not in movie_user:
                    movie_user[movie] = set()
                movie_user[movie].add(user)

        self.movie_count = len(movie_user)

        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] += 1.0 / np.log(1.0 + movie_popular[movie])

        for u, related_users in self.user_sim_matrix.items():
            for v, wuv_raw in related_users.items():
                self.user_sim_matrix[u][v] = \
                        wuv_raw / math.sqrt(len(self.train_set[u]) * len(self.train_set[v]))

    def recommend(self, user):
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = {}
        if user not in self.train_set:
            return []
        watched_movies = self.train_set[user]
        if user not in self.user_sim_matrix:
            return []
        user_sim_vec = sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[:K]
        for v, wuv in user_sim_vec:
            for movie, rating in self.train_set[v].items():
                if movie in watched_movies:
                    continue
                rank.setdefault(movie, 0)
                rank[movie] += wuv * rating

        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    def evaluate(self):
        hit = 0
        rec_count = 0
        test_count = 0
        all_rec_movies = set()
        for user in self.train_set:
            test_movies = self.test_set.get(user, {})
            rec_movies = self.recommend(user)

            for movie, w in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)

            rec_count += len(rec_movies)
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print("precision is: {}, recall is {}, coverage is: {}".format(precision, recall, coverage))

        return precision, recall, coverage


if __name__ == "__main__":
    file_path = "xx.data"

    precision_list = []
    recall_list = []
    coverage_list = []
    user_item_pair_capacity_list = []

    for user_item_pair_capacity in range(100000, 1000000, 100000):
        start = time.time()
        user_cf = UserCF(user_item_pair_capacity)
        user_cf.load_train_test_set(file_path)
        user_cf.calc_user_sim()
        precision, recall, coverage = user_cf.evaluate()
        precision_list.append(precision)
        recall_list.append(recall)
        coverage_list.append(coverage)
        user_item_pair_capacity_list.append(user_item_pair_capacity)
        end = time.time()
        time_past = end - start
        print("time_past in user_item_pair_capacity {} is {}".format(user_item_pair_capacity, time_past))
        print("---")

    plt.title("usercf")
    plt.plot(user_item_pair_capacity_list, precision_list, label="precision")
    plt.plot(user_item_pair_capacity_list, recall_list, label="recall")
    plt.plot(user_item_pair_capacity_list, coverage_list, label="coverage")

    plt.legend()
    plt.savefig("./usercf_result.png")
