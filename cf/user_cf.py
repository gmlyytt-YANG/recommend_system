#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2020 gmlyytt@outlook.com. All Rights Reserved
#
########################################################################
"""
File: user_cf.py
Author: liyang(gmlyytt@outlook.com)
Date: 2020/04/12 09:35:40
"""

import numpy as np
import os


def read_file(file_path):
    """read file

    :param file_path:
    :return: data_list:
    """
    if not os.path.exists(file_path):
        print("there is no such file")
        exit(-1)

    fn = open(file_path, "r", encoding="utf-8")

    data_list = []
    for line in fn:
        data = line.strip().split("::")
        data_list.append(data)

    return data_list

class DataUnit:
    """Self define sort unit."""

    def __init__(self, unit, score):
        self.unit = unit
        self.score = score

    def __lt__(self, other):
        return self.score < other.score


class UserCF:
    """UserCF Class Definition.
    
    :param movies: movies meta info([index]::[Album]::[Category])
    :param ratings: ratings info([user_id]::[item_id]::[rate]::[number])
    :param neighbors_num: the number of neighbors of input user.
    :param recommend_num: the number of recommend items of input user.
    """

    def __init__(self, movies, ratings, neighbors_num, recommend_num):
        self.movies = movies
        self.ratings = ratings
        self.neighbors_num = neighbors_num
        self.recommend_num = recommend_num

    def format_rating(self):
        """generate user->[[item1, format_rate1], [item2, format_rate2], ..., [itemN, format_rateN]]
        and item->[user1, user2, ..., userM]
        """
        self.item_user_dict = {}
        self.user_item_rating_dict = {}

        for base_info in self.ratings:
            if len(base_info) < 4:
                continue
            user = base_info[0]
            item = base_info[1]
            rate = base_info[2]

            if user not in self.user_item_rating_dict:
                self.user_item_rating_dict[user] = [[item, float(rate) / 5]]
            else:
                self.user_item_rating_dict[user].append([item, float(rate) / 5])

            if item not in self.item_user_dict:
                self.item_user_dict[item] = [user]
            else:
                self.item_user_dict[item].append(user)

    def format_user_dict(self, user_id_src, user_id_dst):
        """Generate item_rating_info_list.
        When item of self.user_item_rating_dict[user_id_dst] in self.user_item_rating_dict[user_id_src],
        item_rating_info_list[item] = [rate_src, rate_dst]

        Whne item of self.user_item_rating_dict[user_id_dst] not in self.user_item_rating_dict[user_id_src],
        item_rating_info_list[item] = [0, rate_dst]

        Whne item of self.user_item_rating_dict[user_id_src] not in self.user_item_rating_dict[user_id_dst],
        item_rating_info_list[item] = [rate_src, 0]

        :param user_id_src: 
        :param user_id_dst:

        :return: item_rating_info_list:
        """
        item_rating_info_list = {}

        for item, rate in self.user_item_rating_dict[user_id_src]:
            item_rating_info_list[item] = [rate, 0]

        for item, rate in self.user_item_rating_dict[user_id_dst]:
            if item not in item_rating_info_list:
                item_rating_info_list[item] = [0, rate]
            else:
                item_rating_info_list[item][1] = rate

        return item_rating_info_list

    def get_cos_distance(self, user_id_src, user_id_dst):
        """Get cos distance of user_id_src and user_id_dst.
        
        :param user_id_src: 
        :param user_id_dst:

        :return: cos_distance of user_id_src and user_id_dst
        """
        item_rating_info_list = self.format_user_dict(user_id_src, user_id_dst)
        src_square = 0.0
        dst_square = 0.0
        src_dst_info = 0.0

        for item in item_rating_info_list:
            src_square += item_rating_info_list[item][0] ** 2
            dst_square += item_rating_info_list[item][1] ** 2
            src_dst_info += item_rating_info_list[item][0] * item_rating_info_list[item][1]

        if src_dst_info == 0.0:
            return 0.0

        return src_dst_info / np.sqrt(src_square + dst_square)

    def get_nearest_neighbors(self, user_id):
        """Get self.neighbors_num nearest users of input user_id.
        
        :param user_id:
        """
        neighbors = []

        if user_id not in self.user_item_rating_dict:
            print("user_id not in self.user_item_rating_dict")
            exit(-1)

        for item, rate in self.user_item_rating_dict[user_id]:
            if item not in self.item_user_dict:
                continue
            for candidate_user in self.item_user_dict[item]:
                if candidate_user != user_id and candidate_user not in neighbors:
                    neighbors.append(candidate_user)

        self.neighbors = []
        for candidate_user in neighbors:
            if candidate_user not in self.user_item_rating_dict:
                continue
            cos_distance = self.get_cos_distance(user_id, candidate_user)
            self.neighbors.append(DataUnit(candidate_user, cos_distance))

        self.neighbors.sort(reverse=True)
        self.neighbors = self.neighbors[:self.neighbors_num]

    def get_recommend_list(self):
        """Get recommend list of user_id.

        :return recommend_list:
        """
        recommend_list = []
        for neighbor in self.neighbors:
            item_rating_list = self.user_item_rating_dict[neighbor.unit]
            for item, rate in item_rating_list:
                recommend_list.append(DataUnit(item, rate))

        recommend_list.sort(reverse=True)
        recommend_list = [_.unit for _ in recommend_list][:self.recommend_num]

        return recommend_list

    def recommend(self, user_id):
        """Entry of recommend.

        :param user_id:
        :return: recommend_list:
        """
        self.format_rating()
        self.get_nearest_neighbors(user_id)
        recommend_list = self.get_recommend_list()

        return recommend_list

    def get_precision(self, user_id, recommend_list):
        """Get precision of this recommend algorithm.

        :param user_id:
        :param recommend_list:
        """
        user_real_like_items = [_[0] for _ in self.user_item_rating_dict[user_id]]

        count = 0
        for item in recommend_list:
            if item in user_real_like_items:
                count += 1
        print("precision is: {}".format(float(count) / len(recommend_list)))


if __name__ == "__main__":
    movies = read_file("../data/MovieLens/movies.dat")
    ratings = read_file("../data/MovieLens/ratings_top_100000.dat")

    neighbors_num = 7
    recommend_num = 10

    user_id = "6"
    user_cf = UserCF(movies, ratings, neighbors_num, recommend_num)
    recommend_list = user_cf.recommend(user_id)

    user_cf.get_precision(user_id, recommend_list)
