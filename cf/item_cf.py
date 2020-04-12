#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2020 gmlyytt@outlook.com. All Rights Reserved
#
########################################################################
"""
File: item_cf.py
Author: liyang(gmlyytt@outlook.com)
Date: 2020/04/12 19:56:49
"""

import numpy as np

# 1. 构建item被用户喜欢表
# 2. 计算item的两两相似度，从而记录每个item的最近的K个邻居
# 3. 构建user -> [item1, item2, ..., itemN]的表
# 4. 选取[item1, item2, ..., itemN]与每个item的最近的K个邻居的交集，计算user与item的关联程度
# 5. 选择前P个item