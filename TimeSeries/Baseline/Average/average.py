#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:09:27 2020

@author: ratansingh648
"""

import numpy as np

def average(data):
    assert len(data) > 1, "Following data is not an series"
    return np.mean(data)