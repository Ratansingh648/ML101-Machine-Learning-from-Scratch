#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:09:24 2020

@author: ratansingh648
"""

import numpy as np

def naive(data):
    data = np.array(data)
    assert len(data) > 1 , "Following data is not a series."    
    return data[-1]