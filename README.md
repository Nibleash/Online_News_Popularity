# STUDY PROJECT REPORT : Online News Popularity 📰💫
[![forthebadge](http://forthebadge.com/images/badges/built-with-love.svg)](http://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](http://forthebadge.com)

⚠️ Feel free to have a look at our **DETAILLED REPORT**, done with Streamlit, just **[HERE](https://demydebesnews.herokuapp.com/)** ⚠️

*K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.*

## Overview 🗺️❓

Nowadays, Blogging and writing articles is **a major growing market on the Internet**. Thanks to easy-to-use tools and friendly tutorials, everyone can be an inspiring writer and develop and share their thoughts on any given subject. Unfortunately, **most articles go under the radar** as the readers are drowned in too many choices. It is impossible to read the 4 million blog posts every day on the internet. To **create and maintain an audience**, every writer elaborates some techniques based on their experiences and knowledge. Our goal, here, is to **help them confirm or invalidate** their thoughts with the use of Visualization, Machine Learning and Deep Learning.

This is our final Project for the "Python for Data Analysis" course. With **Hugo DEBES** we worked on the Online News Popularity dataset provided by the UCI Machine Learning Repository - [Online News Popularity dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip). 


<p align="center">
  <img width="460" height="250" src="https://vl-media.fr/wp-content/uploads/2017/10/Good-news-750x400.jpg">
</p>

There are **39797 instances and 61 attributes** (58 predictive attributes, 2 non-predictive, and 1 goal field) in the database. The articles were published by [Mashable](www.mashable.com) and their content as the rights to reproduce it belongs to them. Hence, this dataset does not share the original content but some statistics associated with it. The original content be publicly accessed and retrieved using the provided urls.

---

A previous research paper was published explaining how the dataset was built and the work that has already been done on it. Just [check](https://repositorium.sdum.uminho.pt/bitstream/1822/39169/1/main.pdf)!


## What do you need ? 📚🐍

This project uses :

<img title="Python" alt="python" width="40px" src="https://img.icons8.com/color/32/000000/python--v1.png">|<img title="Colab" alt="Colab" width="40px" src="https://colab.research.google.com/img/colab_favicon_256px.png">|
|--|--|

To download the dataset ir get more info about it, click on this [link](https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip).

For this project we will mainly use the following libraries :
```python
import pandas as pd
import numpy as np
import time
import plotly.express as px
import matplotlib.ticker as ticker
import seaborn as sns

import pickle
!pip install cchardet
import cchardet
from bs4 import BeautifulSoup, SoupStrainer
import requests
import re

from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb
import sklearn.ensemble as sk_en
import sklearn.linear_model as sk_lm
import tensorflow as tf
from tensorflow import keras
import sklearn.metrics as sklm
from collections import defaultdict
import importlib
import sklearn
from sklearn import svm
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")
```

## Key Topics 🔍💡
  
To carry out the study of our dataset, we splitted our project into different points : 

* Model predictions - Online News dataset
* Librairies
* Load Data & Overall Overview
* Preprocessing
  * Web Scrapping Authors
* Data Visualization & Exploration
  * Shares Distribution
    * Original Scale
    * Logarithmic Scale
  * What does make an article successful on Mashable ?
    * Should we use our emotions or stick only to the facts ?
    * Should we transmit positive or negative news ?
    * On what topics should you write ?
    * Brief News or Detailed Explanations ?
    * The length is all about the subject
    * What about vocabulary ?
    * Should you include Images ?
    * When should you publish ?
* Model Training & Performance Evaluation
  * Logistic Regression
  * Support Vector Classifier
  * Random Forest Classifier
  * Light Gradient Boosted Machine
  * Neural Network
* Performance Comparisons 🥇
* Regression Exploration with MLR 🔥

---

Throughout this study we wanted to reply to this question : **Is an online article likely to be popular ?**.
In order to reply to this question, we preprocessed the dataset (treat outliers, features engineering, scaling) and performed multiple Machine Learning models. We manage to improve the previous results obtained on the research paper reaching **67% of accuracy and 70% of F1-Score with LGBM**.

## About the project 🤝🛡️

This project was realized with **Hugo DEBES**, a Data & AI Student at ESILV engineering school.
<p align="left">
</p>

## Helpful Links

* [API - STUDY REPORT](https://demydebesnews.herokuapp.com/)
* [UCI Dataset source](https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip)
* [Dataset Building & Previous researchs](https://repositorium.sdum.uminho.pt/bitstream/1822/39169/1/main.pdf)
