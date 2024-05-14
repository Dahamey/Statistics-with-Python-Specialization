#!/usr/bin/env python
# coding: utf-8

# #### Hypothesis Testing
# 
# From lecture, we know that hypothesis testing is a critical tool in determing what the value of a parameter could be.
# 
# We know that the basis of our testing has two attributes:
# 
# **Null Hypothesis: $H_0$**
# 
# **Alternative Hypothesis: $H_a$**
# 
# The tests we have discussed in lecture are:
# 
# - One Population Proportion
# - Difference in Population Proportions
# - One Population Mean
# - Difference in Population Means
# 
# In this tutorial, I will introduce some functions that are extremely useful when calculating a t-statistic and p-value for a hypothesis test.
# 
# Let's quickly review the following ways to calculate a test statistic for the tests listed above.
# 
# The equation is:
# 
# $$\frac{Best\ Estimate - Hypothesized\ Estimate}{Standard\ Error\ of\ Estimate}$$ 
# 
# We will use the examples from our lectures and use python functions to streamline our tests.

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats.distributions as dist
import statsmodels.api as sm


# ##### One Population Proportion
# 
# Research Question:
# 
# In previous years 52% of parents believed that electronics and social media was the cause of their teenager’s lack of sleep. Do more parents today believe that their teenager’s lack of sleep is caused due to electronics and social media? 
# 
# **Population**: Parents with a teenager (age 13-18)  
# **Parameter of Interest**: p  
# **Null Hypothesis:** p = 0.52  
# **Alternative Hypthosis:** p > 0.52 (note that this is a one-sided test)
# 
# 1018 Parents
# 
# 56% believe that their teenager’s lack of sleep is caused due to electronics and social media.

# In[2]:


n = 1018
p_null = .52
p_hat = .56

sm.stats.proportions_ztest(
    p_hat * n, 
    n, 
    p_null, 
    alternative='larger', 
    prop_var=0.52
)


# ##### Difference in Population Proportions
# 
# Research Question:
# 
# Is there a significant difference between the population proportions of parents of black children and parents of Hispanic children who report that their child has had some swimming lessons?
# 
# **Populations**: All parents of black children age 6-18 and all parents of Hispanic children age 6-18  
# **Parameter of Interest**: p1 - p2, where p1 = black and p2 = hispanic  
# **Null Hypothesis:** p1 - p2 = 0  
# **Alternative Hypthosis:** p1 - p2 $\neq$ 0  
# 
# 
# 91 out of 247 (36.8%) sampled parents of black children report that their child has had some swimming lessons.
# 
# 120 out of 308 (38.9%) sampled parents of Hispanic children report that their child has had some swimming lessons.

# In[5]:


# sample sizes
n1 = 247
n2 = 308

# number of parents reporting that their child had some swimming lessons
y1 = 91
y2 = 120

# estimates of the population proportions
p1 = round(y1 / n1, 2)
p2 = round(y2 / n2, 2)

# estimate of the combined population proportion
p_hat = (y1 + y2) / (n1 + n2)

# estimate of the variance of the combined population proportion
va = p_hat * (1 - p_hat)

# estimate of the standard error of the combined population proportion
se = np.sqrt(va * (1 / n1 + 1 / n2))

# test statistic and its p-value
test_stat = (p1 - p2) / se
p_value = 2 * dist.norm.cdf(-np.abs(test_stat))

# print the test statistic its p-value
print('Test Statistic')
print(round(test_stat, 2))

print('\nP-Value')
print(round(p_value, 2))


# ##### One Population Mean
# 
# Research Question:
# 
# Is the average cartwheel distance (in inches) for adults more than 80 inches?
# 
# **Population**: All adults  
# **Parameter of Interest**: $\mu$, population mean cartwheel distance  
# **Null Hypothesis:** $\mu$ = 80    
# **Alternative Hypthosis:** $\mu$ > 80
# 
# 25 Adults
# 
# $\mu = 82.46$
# 
# $\sigma = 15.06$

# In[6]:


df = pd.read_csv('data/cartwheel_data.csv')


# In[7]:


n = len(df)
mean = df['CWDistance'].mean()
sd = df['CWDistance'].std()
(n, mean, sd)


# In[8]:


sm.stats.ztest(df['CWDistance'], value=80, alternative='larger')


# ##### Difference in Population Means
# 
# Research Question:
# 
# Considering adults in the NHANES data, do males have a significantly higher mean Body Mass Index than females?
# 
# **Population**: Adults in the NHANES data.  
# **Parameter of Interest**: $\mu_1 - \mu_2$, Body Mass Index.  
# **Null Hypothesis:** $\mu_1 = \mu_2$  
# **Alternative Hypthosis:** $\mu_1 \neq \mu_2$
# 
# 2976 Females 
# $\mu_1 = 29.94$  
# $\sigma_1 = 7.75$  
# 
# 2759 Male Adults  
# $\mu_2 = 28.78$  
# $\sigma_2 = 6.25$  
# 
# $\mu_1 - \mu_2 = 1.16$

# In[9]:


df = pd.read_csv('data/NHANES.csv')


# In[10]:


females = df[df['RIAGENDR'] == 2]
male = df[df['RIAGENDR'] == 1]


# In[11]:


n1 = len(females)
mu1 = females['BMXBMI'].mean()
sd1 = females['BMXBMI'].std()

(n1, mu1, sd1)


# In[12]:


n2 = len(male)
mu2 = male['BMXBMI'].mean()
sd2 = male['BMXBMI'].std()

(n2, mu2, sd2)


# In[13]:


sm.stats.ztest(females['BMXBMI'].dropna(), male['BMXBMI'].dropna())


# #### Assumptions Consistency
# 
# ##### One Population Proportion
# - Sample can be considered a simple random sample
# - Large enough sample size
#     - Confidence Interval: At least 10 of each outcome
#     - Hypothesis Test: At least 10 of each outcome
# 
# ##### Two Population Proportions
# - Samples can be considered two simple random samples
# - Samples can be considered independent of one another
# - Large enough sample sizes ()
#     - Confidence Interval: At least 10 of each outcome
#     - Hypothesis Test: At least 10 of each outcome - Where (the common population proportion estimate)
# 
# ##### One Population Mean
# - Sample can be considered a simple random sample
# - Sample comes from a normally distributed population
#     - This assumption is less critical with a large enough sample size (application of the C.L.T.)
# 
# ##### One Population Mean Difference
# - Sample of differences can be considered a simple random sample
# - Sample of differences comes from a normally distributed population of differences
#     - This assumption is less critical with a large enough sample size (application of the C.L.T.)
# 
# ##### Two Population Means
# - Samples can be considered a simple random samples
# - Samples can be considered independent of one another
# - Samples each come from normally distributed populations
#     - This assumption is less critical with a large enough sample size (application of the C.L.T.)
# - Populations have equal variances – pooled procedure used
#     - If this assumption cannot be made, unpooled procedure used
