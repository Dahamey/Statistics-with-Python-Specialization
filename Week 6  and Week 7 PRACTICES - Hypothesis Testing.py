#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import scipy.stats.distributions as dist
import statsmodels.api as sm


# In[2]:


df = pd.read_csv('data/NHANES.csv')


# In[3]:


mapping = {
    1: 'Yes', 
    2: 'No', 
    7: np.nan, 
    9: np.nan
}

gender_map = {
    1: 'Male',
    2: 'Female'
}

citizen_map = {
    1: 'Yes',
    2: 'No',
    7: np.nan,
    9: np.nan
}

df['SMQ020x'] = df.SMQ020.replace(mapping)
df['RIAGENDRx'] = df.RIAGENDR.replace(gender_map)
df['DMDCITZN'] = df.DMDCITZN.replace(citizen_map)


# #### Hypothesis tests for one proportions
# 
# The most basic hypothesis test may be the one-sample test for a proportion.  This test is used if we have specified a particular value as the null value for the proportion, and we wish to assess if the data are compatible with the true parameter value being equal to this specified value.  One-sample tests are not used very often in practice, because it is not very common that we have a specific fixed value to use for comparison. 
# 
# For illustration, imagine that the rate of lifetime smoking in another country was known to be 40%, and we wished to assess whether the rate of lifetime smoking in the US were different from 40%.  In the following notebook cell, we carry out the (two-sided) one-sample test that the population proportion of smokers is 0.4, and obtain a p-value of 0.43.  This indicates that the NHANES data are compatible with the proportion of (ever) smokers in the US being 40%. 

# In[4]:


x = df.SMQ020x.dropna() == 'Yes'

p = x.mean()
se = np.sqrt(0.4 * 0.6 / len(x))
test_stat = (p - 0.4) / se

p_value = 2 * dist.norm.cdf(-np.abs(test_stat))
test_stat, p_value


# The following cell carries out the same test as performed above using the Statsmodels library.  The results in the first (default) case below are slightly different from the results obtained above because Statsmodels by default uses the sample proportion instead of the null proportion when computing the standard error.  This distinction is rarely consequential, but we can specify that the null proportion should be used to calculate the standard error, and the results agree exactly with what we calculated above.  The first two lines below carry out tests using the normal approximation to the sampling distribution of the test statistic, and the third line below carries uses the exact binomial sampling distribution.  We can see here that the p-values are nearly identical in all three cases. This is expected when the sample size is large, and the proportion is not close to either 0 or 1.

# In[5]:


print(sm.stats.proportions_ztest(x.sum(), len(x), 0.4))
print(sm.stats.proportions_ztest(x.sum(), len(x), 0.4, prop_var=0.4))
print(sm.stats.binom_test(x.sum(), len(x), 0.4))


# #### Hypothesis tests for two proportions
# 
# Comparative tests tend to be used much more frequently than tests comparing one population to a fixed value.  A two-sample test of proportions is used to assess whether the proportion of individuals with some trait differs between two sub-populations.  For example, we can compare the smoking rates between females and males. Since smoking rates vary strongly with age, we do this in the subpopulation of people between 20 and 25 years of age.  In the cell below, we carry out this test without using any libraries, implementing all the test procedures covered elsewhere in the course using Python code.  We find that the smoking rate for men is around 10 percentage points greater than the smoking rate for females, and this difference is statistically significant (the p-value is around 0.01).

# In[6]:


dx = df[['SMQ020x', 'RIDAGEYR', 'RIAGENDRx']].dropna()
dx = dx.loc[(dx.RIDAGEYR >= 20) & (dx.RIDAGEYR <= 25), :]

p = dx \
    .groupby('RIAGENDRx')['SMQ020x'] \
    .agg([lambda z: np.mean(z == 'Yes'), 'size'])

p.columns = ['Smoke', 'N']

p_comb = (dx.SMQ020x == 'Yes').mean()
va = p_comb * (1 - p_comb)
se = np.sqrt(va * (1 / p.N.Female + 1 / p.N.Male))

test_stat = (p.Smoke.Female - p.Smoke.Male) / se
pvalue = 2 * dist.norm.cdf(-np.abs(test_stat))

test_stat, pvalue


# Essentially the same test as above can be conducted by converting the "Yes"/"No" responses to numbers (Yes=1, No=0) and conducting a two-sample t-test, as below:

# In[7]:


mapping = {'Yes': 1, 'No': 0}

dx_females = dx.loc[dx.RIAGENDRx == "Female", 'SMQ020x'].replace(mapping)
dx_males = dx.loc[dx.RIAGENDRx == 'Male', 'SMQ020x'].replace(mapping)

sm.stats.ttest_ind(dx_females, dx_males)


# ##### Hypothesis tests comparing means
# 
# Tests of means are similar in many ways to tests of proportions.  Just as with proportions, for comparing means there are one and two-sample tests, z-tests and t-tests, and one-sided and two-sided tests.  As with tests of proportions, one-sample tests of means are not very common, but we illustrate a one sample test in the cell below.  We compare systolic blood pressure to the fixed value 120 (which is the lower threshold for "pre-hypertension"), and find that the mean is significantly different from 120 (the point estimate of the mean is 126).

# In[8]:


dx = df[['BPXSY1', 'RIDAGEYR', 'RIAGENDRx']].dropna()
dx = dx.loc[(dx.RIDAGEYR >= 40) & (dx.RIDAGEYR <= 50) & (dx.RIAGENDRx == 'Male'), :]
sm.stats.ztest(dx.BPXSY1, value=120)


# In the cell below, we carry out a formal test of the null hypothesis that the mean blood pressure for women between the ages of 50 and 60 is equal to the mean blood pressure of men between the ages of 50 and 60.  The results indicate that while the mean systolic blood pressure for men is slightly greater than that for women (129 mm/Hg versus 128 mm/Hg), this difference is not statistically significant. 
# 
# There are a number of different variants on the two-sample t-test. Two often-encountered variants are the t-test carried out using the t-distribution, and the t-test carried out using the normal approximation to the reference distribution of the test statistic, often called a z-test.  Below we display results from both these testing approaches.  When the sample size is large, the difference between the t-test and z-test is very small.  

# In[9]:


dx = df[['BPXSY1', 'RIDAGEYR', 'RIAGENDRx']].dropna()
dx = dx.loc[(dx.RIDAGEYR >= 50) & (dx.RIDAGEYR <= 60), :]

bpx_female = dx.loc[dx.RIAGENDRx == 'Female', 'BPXSY1']
bpx_male = dx.loc[dx.RIAGENDRx == 'Male', 'BPXSY1']

print(bpx_female.mean(), bpx_male.mean())
print(sm.stats.ztest(bpx_female, bpx_male))
print(sm.stats.ttest_ind(bpx_female, bpx_male))


# Another important aspect of two-sample mean testing is "heteroscedasticity", meaning that the variances within the two groups being compared may be different.  While the goal of the test is to compare the means, the variances play an important role in calibrating the statistics (deciding how big the mean difference needs to be to be declared statistically significant).  In the NHANES data, we see that there are moderate differences between the amount of variation in BMI for females and for males, looking within 10-year age bands.  In every age band, females having greater variation than males.

# In[10]:


dx = df[['BMXBMI', 'RIDAGEYR', 'RIAGENDRx']].dropna()
df['agegrp'] = pd.cut(df.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])
df.groupby(['agegrp', 'RIAGENDRx'])['BMXBMI'].agg(np.std).unstack()


# The standard error of the mean difference (e.g. mean female blood pressure minus mean male blood pressure) can be estimated in at least two different ways.  In the statsmodels library, these approaches are referred to as the "pooled" and the "unequal" approach to estimating the variance.  If the variances are equal (i.e. there is no heteroscedasticity), then there should be little difference between the two approaches.  Even in the presence of moderate heteroscedasticity, as we have here, we can see that the results for the two methods are quite similar.  Below we have a loop that considers each 10-year age band and assesses the evidence for a difference in mean BMI for women and for men.  The results printed in each row of output are the test-statistic and p-value.

# In[11]:


for k, v in df.groupby('agegrp'):
    bmi_female = v.loc[v.RIAGENDRx == 'Female', 'BMXBMI'].dropna()
    bmi_female = sm.stats.DescrStatsW(bmi_female)
    
    bmi_male = v.loc[v.RIAGENDRx == 'Male', 'BMXBMI'].dropna()
    bmi_male = sm.stats.DescrStatsW(bmi_male)
    
    print(k)
    print('pooled: ', sm.stats.CompareMeans(bmi_female, bmi_male).ztest_ind(usevar='pooled'))
    print('unequal: ', sm.stats.CompareMeans(bmi_female, bmi_male).ztest_ind(usevar='unequal'))
    print()


# #### Paired tests
# 
# A common situation in applied research is to measure the same quantity multiple times on each unit of analysis.  For example, in NHANES, systolic blood pressure is measured at least two times (sometimes there is a third measurement) on each subject.  Although the measurements are repeated, there is no guarantee that the mean is the same each time, i.e. the mean blood pressure may be slightly lower on the second measurement compared to the first, since people are a bit more nervous the first time they are measured.  A paired test is a modified form of mean test that can be used when we are comparing two repeated measurements on the same unit.
# 
# A paired t-test for means is equivalent to taking the difference between the first and second measurement, and using a one-sample test to compare the mean of these differences to zero. Below we see that in the entire NHANES sample, the first measurement of systolic blood pressure is on average 0.67 mm/Hg greater than the second measurement.  While this difference is not large, it is strongly statistically significant.  That is, there is strong evidence that the mean values for the first and second blood pressure measurement differ.

# In[12]:


dx = df[['BPXSY1', 'BPXSY2']].dropna()
db = dx.BPXSY1 - dx.BPXSY2

sm.stats.ztest(db)


# To probe this effect further, we can divide the population into 10 year wide age bands and also stratify by gender, then carry out the paired t-test within each of the resulting 12 strata.  We see that the second systolic blood pressure measurement is always lower on average than the first.  The difference is larger for older people and for males.  The difference is statistically significant for females over 30, and for males over 60.   
# 
# Conducting many hypothesis tests and "cherry picking" the interesting results is usually a bad practice.  Here we are doing such "multiple testing" for illustration, and acknowledge that the strongest differences may be over-stated.  Nevertheless, there is a clear and consistent trend with age -- older people tend to have greater differences between their first and second blood pressure measurements than younger people.  There is also a difference between the genders, with older men having a stronger difference between the first and second blood pressure measurements than older women.  The gender difference for younger peple is less clear.

# In[13]:


dx = df[['RIAGENDRx', 'BPXSY1', 'BPXSY2', 'RIDAGEYR']].dropna()
dx['agegrp'] = pd.cut(dx.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])

for k, g in dx.groupby(['RIAGENDRx', 'agegrp']):
    db = g.BPXSY1 - g.BPXSY2
    print(k, db.mean(), db.size, sm.stats.ztest(db.values, value=0))


# **Research Question:**
# 
# Considering elderly Hispanic adults (80+) living in the U.S. in 2015-2016, did the proportions of males and females who smoked vary significantly?
# 
# - Males: proportion=0.565, n=16
# - Females: proportion=0.250, n=32
# 
# 
# - Approach 1 - Chi-square Test
#     - Are all expected counts for each cell of the 2x2 table under the null hypothesis greater than 5? YES!
# 
# |        | Smoker | Non-smoker |
# |--------|--------|------------|
# | Female | 8      |     24     |
# | Male   | 7      |     9      |
# 
# - Approach 2 - Fisher's Exact Z Test
#     - Especially for the smaller ones

# In[14]:


# chi-square test

# 9 males out of 16 are smokers
# 8 females out of 32 are smokers

result = sm.stats.proportions_chisquare([9,8],[16,32])
print('X2 statistic:', result[0])
print('P-value:', result[1])


# In[15]:


# fisher's exact test

# 7 male smokers
# 9 male non-smokers
# 8 female smokers
# 24 female non-smokers

odds_ratio, p_value = stats.fisher_exact([[7, 9], [24, 8]])
print('P-value:', p_value)

