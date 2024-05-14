#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import Image


# #### Linear Regression

# In[2]:


Image('images/example_6.png', width=600)


# Next to our coefficients where we pulled off the values to give us our equation of our line, we've got some information that does test hypotheses about those underlying true intercept and slope. Looking at the row with our label of height, we have our coefficient, our estimated slope, our B1 of 1.1. Now, that's not equal to zero, but we're interested in seeing how far away from zero is it to show it there's evidence to conclude that the true slope might not be zero. Well, one piece of information we need is that standard error. So, here, this 0.67 is estimating for us how far away estimated slopes, like the one we got here, are going to be from the true slope on average. Taking those two pieces of information, we form our T statistic. It is measuring how close is our estimated slope, 1.1 from zero in standard error units. Our estimated slope was 1.65 standard errors above zero. So, that's starting to be a pretty good distance from zero. Converting that to a probability value, our p-value is given next, 0.112, which is not that small, certainly not significantly at even at 10 percent level, but this is the p-value for a two-sided alternative. So, if you wanted to assess whether the true slope is zero or not zero, we would report this level for our p-value. Our initial research question was a significant positive relationship between our two variables. So, our alternative theory would be looking for the true slope being greater than zero because that was the direction that made sense. So, our p-value would not be the two tails together for probabilities, but just the one tail. So, we need to take our two-sided p-value and cut that in half. Our p-value for assessing a significant positive linear relationship between cartwheel distance and height turns out to be 0.56. Significant at a 10 percent level, but not at a five percent level, marginally significant.

# In[3]:


Image('images/example_7.png', width=600)


# We might also be interested in reporting a range of values for which we might say is reasonable for this true slope, a confidence interval. That is reported a little further down at 95 percent level for us. So, with 95 percent confidence, the population mean change in cartwheel distance for a one-inch increase in height would be estimated to be anywhere from 0.2 inches shorter, but up to as high as 2.5 inches longer. 

# In[4]:


Image('images/example_8.png', width=600)


# You see a quiz question above. When looking at the coefficients, the estimated coefficient of **head** feature is **0.26343** and also, estimated **intercept** value is **325.573**. So, we can calculate the estimated **brain weight** as follows,
# 
# ```python
# brain_weight = 0.26343 * head + 325.573
# ```

# In[5]:


actual_weight = 1430.86
estimated_weight = 0.26343 * 3500 + 325.573

observed_error = abs(estimated_weight - actual_weight)
observed_error


# And the second question is here,
# 
# **What is the appropriate p-value for testing if there is a significant positive linear relationship between brain weight and head size?**
# 
# As we mentioned before, p value is calculated as two-tailed in the question above. In fact, our null hypothesis is that the **true slope = 0** and our alternative hypothesis is **slope > 0**, not **slope $\ne$ 0**. Because we are investigating the positive linear relationship between two variables. So, our appropriate p-value is **<1e-16**

# #### Logistic Regression

# In[6]:


Image('images/example_9.png', width=600)


# **Please notice that**, in this example, our primary variable of interest is whether or not they completed a cartwheel. Either yes, they successfully completed a cartwheel, which is a one. Or no, they didn't successfully complete the cartwheel, was just coded as a zero. We want to know if based on age, we can predict whether cartwheel is completed or not.
# 
# **For the results**, we can interpret this a little bit differently. So, for each year increase in age, the odds of a successful cartwheel increases by about 1.23 times that of the younger age on average. That 1.23 is calculated by doing e to the slope to that 0.2096. What this means is that for each year increase, the odds of successfully completing a cartwheel increases by a multiplicative factor, so you're multiplying it to the previous odds. In essence what it means, when the odds are greater than one or when the odds increases greater than one is that each year you're more likely to successfully complete a cartwheel. If it were less than one, you would be less likely to successfully complete a cartwheel.
