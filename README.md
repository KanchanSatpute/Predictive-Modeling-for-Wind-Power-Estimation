# Python - Wind Power Estimation

<h2> Executive Summary </h2>
   
Current industrial practice of predicting the power generation heavily relies on a method known as the binning method which is recommended by International Electrotechnical Commission (IEC). This is a simple approach where the main predictor, i.e., wind speed, is divided into discrete bins. The value to be used for representing the power output for a given bin is average for all the data points in each of those bins. <br>

Above mentioned method is a simple method which takes into consideration
only a single predictor namely “Wind speed”. There are other factors
which may significantly impact the power generation but are ignored by
this method.

The data for aforementioned analysis has been collected from a wide
array of sensors installed on the meteorological mast in the wind farm.
Data collection period is one year, so as to factor in the seasonal
changes as well. The data has been averaged over 10 minute intervals for
ease of analysis. We have done some data exploration to understand the
kind of data we are dealing with.

This project aims to try to factor in the predictors that may impact the
power output and see their correlation. We have tried some popular
learning methods like Gaussian process regression, SVM, CART, Random
Forest along with methods like AdaBoost to enhance prediction. We also
have used basic methods like linear regression, sparse models like ridge
and lasso. In this era of deep learning, we have avoided using Neural
Network because the method is kind of a “black box” and is difficult to
explain to a layman. We have used python, a popular programming language
across the industries, to carry out the analysis.

Some of the methods use sample from the data rather than the whole data
for training. We have used training and validation set, to get a proper
idea of accuracy. Cross validation has been applied to SVM and Random
Forest for “hyper-parameter” tuning.

Root Mean Square Error has been used as a measure for accuracy. Even
though, CART being a simpler method as compared to other algorithms like
SVM and GPR, it gives the least RMSE among all the methods used for
prediction. Might be because of how the data is inherently structured.
Finally we compare the RMSE of all the models and use the best among
those methods for final prediction on test dataset.

The project report can be found in the file 'Project_report.md'
