## <p align="center"><u>Predictive Modelling for Wind Power generation</u></p>

<h2> Executive Summary </h3>
   
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

<h2>Introduction</h2>

Wind energy is one of the fastest growing renewable energy sources.
According to US Department of Energy (DOE), the US’ wind power capacity
has surpassed 82 gigawatts at the end of 2016, making it the largest
renewable generation capacity in the United States. A technical wind
resource assessment done in 2016 by DOE estimated that the land based
wind energy potential is 10800 GW (calculated within 200 nautical miles
of shore using 100m hub heights) and applying some exclusions, the
technical resource potential accounted to 2058 GW of capacity or 7203
terawatt-hours, which is almost double the nation’s annual electricity
usage. So you can imagine the amount of energy that is just flowing by,
literally, and we haven’t been able to harness it. Thus, advancements in
this field should be one of the primary goals as it is a clean form of
energy and has its own benefits.

To be able to harness more and more wind energy, we need to analyze the
past data and be able to predict the power output, so that would help
drive our technology based decisions such as design of turbines.

The current industrial practice of estimating the power output relies on
a binning method proposed by International Electrotechnical Commission
(IEC 2005). The basic idea of this method is to convert the continuous
wind speed data into discrete bins of a fixed width. Then the value to
be used for representing the power output for a given bin is found by
averaging all the power output values corresponding to the wind speed in
each bins, namely:

<p align="center">
   <img src="http://latex.codecogs.com/svg.latex?y_i%3D%5Cfrac%7B1%7D%7BN_%7Bi%7D%7D%5Csum_%7Bj%3D1%7D%5E%7BN_i%7Dy_%7Bi%2Cj%7D">
</p>

where ![img](http://latex.codecogs.com/svg.latex?y_%7Bi%2Cj%7D) is the power output of the ![img](http://latex.codecogs.com/svg.latex?j%5Et%5Eh) data point in bin *i*
and N_i is the number of data points in bin *i*. So this method takes
only wind speed into consideration while predicting the power output and
ignores all other environmental variables which might have a significant
effect on the analysis.

In a way, it makes sense because wind speed is the major factor that
would decide your power output. But other factors like temperature, air
pressure, humidity, etc. also impact our output variable and thus should
be taken into consideration while performing the analysis.

Data was collected using the wide array of sensors installed on
meteorological masts in wind farms. The environmental variables measured
include wind speed, V, wind direction, D, temperature, T, air pressure,
P, and humidity, H. The data was collected over a period of one year so
as to take seasonality into account. The final data used for analysis
was averaged over 10 minute intervals. Wind speed standard deviation was
calculated for each 10 minute interval. So, the data used for our
analysis included wind speed, V, wind speed stdev, SD, wind direction,
D, environment temperature, T, turbulence intensity I. Turbulence was
calculated using the following formula:

<p align="center">
   <img src="http://latex.codecogs.com/svg.latex?I%3D%5Cfrac%7B%5Ctext%7BSD%7D%7D%7BV%7D">
</p>

<h2>Data</h2>

### **2.1 The Dataset**

The dataset is from a wind farm and were collected by sensors on a
meteorological mast, while the power output was measured at a wind
turbine. The data is recorded as a ten-minute average and the training
data has 30997 data records, covering the whole year of 2015; while the
test data has 16499 data records, covering the first six months of 2016.
The datasets have fewer than the expected data records due to problems
in industrial settings. Recording time, average wind speed (m/s),
standard deviation of wind speed (m/s), average wind direction (degrees)
and temperature (℃) are the 5 attributes in the data. Wind power
(normalized) is the output which is expressed in a value between 0 to 1.

For our analysis, we have not considered the input variable of recording
time and added a new one called turbulence Intensity which can be
calculated as I = σ/V, where σ is the standard deviation of wind speed
and V is the average wind speed.

### **2.2 Correlation among Attributes**

We can intuitively get an image of the attributes that are highly
related to the output and those that are not. Also, highly correlated
attributes do not provide any additional valuable information for
prediction. For this purpose, we have plotted the correlation matrix
using the *seaborn.heatmap* tool as shown in Figure 1.

<p align="center">
   <img src="https://github.com/KanchanSatpute/Python-Wind-Power-Estimation/blob/master/Plots/CorrelationMatrix.png">
</p>
<p align="center">
   <i>Figure 1: Heat map showing correlation</i>
</p>

In figure 1, we can see that the wind power is highly correlated with
the wind speed and standard deviation of wind speed, whereas, medium
correlation can be seen with wind direction and turbulence intensity.

To illustrate the effect of correlation further, we have plotted the
scatterplots between all the pairs of attributes and power output, using
the *seaborn.PairGrid* tool.

<p align="center">
   <img src="https://github.com/KanchanSatpute/Python-Wind-Power-Estimation/blob/master/Plots/PairPlots.png" width="700" height="600">
</p>
<p align="center">
   <i>Figure 2: Pair plots</i>
</p>

Figure 2 consists of the scatterplots between all the factors as well as
the output and the histograms representing the underlying distribution
of each of them. We observe some non-linear relationships in these plots
and they appear to differ according to wind conditions. This suggests
that correlation exists among wind speed or wind direction and the other
factors.

Figure 3 shows the plot of average power output per month to see if it
has any relation with time. Although there is no perfect visible trend
in this plot, we can say that the power output is comparatively higher
for the months of (Nov – April) and are lower for the months of (May –
Oct).

<p align="center">
<img src="https://github.com/KanchanSatpute/Python-Wind-Power-Estimation/blob/master/Plots/Avg%20power%20each%20month.JPG">
</p>
<p align="center">
<i>Figure 4: Average power for each month</i>
</p>

This implies that spring season generates comparatively higher power
output on an average, which is a little intuitive because of the change
in weather conditions. This finding suggests that we should analyze the
change of input variables with time to see if there is a larger
variability in data as time changes.

<p align="center">
   <img src="https://github.com/KanchanSatpute/Python-Wind-Power-Estimation/blob/master/Plots/Avg_pred_values_per_month.JPG">
</p>
<p align="center">
   <i>}Figure 5: Average values of predictors over each month</i>
</p>

Figure 4 represents the average values of all the attributes taken over
each month for the training data and gives an idea of how each of them
vary with time. We can see that there is higher variability in wind
direction and temperature. So, fitting a model with these attributes
might not give a good prediction accuracy on the test data, because the
period in which both these data were collected are different. So, we
should consider the attributes that are consistent with time and have
less variability. Also, the standard deviation of wind speed has already
been taken into consideration by adding the turbulence intensity in the
set of attributes. Therefore, we will consider only the two attributes
‘wind speed’ and ‘turbulence intensity’ for fitting our model and doing
prediction.

We have further separated our training data into training and validation
sets of 21697(70%) and 9300(30%) each using the
*sklearn.model\_selection* tool. All the models will be trained on the
training set and validated on validation set.

<h2>Methods</h2>

This section discusses the methods or algorithms used for building the
prediction model. We have tried to use majority of the popular learning
methods available and decrease RMSE as much as possible.

For gaining proper intuition into the accuracy of the model, the data
has been divided into 70% training data and 30% validation set.

### **3.1 Linear Regression**

There is not much point of applying this method because as compared to
binning method, this method tries to fit a linear function on the data.
While we are looking for a method that does similar job compared to the
binning method.

But generally, linear regression is the pathway that leads onto better
models and hence one always start with the simplest method of all.

Training RMSE = 0.08099

Validation RMSE = 0.07799

As expected, this method does not perform better than the IEC binning
method.

Below shown are the plots: (Left panel): Predicted values vs Output,
(Right panel): Residual vs Predicted values

![](media/image6.png){width="2.5680008748906387in"
height="1.6940737095363079in"}
![](media/image7.png){width="2.599758311461067in"
height="1.7920002187226596in"}

[]{#_Toc512950106 .anchor}Figure : (Left Panel) Predicted value vs
Output, (Right Panel) Residual vs. Predicted output

### **3.2 Ridge Regression**

Similarly, using sparse model won’t be of much help since ridge
regression is built up on linear regression only and the basic algorithm
remains the same except it imposes penalty to more variables used. Also,
this method produces similar RMSE when compared to linear regression.

Training RMSE = 0.081

Validation RMSE = 0.078

### **3.3 Lasso Regression**

Lasso is a shrinkage method just like Ridge and follows similar kind of
principles, except it has absolute value rather the squared value in the
penalty term, so Lasso might reduce some of the unimportant variables to
zero. But basic idea remains the same. Thus, it does not give
satisfactory results.

Training RMSE = 0.0831

Validation RMSE = 0.0845

### **3.4 Support Vector Machine**

Tries to fit a hyperplane such that error is minimized and margin is
maximized, with some error tolerated (depending on how much penalty you
impose on errors). There are several options of kernels to use. The
library we have used provides options like ‘rbf’ – radial basis
function, ‘poly’ – higher degree function, ‘linear’, ‘sigmoid’, etc.

<p align="center">
   <img src="https://github.com/KanchanSatpute/Python-Wind-Power-Estimation/blob/master/Plots/svm%20-%20pred%20vs%20output.png" width="500" height="400">
</p>
<p align="center">
   <i>Figure 6: Predicted output vs Actual output</i>
</p>

The above figure shows the plot of predicted power vs. actual power. We
did a 10 – fold cross validation for different combinations of kernel
and cost parameter. The ‘rbf’ – radial basis function, with a cost
parameter of 0.5 gives the best result (in our case least RMSE)

Training RMSE = 0.0475

Validation RMSE = 0.04704

### **3.5 Decision Tree**

Decision Tree is a much simpler algorithm as compared to other
algorithms. Decision tree gives much better results as compared to
linear, ridge, lasso and SVM. This is because when you compare this
method with the industry standard IEC binning method, they both have
some similarity. In IEC binning method, you make bins of predefined bin
width and then put the data points in each of that bins and for
prediction we average the corresponding output values in each bin. In
decision tree, we do a similar kind of procedure, except we don’t
specify any kind of “node size” except the min. node size. And we do the
prediction by taking the average at each node. So this method is quite
analogous to the IEC binning method. And thus, provides better RMSE
values.

Training RMSE = 0.0017

Validation RMSE = 0.0552

### **3.6 Decision Tree with AdaBoost**

We have applied AdaBoost to decision trees to enhance the prediction
accuracy.

Training RMSE = 0.0495

Validation RMSE = 0.0494

### **3.7 Random Forest**

Random forest is basically a bag of trees that have been trained on
bootstrap samples generated from randomly selected predictors from the
training data. So, while given the number of trees to be developed, we
get several decision trees. So when a prediction is to be done on out of
sample data point, we take the average of all predictions obtained from
all the decision trees. So, basically, its an ensemble of trees built
after bootstrapping and bagging.

Training RMSE = 0.0356

Validation RMSE = 0.039

<h2>Conclusion</h2>

After applying above predictive modelling techniques and carefully
observing the output, we have chosen Random Forest to be the best model
so far. As discussed earlier in decision trees, it is very much
analogous to IEC binning method but a little bit more upgraded. So
Random forest gives better prediction as compared to SVM. And random
forest builds a group of decision trees based on bootstrap samples, thus
decreasing the variance, so helps improving the prediction accuracy.

Moreover, an important feature of random forest is its use of out-of-bag
(OOB) samples. An OOB error estimate is almost identical to that
obtained by N-fold cross-validation. Hence unlike many other nonlinear
estimators, random forests can be fit in one sequence, with
cross-validation being performed along the way. Once the OOB error
stabilizes, the training can be terminated.

Decision trees are known for showing high variance and low bias. They
are often accurate but show a large degree of variability, between
different data samples taken from the same data. Random Forest can
reduce this variance that can cause errors in decision trees by
aggregating the different outputs of the individual trees. Thus, Random
Forest transforms low bias high variance decision trees into a model
that has both, low variance and low bias.
