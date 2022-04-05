# **Modeling Details**

## Approach

The NASA FIRMS data contains active fires labeled with latitude/longitude coordinates and a timestamp. These fires were detected by a satellite. Our model aims to predict what areas will be burning tomorrow. We are specifically interested in predicting where current fires will spread, rather than where new fires will start. To conform the data to our use case, we converted the timestamp to a date, as we are not concerned about the exact moment that the NASA satellite detected the fire - we are only interested in the fact that it was detected on a particular day. While the resolution of the satellite remains constant at 375 meters, the satellite&#39;s exact day-to-day path is variable due to the rotation of the earth and the revolution of the satellite around the earth. This makes it virtually impossible to predict exactly which coordinates the NASA satellite will detect to be burning in the next 24 hours. Therefore, we have transformed our map into a grid of one-square-mile squares. We consider a square to be burning today if there is at least one coordinate pair detected by the satellite within the bounds of that square today. Each square corresponds to a single sample in our model and the binary target variable describes whether we expect that square to burn tomorrow or not.

We consider two main factors when predicting whether a square will burn tomorrow: (1) the strength of the fire at that square and its nearest neighbors, both today and in the past two days, and (2) the weather at that square and its nearest neighbors, both today&#39;s values and tomorrow&#39;s forecast. We define a square&#39;s fire strength today as the number of coordinate pairs detected by the satellite within the bounds of that square today. We consider the following weather conditions: precipitation (intensity and maximum intensity), temperature (high and low), humidity, wind (speed, gust, and bearing), and cloud cover.

The team obtained NASA satellite data from January 1, 2014 to February 1, 2022. If we were to consider all 147,000 square miles on all 2953 days, we would have 40 million samples in our dataset. Not only would this incur heavy computational and weather API costs, but it would not be useful to predict the spread of fires. Rather, that approach would be more suitable if we were also interested in where fires will start. Therefore, we decided to choose a threshold defined as the maximum distance a square must be from a square burning today in order for it to be modeled. That is, the distance between the midpoints of the two squares must be at or below this threshold. We assume that all squares outside of this threshold will not burn tomorrow. If we were building a prototype, we would determine the maximum distance that a fire can spread in a day and use that to define the threshold. However, due to the proof of concept nature of this case competition, we instead analyzed the marginal benefit of increasing this threshold by one mile. The marginal benefit is defined as the additional percentage of all burning squares in the dataset that would be modeled by increasing the threshold by one mile. In the dataset, 47% of squares burning tomorrow were burning today. 65% of squares burning tomorrow were at most one mile away from a square burning today. Therefore, the marginal benefit of increasing the threshold from zero miles to one mile is 18% (65% - 47%). The marginal benefit of increasing the threshold to two miles is under 4%. The marginal benefit plateaus after this point, so we selected a threshold of two miles.

## Modeling

Our model is trained on data from January 1, 2014 through December 31, 2020. We used 2021 as a test set to tune our model. Then, using the best performing model, we predicted on January 1, 2021 through February 1, 2022, and these predictions are shown in the app.

The target variable was highly imbalanced in our training set. Non-burning squares (i.e. squares not burning tomorrow) outweighed burning squares nine-fold, so we upsampled burning squares by duplicating each sample eight times.

As aforementioned, we considered fire strength and weather to predict the condition of each square tomorrow. The nomenclature for the fire strength predictors is _FireStength(y, x, d)_, where y is between -3 and 3, x is between -3 and 3, and d is between 0 and -2, inclusive. _FireStrength(-2, 1, -1)_, for example, corresponds to the fire strength one-day-ago at the (midpoint of the) square 2 miles south and 1 mile to the east of the (midpoint of the) square being modeled. The nomenclature of the weather predictors is _metric(y, x, d)_ where y is between -2 and 2, x is between -2 and 2, x and y are both divisible by 2, d is either 0 (current) or 1 (forecast), and the metric is one of the following: precipIntensity, precipIntensityMax, temperatureHigh, temperatureLow, humidity, windSpeed, windGust, windBearing, windSpeedSquared, or cloudCover. windSpeedSquared was engineered following conversations with a NOAA incident responder, who emphasized that wind speed has a quadratic effect on fire spread. x and y must both be divisible by 2 because neighboring weather data is highly correlated and we aimed to avoid overfitting by increasing the spacing between points. Finally, we considered interaction effects between fire strength and weather by engineering features that are products of one fire strength feature and one weather feature. The nomenclature for these is _metric\_x\_FireStrength(direction)_ and _metric\_x\_FireStrength(direction)(forecast)_, where direction is N, S, E, W, NE, NW, SE, or SW, and &#39;(forecast)&#39; is a suffix added to the feature name to indicate that the weather feature is a forecast. For example, _windSpeed\_x\_FireStrength(N)(forecast)_ is the product of _windSpeed(2, 0, 1)_ and _FireStrength(1, 0, 0)_. It is useful to capture this information in the model because high winds where there is no fire do not increase the chance of a fire spreading, nor do low winds where there is no fire, yet high winds where there is a fire can greatly increase this chance. For squares greater than one mile away from the nearest burning square, _FireStrength(1, 0, 0)_ is substituted with _FireStrength(2, 0, 0)_ in the calculation above. It would be unwise to include in the model a feature that is calculated differently depending on some criteria, therefore, squares greater than one mile from the nearest burning square are modeled separately from squares up to one mile away from the nearest burning square.

A Random Forest Classifier was used for modeling. This model uses an ensemble of decision trees to output a confidence level for each sample. This confidence level is between 0 and 1 and indicates how confident the model is that a square will burn tomorrow. For example, a value of 0.5 indicates that the model is 50% confident that the square will burn tomorrow. We still consider the target variable to be binary, however, because all predictions with a confidence value at or above a given threshold are converted to 1 and those below the threshold are converted to 0, where 1 is a prediction that the fire will burn tomorrow and 0 is a prediction that it will not. The tool allows the user to choose the confidence threshold. A user may elect to choose a higher threshold if he or she has a limited amount of resources available, or a lower threshold if he or she has a significant amount of resources available.

## Evaluation

We tested different features, upsampling parameters, and Random Forest hyperparameters to determine the best model. The measure we used to determine the best model was f1-score. In making this decision, we took into account the main drawback of using accuracy for imbalanced classification problems, which is that a highly accurate model might still produce a large amount of False Negatives or False Positives. In our use case, the consequence of either of these is considerable. See the definitions below.

True Positive (TP): A square where fire is detected tomorrow, and where the model predicted that it will burn

False Negative (FN): A square where fire is detected tomorrow, and where the model predicted that it will not burn

True Negative (TN): A square where fire is not detected tomorrow, and where the model predicted that it will not burn

False Positive (FP): A square where fire is not detected tomorrow, and where the model predicted that it will burn

A False Negative could result in an uncontrolled fire. A False Positive could result in misallocated resources. While accuracy does not provide information about False Negatives or False Positives, f1-score does so, as it is a function of precision and recall. See the definitions below.

![Image](assets/metrics.png)

Maximizing both precision and recall is critical, but the two measures are inversely proportional. Therefore, f1-score, calculated as the harmonic mean of precision and recall, is often used to balance the two. As aforementioned, the model outputs a percent confidence that a square will burn. The default threshold for classification problems is usually 50%, but this is often not optimal for problems with imbalanced classes. In our case, a confidence threshold of 41% resulted in the highest test set f1-score. The value was 0.69 out of 1, and values above 0.5 are generally considered strong.

The UI allows the user to view a confusion matrix. The confusion matrix summarizes the performance of a classification model. The rows represent the actual (AKA truth) classes and the columns represent the predicted classes. The two classes in our use case are &quot;fire&quot; and &quot;no fire&quot;. The cells in the matrix indicate the total occurrences of the result corresponding to the actual class in that row and the predicted class in that column. The four results/cells are:

TP: Intersection of actual &quot;fire&quot; and predicted &quot;fire&quot;

TN: Intersection of actual &quot;no fire&quot; and predicted &quot;no fire&quot;

FN: Intersection of actual &quot;fire&quot; and predicted &quot;no fire&quot;

FP: Intersection of actual &quot;no fire&quot; and predicted &quot;fire&quot;

Feature importance can be evaluated for each individual prediction using SHAP values. SHAP [(SHapley Additive exPlanations)](https://christophm.github.io/interpretable-ml-book/shap.html) is a game theoretic approach that treats features as players who compete against each other to affect the target variable. When considering a single prediction, each feature is assigned a SHAP value which, in short, is its marginal contribution to that prediction. Every feature and its marginal contribution on a prediction can be visualized using a SHAP force plot. In our app, clicking on a predicted square will generate its force plot. The red, right-pointing arrows represent features that had a positive contribution on the output (confidence level) and the blue, left-pointing arrows represent features that had a positive contribution. The size of the arrow corresponds to the magnitude of the contribution. The larger arrows are labeled with the corresponding feature and its value in the form of _FeatureName = value_ and these labels can be uncovered for the smaller arrows by hovering over them. Note that this _value_ is the actual value of the feature used as model input rather than the feature&#39;s marginal contribution - the latter is only represented by the size of the arrow. The final insight from the plot is the output (confidence level in decimal form) in bold. The output is equal to the sum of every feature's marginal contribution plus the average output across all predictions.

## Fire Identification and Affected Counties

The UI allows the user to zoom into specific fires. Each fire in the dataset has a unique ID. In order to classify data points as part of the same wildfire, we used the ST-DBSCAN clustering algorithm. This algorithm considers a maximum spatial threshold and maximum temporal threshold that two data points must be within in order to be classified as part of the same fire. The maximum spatial threshold used was miles so that two adjacent or diagonal modeled squares would be classified as part of the same fire. The maximum temporal threshold used was one day. A higher temporal threshold would increase the chance of multiple fires being classified as one fire. While this is already possible using a one-day threshold, it is often the cause of a fire separating into multiple fires. Using a temporal threshold of zero would not allow the user to examine the spread of a particular fire over multiple days.

Each fire is labeled with the county or counties that have the greatest chance of being affected by the fire. These were determined using two sources: (1) Google Geolocation API and (2) California County Coordinates. The API was used to label each historic data point with a county. The CA County Coordinates were used to supplement the API - as the API was often inaccurate - by providing boundaries for each county in the form of a maximum and minimum latitude and a maximum and minimum longitude. This source essentially provides a circumscribed rectangle around each county. This results in many overlapping rectangles, which is why this source cannot be used on its own. After being assigned a county by the API, historic coordinates with a county outside of that county&#39;s rectangular boundaries were discarded (they were still used in modeling, just not in this county identification process). Finally, for each fire, the county names from all historic coordinates within that fire&#39;s boundaries were aggregated into a list. Any county appearing in less than 1% of list elements was discarded, then each remaining unique county appears alongside that fire&#39;s ID.

## Backend Modeling Deployment

To run the backend modeling locally, change the _local_ variable in line 18 in common.py to True. Then, ensure that all requirements in requirements.txt are installed. Finally, run the scripts in the following order:

data\_prep \&gt;\&gt; modeling \&gt;\&gt; display

## Data Sources

Current Fire Data: NASA, Fire Information for Resource Management System (FIRMS), https://earthdata.nasa.gov/earth-observation-data/near-real-time/firms

Weather (historical): Dark Sky API, Dark Sky by Apple, https://darksky.net/

County Coordinates: Google Geolocation API, https://developers.google.com/maps/documentation/geolocation/overview

&quot;California county coordinates &quot;, American Library Association, June 25, 2007.

http://www.ala.org/rt/magirt/publicationsab/ca (Accessed February 22, 2022)

Document ID: 394d6a85-4b8b-ed44-7991-e4862e4ba271

## SHAP

Lundberg, Scott M., en Su-In Lee. &quot;A Unified Approach to Interpreting Model Predictions&quot;. Advances in Neural Information Processing Systems. Red I. Guyon et al. vol 30. Curran Associates, Inc., 2017. Web.

## Clustering Article and Original Code

Vega Orozco, C., Tonini, M., Conedera, M. et al. Cluster recognition in spatial-temporal sequences: the case of forest fires. Geoinformatica 16, 653–673 (2012). [https://doi.org/10.1007/s10707-012-0161-z](https://doi.org/10.1007/s10707-012-0161-z)

Atila Jr and Tharaka Devinda, ST-DBSCAN, (2020), GitHub repository, [https://github.com/gitAtila/ST-DBSCAN](https://github.com/gitAtila/ST-DBSCAN)
