## Multi-Class Predictive Modeling Challenge

* The challenge was to build a classifier to select appropriate FDA food inspection violation codes from text descriptions, and from a boolean indicating whether the violation was critical.

* The classes represented by the 56 FDA codes were severely imbalanced, with 4 having fewer than 65 instances, and 4 having greater than 100k. (Minimum: 27, maximum: 171,202.)

![FDA code frequencies](fda_q_fixed frequenciesFdaCodeFrequencies.png "FDA code frequencies")

* Four classes (7, 26, 27 & 29) were always critical violations, while four (30, 32, 34 & 46) never were critical violations. The `FDAISCRITICAL` boolean therefore is predictive, and should be included.

* Work was done on a notebook computer with 12 GB of memory, which was somewhat marginal for random forest models.

* Python code used mainly pandas, scikit-learn and gensim packages.

* Comments varied in length from 0 to 844 words (mode of 8) but only 1% had more than 140 words.

![Message word counts](WordCountsHistWordCountsMax844Mode8.png "Message word counts")

### Preprocessing and feature construction

#### Class splits and balancing

The very small classes had insufficient instances for statistically robust training or testing. The best possible is a 50-50 split. However, if the splitting is done randomly from the entire data set, there are no guarantees that the small classes would be evenly split. (There would be a 12% chance that random assignment would place more than 18 of 27 instances of the smallest class in the same test-train split.) For this reason, the splits were done individually for each class.

Manual balancing of classes was not needed, since the model algorithms each had a `class_weight='balanced'` option.

#### TF-IDF features

These were created for unigrams and bigrams using `scikit-learn`'s `TFIDFcountVectorizer()` (with `min document count=5`). No stemming or lemmatization was done. This lead to a large, but manageable<sup>&dagger;</sup>, count of features ~195k.

<sup>&dagger;</sup>It's worth noting that the random forest model consumed enough memory that swapping occurred.

For one model, `gensim` was used to construct unigrams and *commonly-occuring* bigrams by applying `gensim.models.phrases.Phrases()` and `gensim.models.phrases.Phraser()`.
By restricting bigrams to common occurrences, the total feature count was dramatically reduced to ~13k.

### Basic model results

Results for the notebook [PredictiveModelingChallenge.ipynb](https://github.com/mw0/MLnotebooks/blob/master/multiClass/PredictiveModelingChallenge.ipynb) are shown for the first 4 lines of the table below.
After constructing the features (as one-hot encodings), the models used were linear regression, random forest, linear support vector machine, and complement naive Bayes.

Linear regression and linear support vector machine results were identical, while random forest performed moderately better.
Random forest used default parameters, except `n_estimators=180` and `max_depth=250`.
Due to swapping during  training (to flash, fortunately), the random forest results came at the expense of a training time that was multiples of those for the other two.

<table>
<tr><th rowspan=2>Model</th><th rowspan=2>Classifier</th><th colspan=2>Text Features</th><th colspan=3>Metrics</th><th rowspan=2>Minutes to train</th></tr>
<tr><th>TF-IDF</th><th>Topic model</th><th>precision</th><th>recall</th><th>f<sup>1</sup></th></tr>
<tr><td>0</td><td>LR</td><td>&check;</td><td>&cross;</td><td>0.87</td><td>0.86</td><td>0.86</td><td>23</td></tr>
<tr><td>1</td><td>RF</td><td>&check;</td><td>&cross;</td><td>0.89</td><td>0.89</td><td>0.89</td><td>190</td></tr>
<tr><td>2</td><td>LSVC</td><td>&check;</td><td>&cross;</td><td>0.86</td><td>0.86</td><td>0.86</td><td>42</td></tr>
<tr><td>3</td><td>CNB</td><td>&check;</td><td>&cross;</td><td>0.80</td><td>0.79</td><td>0.79</td><td>0.05</td></tr>
<tr><td>4</td><td>LR</td><td>&cross;</td><td>&check;</td><td>0.42</td><td>0.40</td><td>0.38</td><td>128 + 7</td></tr>
</table>