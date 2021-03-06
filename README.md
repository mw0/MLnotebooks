## Machine Learning Notebooks

* [HuggingFace](https://github.com/mw0/MLnotebooks/tree/master/HuggingFace)
  * Apps derived from HuggingFace implementation of NLU transformer models
* [InterpretableML](https://github.com/mw0/MLnotebooks/tree/master/InterpretableML)
  * tutorial, originating from Domino Data Labs, for application of `LIME` and `shap` (Shapley values) for estimating contributions of features to a given instance prediction
* [multiClass](https://github.com/mw0/MLnotebooks/tree/master/multiClass)
  * Classifiers for selecting from on of 56 codes, based upon a "critical" boolean and text notes of varying length
  * Classes are very un-balanced, with item counts spanning more than 3 orders of magnitude
  * Apply several classification models, using TD-IDF type features from scikit learn and from gensim
  * LSTM-based model, including auxiliary feature for boolean
  * Apply classification models using LDA topic weights as features (no surprise, not a success)
  * `scikit-learn`, `gensim` and `TensorFlow`
* [styleTransfer](https://github.com/mw0/MLnotebooks/tree/master/styleTransfer)
  * Applies [Gatys *et al., *Image style transfer using convolutional neural networks*, CVPR 2016](http://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html) to create images from photographs that match an artistic style.
* [KaggleWeather](https://github.com/mw0/MLnotebooks/tree/master/KaggleWeather)
  * *A work in progress!*
  * Examples of time-series analyses applied to Kaggle weather data set
  * Includes a primer on the use of wavelet scaleograms
* PredictDriverConversion
  * binary classifier based upon time &Delta;s from driver sign-up date to different actions, and upon make and model of vehicle.
* SoccerRankingMatrixFactorization
  Given a sparse matrix of points scored in actual recent matches:
  * applies double instances of matrix factorization to predict points for play between pairs of all teams
  * sums outcomes to determine overall rankings
  * applies time discounting and match weighting in cost function
  * treats home and away games as distinct
  * infers confederation strengths (which differ substantially from FIFA's)
* KerasProjects
  * Multiple re-works of examples from Francois Chollet's *Deep Learning with Python*
  * Example, from Analytics Vidhya, of CNN classifier of age in actor images from film stills
* SparkProjects
  * Exploratory analysis of large server logs using Spark
