## Machine Learning Notebooks

* [InterpretableML](https://github.com/mw0/MLnotebooks/tree/master/InterpretableML)
  * tutorial, originating from Domino Data Labs, for application of `LIME` and `shap` (Shapley values) for estimating contributions of features to a given instance prediction
* [KaggleWeather]()
  * Examples of time-series analyses applied to Kaggle weather data set
  * Includes a primer on the use of wavelet scaleograms
* [KerasProjects]()
  * Multiple re-works of examples from Francois Chollet's *Deep Learning with Python*
  * Example, from Analytics Vidhya, of CNN classifier of age in actor images from film stills
* [multiClass](https://github.com/mw0/MLnotebooks/tree/master/multiClass)
  * Classifiers for selecting from on of 56 codes, based upon a "critical" boolean and text notes of varying length
  * Classes are very un-balanced, with item counts spanning more than 3 orders of magnitude
  * `scikit-learn` and `gensim`
* [SparkProjects]()
  * Exploratory analysis of large server logs using Spark
* [styleTransfer](https://github.com/mw0/MLnotebooks/tree/master/styleTransfer)
  * Applies [Gatys *et al., *Image style transfer using convolutional neural networks*, CVPR 2016](http://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html) to create images from photographs that match an artistic style.
* [PredictDriverConversion](https://github.com/mw0/MLnotebooks/tree/master/PredictDriverConversion)
  * binary classifier based upon time &Delta;s from driver sign-up date to different actions, and upon make and model of vehicle.
* [SoccerRankingCollabFiltering]()<br>
  Given a sparse matrix of points scored in actual recent matches:
  * applies double instances of matrix factorization to predict points for play between pairs of all teams
  * sums outcomes to determine overall rankings
  * applies time discounting and match weighting in cost function
  * treats home and away games as distinct
  * infers confederation strengths (which differ substantially from FIFA's)
