## Interpretable ML with SHAP and LIME

* [SHAP and LIME](#shap-and-lime)

### SHAP and LIME

Shapley values <sup>&dagger;</sup>: originating in cooperative game theory, these are computed as the contributions of a single player (for our purposes, a model feature) to the total outcome. This is computed for feature *i* by examining the average of model outputs for all coalitions that can be formed by *excluding* feature *i*.
For examples, refer to &sec;5.9 of Christoph Molnar's [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/shapley.html#fn41), and the [Airport problem](https://en.wikipedia.org/wiki/Airport_problem).

Local Interpretable Model-agnostic Explanations (LIME)<sup>&ddagger;</sup>:
to explain a single instance *X<sub>i</sub>*, points *X<sub>j</sub>* are sampled in the neighborhood of *X<sub>i</sub>* and the black box model *f* is used to make a class prediction.
Given the individual feature value differences, *Δ<sub>i</sub>*, and weighting these results by proximity values *π<sub>X<sub>j</sub></sub>*, the impact of individual features can be estimated. Thes results are represented by the 

<sup>&dagger;</sup>[Shapley, Lloyd S. *A value for n-person games.* Contributions to the Theory of Games 2.28 (1953): 307-317](https://www.degruyter.com/view/books/9781400881970/9781400881970-018/9781400881970-018.xml)
<sup>&ddagger;</sup>[Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. “Why should I trust you?: Explaining the predictions of any classifier.” Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. ACM (2016)](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)

* These examples are modified from a Jupyter notebook provided by Domino Data Lab, described in their 14 Aug 2019 webinar.

* For the k-nearest neighbors model, computing Shapley values is prohibitive, due to the $\frac{k'(k'-1)}{2}$ distances that must be computed for each permutation of features left out; instead cluster size-weighted k-means values are supplied to `shap.KernelExplainer()`.
* Examples are provided for XGBoost Shapley force plots that generate separate image files that can be inserted into documents.
* As XGBoost `.predict()` requires DMatrix inputs, a wrapper function is invoked when generating LIME plots.
* For XGBoost there are examples of how to export images (PNG files in this case).
