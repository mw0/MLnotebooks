## Implicit

Implicit feature-based recommendations using Implicit code.

This makes use of Ben Frederickson's very well-optimised [Implicit](https://github.com/benfred/implicit) code.
For documentation, refer to [Fast Python Collaborative Filtering for Implicit Datasets](https://implicit.readthedocs.io/en/latest/).

More will be included here after grid search is complete. Most descriptions of the code rely on anecdotal indicators of the model's validity, rather than metrics derived from all results. (As will be shown in more detail, model *fitting* for the last.fm dataset takes < 1 minute on a GTX 1080 Ti GPU, but generating recommendations for all users, or computing evaluation metrics takes much, much longer.)
