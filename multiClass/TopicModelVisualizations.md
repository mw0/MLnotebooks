## Topic model visualizations

The best-seeming LDA topic model results (from all of two attempts) arose when setting the topic number to 8.
The visualizations below were obtained using `pyLDAvis`.

The first panel represents all topics in an inter-topic distance visualization, using dominant principle component axes.
These show very good inter-topic separation &mdash; which is surprising, given that I would expect food inspection report comments to have many overlapping terms. Of course, projections onto less-significant PCA components would reveal overlap not seen here.

![All topics, overall word frequencies](images/Topic0.png "All topics, overall word frequencies")

Below are plots of the same for individual topics, and the associated word frequencies:

![Topic 1, word frequencies](images/Topic1.png "Topic 1, word frequencies")


![Topic 2, word frequencies](images/Topic2.png "Topic 2, word frequencies")


![Topic 3, word frequencies](images/Topic3.png "Topic 3, word frequencies")


![Topic 4, word frequencies](images/Topic4.png "Topic 4, word frequencies")


![Topic 5, word frequencies](images/Topic5.png "Topic 5, word frequencies")


![Topic 6, word frequencies](images/Topic6.png "Topic 6, word frequencies")


![Topic 7, word frequencies](images/Topic7.png "Topic 7, word frequencies")


![Topic 8, word frequencies](images/Topic8.png "Topic 8, word frequencies")
