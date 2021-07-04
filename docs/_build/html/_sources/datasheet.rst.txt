Datasheet for SHIFT28M
====================================

=====================
Motivation
=====================

    For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.

Many machine learning algorithms assume that the training data and the test data are generated from the same distribution.
In the real world, however, this assumption is most often violated.
Many robust algorithms against such dataset shifts have been studied, but they often experiment with artificially induced dataset shifts on originally i.i.d. datasets.
Although such experiments seem reasonable, `recent studies <https://papers.nips.cc/paper/2020/hash/d8330f857a17c53d217014ee776bfd50-Abstract.html>`_ have reported that there is no correlation between robustness to artificial dataset shifts and robustness to natural dataset shifts.
The main motivation of the SHIFT28M project is to provide a dataset that contains natural dataset shifts collected from a web service that was actually in operation for several years.
In addition, the SHIFT28M dataset has several types of dataset shifts, allowing us to evaluate the robustness of the model to different types of shifts (e.g., covariate shift and target shift).

    Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

TODO

    Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number.

TODO