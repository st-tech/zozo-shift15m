# Datasheet for SHIFT15M

## Motivation
##### ```For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.```
Many machine learning algorithms assume that the training data and the test data are generated from the same distribution.
In the real world, however, this assumption is most often violated.
Many robust algorithms against such dataset shifts have been studied, but they often experiment with artificially induced dataset shifts on originally i.i.d. datasets.
Although such experiments seem reasonable, [recent studies](https://papers.nips.cc/paper/2020/hash/d8330f857a17c53d217014ee776bfd50-Abstract.html) have reported that there is no correlation between robustness to artificial dataset shifts and robustness to natural dataset shifts.
The main motivation of the SHIFT15M project is to provide a dataset that contains natural dataset shifts collected from a web service that was actually in operation for several years.
In addition, the SHIFT15M dataset has several types of dataset shifts, allowing us to evaluate the robustness of the model to different types of shifts (e.g., covariate shift and target shift).

##### ```Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?```
The initial version of the dataset was created by Masanari Kimura, Yuki Saito, Kazuya Morishita, Ryosuke Goto, and Takuma Nakamura most of whom were researchers at the ZOZO Research.

##### ```Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number.```
Not applicable.

##### ```Any other comments?```

## Composition
##### ```What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.```
The SHIFT15M dataset is a collection of outfits posted to the fashion website IQON (which is no longer providing this service). A record represents the posted outfit, the user who posted it, and some meta-information, it has 5 fields.
- set_id: An ID that identifies the outfit that was posted.
- items: Provides information about the items that comprise the posted outfit and consists of 4 subfields.
  - item_id: An ID that identifies an item.
  - category_id1: An ID indicating the item category (e.g., outerwear, tops, ...).
  - category_id2: An ID indicating the item subcategory (e.g., T-shirts, blouses, ...).
  - price: Price of the item (Japanese yen).
- user: Provides information about the user who posted the outfit and consists of 2 subfields.
  - An ID that identifies the user who posted the outfit.
  - A list of brands that users have voted for as their favorites. The number is an ID that identifies the brand.
- like_num: the number of times this outfit has been favorited by other users.
- publish_date: The date the outfit was posted.

##### ```How many instances are there in total (of each type, if appropriate)?```
The dataset consists of 15,218,721 item images and 2,555,147 outfits which created by users of IQON.

##### ```Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).```
We collected outfits posted on a Japanese fasion website "IQON". This website has about 2M users, almost all are Japanese women. Most of them are in their 20s and 30s. The collection period was from 01/01/2000 to 04/06/2020.
An outfit is a set of multiple items, and each item has a corresponding category. In SHIFT15M, outfits that contain 4 or more items belonging to the main categories (outerwear, tops, bottoms, shoes,bags, hats, and accessories) were collected.

##### ```What data does each instance consist of? “Raw” data (e.g., unprocessed text or images)or features? In either case, please provide a description.```
Each item consists of 4096-dimensional features extracted via the VGG16 model trained using the ILSVRC2012 dataset.

##### ```Is there a label or target associated with each instance? If so, please provide a description.```

Yes. Each instance has several numerical values (category ID, number of likes).
We can switch between several tasks by choosing one of these as the target variable.

##### ```Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.```
Items that do not belong to the main categories, such that underwear and background images for collage, are missing. The items field consists only of items that belong to the main categories, but the original outfit may contain items other than these.

##### ```Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)? If so, please describe how these relationships are made explicit.```
Each instance is assigned the ID of the user who submitted the outfit.

##### ```Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.```

SHIFT15M is a dataset with multiple dataset shifts observed in the real world.
We provide software that makes it easy to experiment with different types and sizes of shifts.
SHIFT15M was collected between 2010~2020, and our software allows automatic train/val/test splitting by specifying the shift type and magnitude.

##### ```Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.```

No.

##### ```Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.```

The dataset is self-contained.

##### ```Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctorpatient confidentiality, data that includes the content of individuals non-public communications)? If so, please provide a description.```

No.

##### ```Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.```

No.

##### ```Does the dataset relate to people? If not, you may skip the remaining questions in this section.```

Yes.
Each instance is a combination of outfits created by an individual and preferred by that individual.


##### ```Does the dataset identify any subpopulations (e.g., by age, gender)? If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.```

No.

##### ```Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.```

It is impossible to identify individuals from the dataset.

##### ```Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? If so, please provide a description.```

No.

##### ```Any other comments?```

Table 1 summarizes some dataset statistics and Figure 1 shows examples of items.
All images in the dataset are color.


| Property            | Value      |
|---------------------|------------|
| Number of Instances | 2,555,147  |
| Number of Items     | 15,218,721 |

> Table 1. A summary of dataset statistics.

> Figure 1. TODO

## Collection Process
##### ```How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.```

Except for the item attributes, the data was generated by users. Item attributes (category and price) were collected from e-commerce sites that sell the item. All data was viewable on the website.

##### ```What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)? How were these mechanisms or procedures validated?```

Users were able to create and publish their outfits using an editor provided by the website. The items selected in the editor are registered as an outfit, and this function was tested based on general software development procedures.

##### ```If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?```
We collected a complete dataset without sampling to create our dataset, except for data deleted by the user.

##### ```Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?```
The employees in ZOZO Technologies, Inc. or VASILY, inc. (merged into ZOZO Technologies, Inc.) were involved in collecting data.

##### ```Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.```

The dataset was collected in the period of 2010~2020.
Each outfit includes a timestamp that describes when the outfit created.

##### ```Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.```
No.

##### ```Does the dataset relate to people? If not, you may skip the remaining questions in this section.```

Yes.
Each instance is a combination of outfits created by an individual and preferred by that individual.

##### ```Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?```

Collected directly through the website.

##### ```Were the individuals in question notified about the data collection? If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.```

Notified in the Terms of Service.

![](./assets/iqon_terms_of_service/license.png)

##### ```Did the individuals in question consent to the collection and use of their data? If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.```

The use of the service was deemed as consent.

![](./assets/iqon_terms_of_service/introduction.png)

##### ```If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).```

It is possible to contact the company that provided the service.

https://tech.zozo.com/privacy/info/

##### ```Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis)been conducted? If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.```
No, there had been no potential impact analysis conducted.

##### ```Any other comments?```

## Preprocessing/cleaning/labeling

##### ```Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remainder of the questions in this section.```
We extracted the CNN features from images and treated them as input data in our image-based tasks. As a result, our dataset contains the features but does not include raw photos, making them anonymized.
The CNN we used is an official pre-trained VGG16, and we adopted the outputs of the 'fc6' layer before applying ReLU as the feature.
We used the Chainer implementation for extracting CNN features. For more information on the Chainer implementation, please refer to the reference page:
https://docs.chainer.org/en/v7.8.0/reference/generated/chainer.links.VGG16Layers.html
We exclude the outfits that contain less than four items. Other than that, we did not remove any instances in creating our dataset. However, we excluded some data in each independent task. In detail, please refer to each task description.

##### ```Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data.```
No.

##### ```Is the software used to preprocess/clean/label the instances available? If so, please provide a link or other access point.```
All software are provided on the [SHIFT15M repository](https://github.com/st-tech/zozo-shift15m).

##### ```Any other comments?```

## Uses

##### ```Has the dataset been used for any tasks already? If so, please provide a description.```

Benchmarks using this dataset and the specified evaluation protocol are listed in https://github.com/st-tech/zozo-shift15m/tree/main/benchmarks.

##### ```Is there a repository that links to any or all papers or systems that use the dataset? If so, please provide a link or other access point.```

All benchmarks that use this dataset will be available at https://github.com/st-tech/zozo-shift15m/tree/main/benchmarks.

##### ```What (other) tasks could the dataset be used for?```

##### ```Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks) If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?```
No.

##### ```Are there tasks for which the dataset should not be used? If so, please provide a description.```
This dataset is distributed in a way that excluding raw images and anonymizing the users/brands. Therefore, it requires the dataset users not to reconstruct raw images from the image features or restore the anonymized parts in a future task.

##### ```Any other comments?```

## Distribution

##### ```Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? If so, please provide a description.```

Yes. The dataset will be distributed to third parties based on the licence.

##### ```How will the dataset will be distributed (e.g., tarball on website, API, GitHub)? Does the dataset have a digital object identifier (DOI)?```

##### ```When will the dataset be distributed?```
The dataset will be first released in August 2021.

##### ```Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.```
The dataset will be distributed under the license of CC BY-NC 4.0. In detail, please refer to https://creativecommons.org/licenses/by-nc/4.0/

##### ```Have any third parties imposed IP-based or other restrictions on the data associated with the instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.```
There are no fees or restrictions.

##### ```Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.```

Unknown

##### ```Any other comments?```

## Maintenance

##### ```Who will be supporting/hosting/maintaining the dataset?```
ZOZO Research is supporting/maintaining the dataset.

##### ```Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?```
All changes to the dataset will be announced through the GitHub Releases.

##### ```If the dataset becomes obsolete how will this be communicated?```
This will be posted on the dataset webpage.

##### ```How can the owner/curator/manager of the dataset be contacted (e.g., email address)?```
All questions and comments can be sent to GitHub Issues on [SHIFT15M repository](https://github.com/st-tech/zozo-shift15m).

##### ```Is there an erratum? If so, please provide a link or other access point.```
All changes to the dataset will be announced through the GitHub Releases.
Errata are listed under the “Errata” section of [SHIFT15M repository](https://github.com/st-tech/zozo-shift15m).


##### ```If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)? If so, please describe these limits and explain how they will be enforced.```
No.

##### ```Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to users.```
They will continue to be supported with all information on [SHIFT15M repository](https://github.com/st-tech/zozo-shift15m).
We also provide the contribution guides for software that supports the dataset.

##### ```If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.```
Others may do so and should contact the original authors about incorporating fixes/extensions.

##### ```Any other comments?```

