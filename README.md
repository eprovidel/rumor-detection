# Cross-lingual cross-domain  transfer learning for rumor detection

_Authors_:

- Eliana Providel
- Marcelo Mendoza
- Mauricio Solar

In case of any questions please mail to eliana.providel@uv.cl

This repository presents all code used in the experiments of our work entitled
"Exploring Cross-Lingual Cascade-Learning Models for Fake News Detection in
Spanish". This work combines cascade-learning with the usage of multi-lingual
word embeddings in order to transfer learning from high-resource languaces, in
this case English, to lower-resource languages, in particular Spanish. More
specifically, it combines the tasks of _stance classification_ and _bot
detection_ as auxiliary tasks to the main tasks of _fake news detection_.

All models have been trained on different GPUs, with at least 12GB of memory,
using the [vast.ai](https://vast.ai) platform.

## Datasets

This work uses several datasets: Twitter16, RumorEval2019, PAN2019, and
_Disinformation data_. We describe them below.

### Twitter16

We use the Twitter16 dataset as published by Wei Gao et al. in [this
link](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0). The
folder contains both Twitter15 and Twitter16 datasets. As specified by the
original authors: _"Note that constrained by the terms of Twitter service, we
cannot contain the content of the rest of the tweets. Data users can obtain the
specifics based on the provided tweet IDs and uids by their own."_

### RumorEval2019

Dataset for the RumorEval competition of 2019, download link at:
https://figshare.com/articles/dataset/RumourEval_2019_data/8845580

### PAN2019

Dataset for the PAN2019 competition: _Author Profiling: Bots and Gender
Profiling_. The dataset is publicly available at
https://zenodo.org/records/3692340, however the files are restricted to users
that are granted access by the authors.

### Disinformation data

A recently-published dataset regarding disinformation data in Spanish.
Available at https://github.com/marcelomendoza/disinformation-data


