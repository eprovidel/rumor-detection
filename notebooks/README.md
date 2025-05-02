# Notebooks

This folder contains all jupyter notebooks used throughout this work, including
the construction of embeddings, the execution of auxiliary tasks, as well as
the construction of CLMs and the evaluation of those.

A brief description of each is provided below:

```plain
.
├── BERT_torch_run_fnd.ipynb
├── README.md
├── TweetUtils.py
├── bot_detection.ipynb
├── create_embeddings.ipynb
├── create_embeddings_pluralismo.ipynb
├── fake_news_detection.ipynb
├── stance_classification.ipynb
└── zero_shot_evaluation.ipynb
```

| File | Description |
| --- | --- |
| [BERT_torch_run_fnd.ipynb](https://github.com/eprovidel/rumor-detection/blob/main/notebooks/BERT_torch_run_fnd.ipynb) | Used to train and store BERT models related to fake news detection |
| [TweetUtils.py](https://github.com/eprovidel/rumor-detection/blob/main/notebooks/TweetUtils.py) | Utilities to load and process Twitter posts |
| [bot_detection.ipynb](https://github.com/eprovidel/rumor-detection/blob/main/notebooks/bot_detection.ipynb) | Solves the Bot detection auxiliary task |
| [create_embeddings.ipynb](https://github.com/eprovidel/rumor-detection/blob/main/notebooks/create_embeddings.ipynb) | Used to create word embeddings for Twitter16 messages |
| [create_embeddings_pluralismo.ipynb](https://github.com/eprovidel/rumor-detection/blob/main/notebooks/create_embeddings_pluralismo.ipynb) | Used to create word embeddings for Disinformation-data messages |
| [fake_news_detection.ipynb](https://github.com/eprovidel/rumor-detection/blob/main/notebooks/fake_news_detection.ipynb) | Used to solve the fake news detection task using a BI-GRU neural network with the CLM-based embeddings |
| [stance_classification.ipynb](https://github.com/eprovidel/rumor-detection/blob/main/notebooks/stance_classification.ipynb) | Solves the Stance classification auxiliary task |
| [zero_shot_evaluation.ipynb](https://github.com/eprovidel/rumor-detection/blob/main/notebooks/zero_shot_evaluation.ipynb) | Used for zero-shot evaluation on Disinformation-data |
