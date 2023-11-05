# Text detoxification
## Report about main ideas and solution path

### 1. Domain exploration

The fisrt stage of my solution of this assignment was a domain exploratio. First of all I read and examine provided article [1]. To get better understanding of the problem and to find more ideas for the solution I also get acquainted with some articles, such as [2] and [3], from the references.

I got several conclusions from this stage:

1. I have paralel dataset, which means I have supervised problem which is slightly different from the problem discussed in [1]. But the methods from [1] are also applicable.
2. I need to find a way to measure toxicity by myself. For example create a classifier. This is needed to for the evaluation of translated sentences.
3. I need to use pretrained models. For example models listed in [1].

### 2. Model chosing

After several unsuccsessfull tries I decided to stop on pretrained GeDi model proposed in [1]. The main reason for it was that it was pretrained specifically for this problem. But important to notice, that it was pretrained in unsupervised manner. 

As authors mentioned in their another article [3], some big pretrained models (such as Bert) with fine tunning around 3-4 epochs with paralel dataset tend to result better than GeDi. But the problem was that I did not have enough computational power. However trying some fined tunned models on paralel dataset may be a topic for the future research

I decided to create my own classifier of toxic sentences to evaluate the results of GeDi model. For that I used DistilBert with logistic regression. Code can be found here: https://www.kaggle.com/saidkamalov/pmldl-classifier . I trained it in kaggle and then just save the logistic regression.

As a baseline I decided to remove toxic words from the sentences. List of toxic word I got from [4]

### 3. Evaluation

For evaluation I used my trained classifier for toxicity and blue score for similarity.


## References
[1] Text Detoxification using Large Pre-trained Neural Models (https://arxiv.org/pdf/2109.08914.pdf)

[2]Proceedings of NAACL-HLT 2018, pages 129–140
New Orleans, Louisiana, June 1 - 6, 2018. c©2018 Association for Computational LinguisticsDear Sir or Madam, May I Introduce the GYAFC Dataset:
Corpus, Benchmarks and Metrics for Formality Style Transfer (https://aclanthology.org/N18-1012.pdf)

[3]ParaDetox: Detoxification with Parallel Data
(https://aclanthology.org/2022.acl-long.469.pdf)

[4] https://github.com/Orthrus-Lexicon/Toxic/blob/main/Toxic%20words%20dictionary.txt