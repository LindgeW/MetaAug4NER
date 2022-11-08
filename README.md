### Robust Self-augmentation for NER with Meta-reweighting

This repository contains the code for [Robust Self-Augmentation for Named Entity Recognition with Meta Reweighting](https://arxiv.org/pdf/2204.11406.pdf)  (NAACL2022).

#### Requirements
+ Python >= 3.6
+ Torch >= 1.3
+ transformers >= 4.0
+ [higher](https://github.com/facebookresearch/higher)
    + Core Thought: the complex calculation of higher-order gradients is simplified to a first-order approximation (e.g., to do the first-order Taylor expansion)

#### Prepare
 1. Get partial training set: `python processing/sample.py 0.05|0.1|0.3`
 2. Build the entity dictionary: `python processing/build_ner_dic.py train_data_file ent.dic cn|en`
 3. Obtain the word-to-vectors training on [Wikipedia](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/)
 4. Produce pseudo-labeled training set：`python processing/cn|en_aug_util.py train_data_file aug_train_data_file ent.dic ratio aug_times`
 
&ensp;&ensp;Note: The data format is *BIOES* CoNLL. The `processing/conll_util.py` script provides the format transformation.

#### Related Work
- [Learning to Reweight Examples for Robust Deep Learning](https://proceedings.mlr.press/v80/ren18a/ren18a.pdf)
- [Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting](https://proceedings.neurips.cc/paper/2019/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf)
- [Distilling Effective Supervision from Severe Label Noise](https://data.vision.ee.ethz.ch/cvl/webvision/videos-slides-2020/papers/cvpr/P1/paper.pdf)
- [Meta Soft Label Generation for Noisy Labels](https://arxiv.org/pdf/2007.05836.pdf)
- [Meta Label Correction for Noisy Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/17319/17126)
- [Semi-Supervised Learning with Meta-Gradient](http://proceedings.mlr.press/v130/xiao21a/xiao21a.pdf)
