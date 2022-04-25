### Robust Self-augmentation for NER with Meta-reweighting

#### To Train
```
// 中文ontonote4 采样5%的数据
python train_reweight.py --cuda 1 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --update_step 1 --patient 3 --genre onto_5 --aug_genre onto_5_repl1 --train_type aug &> onto_5.log &
```

#### Higher
核心思想：将计算复杂的高阶导数简化成一阶近似（如一阶泰勒展开）

[a high-order optimization library for meta-learning](https://github.com/facebookresearch/higher)

#### Related Work
- [Learning to Reweight Examples for Robust Deep Learning](https://proceedings.mlr.press/v80/ren18a/ren18a.pdf)
- [Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting](https://proceedings.neurips.cc/paper/2019/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf)
- [Distilling Effective Supervision from Severe Label Noise](https://data.vision.ee.ethz.ch/cvl/webvision/videos-slides-2020/papers/cvpr/P1/paper.pdf)
- [Meta Soft Label Generation for Noisy Labels](https://arxiv.org/pdf/2007.05836.pdf)
- [Meta Label Correction for Noisy Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/17319/17126)
- [Semi-Supervised Learning with Meta-Gradient](http://proceedings.mlr.press/v130/xiao21a/xiao21a.pdf)
