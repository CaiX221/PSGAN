# PSGAN
Code for our CVPR 2020 **oral** paper "[PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer](https://arxiv.org/abs/1909.06956)".

Contributed by [Wentao Jiang](https://wtjiang98.github.io), [Si Liu](http://colalab.org/people), Chen Gao, Jie Cao, Ran He, [Jiashi Feng](https://sites.google.com/site/jshfeng/), [Shuicheng Yan](https://www.ece.nus.edu.sg/stfpage/eleyans/).

This code was further modified by [Zhaoyi Wan](https://www.wanzy.me).

In addition to the original algorithm, we added high-resolution face support using Laplace tranformation.

![](psgan_framework.png)

## Our Modifications (COMS3168BC Final Project)

This fork extends the original PSGAN with a **dense structural pathway** inspired by the structural ControlNet of Stable-Makeup (SIGGRAPH 2025). Whereas the original PSGAN only consumes 68 `dlib` landmarks as sparse coordinates inside the AMM module, our version additionally feeds a colored **SPIGA structure map** of the source face through a new lightweight `StructureEncoder`, whose features are added directly into TNet's bottleneck. This improves identity preservation without changing the GAN backbone.

The work is organized across two branches:

- **`master`** — model architecture, data preparation, and training-side changes.
- **`feat/spiga-inference`** — inference-side changes so that `demo.py` can run on the modified model.

### What changed (master branch)

- **`psgan/net.py`** — added a `StructureEncoder` class (3-layer CNN, `(3,256,256) → (256,64,64)`); extended `Generator.forward` with a `structure_map` keyword argument, injected after the SPADE modulation in the bottleneck.
- **`generate_spiga.py`** *(new)* — preprocessing script that renders 68-point SPIGA landmarks into colored 256×256 RGB structure maps (jawline=lime, brows=yellow, nose=orange, eyes=magenta, lips=cyan/blue). Failed detections produce a black-image fallback.
- **`tools/data_reader.py`** — `read_file` now also loads the SPIGA map alongside image / seg / dlib landmarks.
- **`data_loaders/makeup_dataloader.py`** — items returned to the trainer are now 4-tuples `[image, mask, dist, spiga]`.
- **`psgan/solver.py`** — `Solver.train` unpacks the new `spiga_s, spiga_r`; `load_checkpoint` now loads the original PSGAN weights with `strict=False` so the new `StructureEncoder` initializes from scratch (Xavier) while the rest of the network reuses the pretrained backbone.

### What changed (feat/spiga-inference branch)

- **`psgan/preprocess.py`** — imports the SPIGA functions from `generate_spiga.py` so training and inference distributions stay consistent. New `_generate_spiga_map` helper with the same black-image fallback. `__call__` now appends a SPIGA tensor as the 4th element of its returned tuple.
- **`psgan/solver.py`** — `Solver.test` and `Solver.generate` accept and forward the new `structure_map` argument.
- **`psgan/inference.py`** — unchanged (the existing star-unpacking propagates the new tensor automatically).

### Updated data layout

```
data
├── images/      (existing) makeup/, non-makeup/
├── segs/        (existing) makeup/, non-makeup/
├── landmarks/   (existing) makeup/, non-makeup/   -- dlib 68-point
└── spiga/       (new)      makeup/, non-makeup/   -- SPIGA structure maps
```

Generate the new `spiga/` subtree with `python3 generate_spiga.py` before training.

## Checklist
- [x] more results 
- [ ] video demos
- [ ] partial makeup transfer example
- [ ] interpolated makeup transfer example
- [x] inference on GPU
- [x] training code

## Requirements
The code was tested on Ubuntu 16.04, with Python 3.6 and PyTorch 1.5.

For face parsing and landmark detection, we use dlib for fast implementation.

For our added structural pathway, we additionally require [SPIGA](https://github.com/andresprados/SPIGA) for refined facial landmark detection.

If you are using gpu for inference, *do* make sure you have gpu support for dlib.

## Test
Checkout the `feat/spiga-inference` branch first, then run `python3 demo.py` or `python3 demo.py --device cuda` for gpu inference.

## Train
1. Download training data from [link](https://drive.google.com/drive/folders/1ubqJ49ev16NbgJjjTt-Q75mNzvZ7sEEn?usp=sharing), and move it to sub directory named with "data". (For BaiduYun users, you can download the data [here](https://pan.baidu.com/s/1ZF-DN9PvbBteOSfQodWnyw). Password: rtdd)

Your data directory should be looked like:
```
data
├── images
│   ├── makeup
│   └── non-makeup
├── landmarks
│   ├── makeup
│   └── non-makeup
├── makeup.txt
├── non-makeup.txt
├── segs
│   ├── makeup
│   └── non-makeup
```

2. Generate SPIGA structure maps for both splits: `python3 generate_spiga.py`. This will populate `data/spiga/makeup/` and `data/spiga/non-makeup/`.

3. `python3 train.py`

Detailed configurations can be located and modified in configs/base.yaml, where
command-line modification is also supportted.

*Note: * Although multi-GPU training is currently supported, due to the limitation of pytorch data parallel and gpu cost, the numer of
adopted gpus and batch size are supposed to be the same.

## More Results

#### MT-Dataset (frontal face images with neutral expression)
![](MT-results.png)

#### MWild-Dataset (images with different poses and expressions)
![](MWild-results.png)

#### Video Makeup Transfer (by simply applying PSGAN on each frame)
![](Video_MT.png)

## Citation
Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the url LaTeX package.

~~~
@InProceedings{Jiang_2020_CVPR,
  author = {Jiang, Wentao and Liu, Si and Gao, Chen and Cao, Jie and He, Ran and Feng, Jiashi and Yan, Shuicheng},
  title = {PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}

@article{liu2021psgan++,
  title={PSGAN++: robust detail-preserving makeup transfer and removal},
  author={Liu, Si and Jiang, Wentao and Gao, Chen and He, Ran and Feng, Jiashi and Li, Bo and Yan, Shuicheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={11},
  pages={8538--8551},
  year={2021},
  publisher={IEEE}
}
~~~

## Acknowledge
Some of the codes are built upon [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) and [BeautyGAN](https://github.com/wtjiang98/BeautyGAN_pytorch). 

You are encouraged to submit issues and contribute pull requests.
