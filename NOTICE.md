# NOTICE — third-party data

The files under `data/` are **not** original to this repository and are **not**
covered by the MIT License that applies to the source code.

## Origin

The character images, the skeleton images derived from them, and the
ground-truth adjacency matrices derive from:

> **Graph-Based Evolutionary Dataset for Oracle Bone Characters**

which is distributed with the following statement (quoted from its `Readme.txt`):

> This file includes oracle bone characters represented by image, graph (matrix)
> and vectors.
>
> "Image" file includes 3,780 images of oracle bone characters come from the
> literature "Study on the evolution of Chinese characters based on few-shot
> learning: From oracle bone inscriptions to regular script".
>
> If you use these datasets, Please cite:
> Wang, M. et al. Study on the evolution of Chinese characters based on few-shot
> learning: From oracle bone inscriptions to regular script. PLoS ONE 17,
> e0272974 (2022).
>
> "Matrix" file includes 3,780 graphs (matrixes) of oracle bone characters.
>
> If you have any questions, please contact qjjiao@aynu.edu.cn.

## What this repository adds

* `data/skeleton_*` — skeletonised versions of the original character images.
* `data/Matrix_*` — the ground-truth adjacency matrices redistributed unchanged,
  together with the subsets used at the 756 / 1512 scales.
* `data/Image_3780` — the original character images, redistributed unchanged.

No new rights are claimed over the underlying character images or graphs.

## Required citation

If you use anything under `data/`, you must cite the original publication in
addition to the paper that accompanies this repository:

```bibtex
@article{wang2022oracle,
  title   = {Study on the evolution of Chinese characters based on few-shot
             learning: From oracle bone inscriptions to regular script},
  author  = {Wang, M. and others},
  journal = {PLoS ONE},
  volume  = {17},
  pages   = {e0272974},
  year    = {2022}
}
```

If you believe any file in this repository should not be redistributed, please
open an issue and it will be removed promptly.
