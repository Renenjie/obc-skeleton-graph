# NOTICE — third-party data

The files under `data/` are **not** original to this repository and are **not**
covered by the MIT License that applies to the source code.

## Origin

The character images, the ground-truth adjacency matrices and the dataset
scales used here all derive from the **Graph-Based Evolutionary Dataset for
Oracle Bone Characters (GBEDOBC)**, published as:

> Q. Jiao, J. Wu, Q. Liu, H. Zhang, Z. Zhang, B. Li, J. Xiong, G. Liu and
> Y. Liu, "A graph-based evolutionary dataset for oracle bone characters from
> inscriptions to modern Chinese scripts," *npj Heritage Science*, vol. 13,
> no. 1, Art. no. 369, 2025.
> DOI: [10.1038/s40494-025-01951-0](https://doi.org/10.1038/s40494-025-01951-0)

- Dataset repository: <https://github.com/BrisksHan/GBEDOBC>
- Dataset contact: qjjiao@aynu.edu.cn

The 3,780 glyph **images** distributed in that dataset are in turn taken from:

> M. Wang et al., "Study on the evolution of Chinese characters based on
> few-shot learning: From oracle bone inscriptions to regular script,"
> *PLoS ONE*, vol. 17, Art. no. e0272974, 2022.
> DOI: [10.1371/journal.pone.0272974](https://doi.org/10.1371/journal.pone.0272974)

This chain of provenance is stated verbatim in the dataset's own `Readme.txt`:

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

## What this repository takes from the dataset

| Directory | Relationship to GBEDOBC |
| --------- | ----------------------- |
| `data/Image_3780/` | original glyph images, redistributed **unchanged** |
| `data/Matrix_3780/` | ground-truth adjacency matrices, redistributed **unchanged** |
| `data/Matrix_756/`, `data/Matrix_1512/` | subsets of the 3780 graphs, redistributed **unchanged** |
| `data/skeleton_756/`, `skeleton_1512/`, `skeleton_3780/` | skeletonised versions of the GBEDOBC images, produced for this work |

No new rights are claimed over the underlying character images or graphs. The
skeleton images are derivative works of the GBEDOBC images and are distributed
under the same terms as the originals.

## Required citation

If you use anything under `data/`, you must cite the dataset publication **in
addition to** the paper that accompanies this repository:

```bibtex
@article{jiao2025gbedobc,
  title   = {A graph-based evolutionary dataset for oracle bone characters
             from inscriptions to modern Chinese scripts},
  author  = {Jiao, Qingju and Wu, Jingwen and Liu, Qi and Zhang, Han and
             Zhang, Zhan and Li, Bang and Xiong, Jing and Liu, Guoying and
             Liu, Yongge},
  journal = {npj Heritage Science},
  volume  = {13},
  number  = {1},
  pages   = {369},
  year    = {2025},
  doi     = {10.1038/s40494-025-01951-0}
}
```

If you redistribute or publish results based on the glyph images themselves,
please also cite the original image source:

```bibtex
@article{wang2022oracle,
  title   = {Study on the evolution of Chinese characters based on few-shot
             learning: From oracle bone inscriptions to regular script},
  author  = {Wang, M. and others},
  journal = {PLoS ONE},
  volume  = {17},
  pages   = {e0272974},
  year    = {2022},
  doi     = {10.1371/journal.pone.0272974}
}
```

If you believe any file in this repository should not be redistributed, please
open an issue and it will be removed promptly.
