# DyNet
### **Dynamic Pre-training: Towards Efficient and Scalable All-in-One Image Restoration**

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

#### [Akshay Dudhane](https://scholar.google.com/citations?user=BG_XEmkAAAAJ&hl=en), [Omkar Thawakar](https://github.com/OmkarThawakar/), [Sayed Waqas Zamir](https://github.com/swz30/), [Salman Khan](https://salman-h-khan.github.io/), [Ming-Hsuan Yang](https://scholar.google.com.pk/citations?user=p9-ohHsAAAAJ&hl=en) and [Fahad Khan](https://sites.google.com/view/fahadkhans/home)


#### **Mohamed bin Zayed University of AI, Inception Institute of AI, Australian National University, University of California - Merced, Yonsei University, Google Research, Linköping University**

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2404.02154.pdf)

## Latest 
- `2024/04/02`: We released our  on [arxiv](https://arxiv.org/pdf/2404.02154.pdf). Stay tuned for our Million-IRD dataset, code, and trained models.

<br>
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
All-in-one image restoration tackles different types of degradations with a unified model instead of having task-specific, non-generic models for each degradation. The requirement to tackle multiple degradations using the same model can lead to high-complexity designs with fixed configurations that lack the adaptability to more efficient alternatives. We propose DyNet, a dynamic family of networks designed in an encoder-decoder style for all-in-one image restoration tasks. Our DyNet can seamlessly switch between its bulkier and lightweight variants, thereby offering flexibility for efficient model deployment with a single round of training. This seamless switching is enabled by our weights-sharing mechanism, forming the core of our architecture and facilitating the reuse of initialized module weights. Further, to establish robust weights initialization, we introduce a dynamic pre-training strategy that trains variants of the proposed DyNet concurrently, thereby achieving a 50% reduction in GPU hours. To tackle the unavailability of a large-scale dataset required in pre-training, we curate a high-quality, high-resolution image dataset named Million-IRD, having 2M image samples. We validate our DyNet for image denoising, deraining, and dehazing in an all-in-one setting, achieving state-of-the-art results with 31.34% reduction in GFlops and a 56.75% reduction in parameters compared to baseline models
</details>

