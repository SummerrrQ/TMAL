# Boosting Few-Shot Action Recognition via Time-enhanced Multimodal Adaptation Learning（TMAL）


![屏幕截图 2024-08-08 143408](https://github.com/user-attachments/assets/a1719c3a-66da-4666-9a72-e540e3a11324)
> **Boosting Few-Shot Action Recognition via Time-enhanced Multimodal Adaptation Learning**<br>
>
>
>
>
>* We introduce a novel and effective multimodal adaptation learning framework for few-shot action recognition that incorporates both spatial and temporal information, while also leveraging semantic knowledge to enhance action comprehension.
>* We implement the temporal adaptation to adapt the image model to the video tasks by incorporating lightweight adapters
>* We develop the hierarchical visual-text and visual-flow fusions to effectively integrate multimodal cues for comprehensive representation learning. In particular, the incorporation of optical flow further enhances temporal modeling in videos, which is critical for action analysis.
>* We conduct extensive experiments to showcase the potential and efficacy of the proposed approach, which consistently achieves comparable results across four challenging public benchmark datasets.
>
>

As our paper is currently under review, we will promptly release the code once it is accepted...
## Environment

We use conda to manage the Python environment. The dumped configuration is provided at [environment.yaml](environment.yaml)

## Data preparation

- [SSV2](https://20bn.com/datasets/something-something#download)
- [Kinetics](https://github.com/Showmax/kinetics-downloader)
- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

Thank you to the team of [CLIP-FSAR] (https://github.com/alibaba-mmai-research/CLIP-FSAR)for providing the [splits](configs/projects/CLIPFSAR)

## Backbone preparation

We use the CLIP checkpoints from the [official release](https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/clip.py#L30).we used CLIP-ViT-B/16 as our backbone.

## Acknowledgements

This code is based on [CLIP-FSAR](https://github.com/alibaba-mmai-research/CLIP-FSAR), [ST-Adapter](https://github.com/linziyi96/st-adapter) codebase, which  provids with innovative ideas in comprehensive video understanding for video classification and temporal modeling. Some flow data processing code comes from [Flownet2.0](https://github.com/NVIDIA/flownet2-pytorch) .Thanks for their awesome works!
