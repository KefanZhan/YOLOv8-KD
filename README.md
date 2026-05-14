# *Awesome Knowledge Distillation Methods Implemented on YOLOv8*

---
## Supported KD Methods
Currently supported KD methods are as blew:
- Attention-based Feature-level Distillation [(AFD, AAAI-2021)](https://cdn.aaai.org/ojs/16969/16969-13-20463-1-2-20210518.pdf)
- Distilling Knowledge via Knowledge Review [(ReviewKD, CVPR-2021)](https://arxiv.org/pdf/2104.09044)
- Focal and Global Knowledge Distillation for Detectors[(FGD, CVPR-2022)](https://arxiv.org/abs/2111.11837)
- One-to-one Self-teaching [(OST, TGRS-2023)](https://ieeexplore.ieee.org/abstract/document/10175627)
- Cross-Head Knowledge Distillation for Dense Object Detection[(CrossKD, CVPR-2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.pdf)
- Knowledge Distillation via Cross Supervising with Attention for Remote Sensing Object Detection[(CSAKD, BMVC-2025)](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_398/paper.pdf)

---
## Hybrid Quantization
We also provide implementation of [*Hybrid Quantization*](https://ieeexplore.ieee.org/abstract/document/10175627), <br>
which can be utilized in combination with KD. <br>
To fix the quantization bit-width，you can set it in [here](ultralytics/engine/trainer.py).

--- 
## Train & Test
Please refer to [main.py](main.py), <br>
and we have pre-defined some extra args for KD in [here](ultralytics/cfg/default.yaml).

---
## Time
2025.6.22 open the code <br>
2025.7.4  update Hybrid Quantization
2026.5.15 update CSAKD （First academic project of my bachelor’s degree）
---
## Reference
1. https://github.com/ultralytics/ultralytics
2. https://github.com/clovaai/attention-feature-distillation
3. https://github.com/dvlab-research/ReviewKD
4. https://github.com/yzd-v/FGD
5. https://github.com/icey-zhang/GHOST
6. https://github.com/jbwang1997/CrossKD

---
## Acknowledgements
This code is built on [YOLOv8 (PyTorch)](https://github.com/ultralytics/ultralytics). We thank the authors for sharing the codes.<br>
Shout out to my advisor Hu and my teammates — I couldn’t have pulled this off without you all.<br>
May we all glow and thrive in our respective careers ahead.

---
## Contact
If you have any questions, please contact me by email (kefanzhan@smail.xtu.edu.cn).
