# *Awesome Knowledge Distillation Methods Implemented on YOLOv8*

---
## Supported KD Methods
Currently supported KD methods are as blew:
- Attention-based Feature-level Distillation [(AFD, AAAI-2021)](https://cdn.aaai.org/ojs/16969/16969-13-20463-1-2-20210518.pdf)
- Distilling Knowledge via Knowledge Review [(ReviewKD, CVPR-2021)](https://arxiv.org/pdf/2104.09044)
- Focal and Global Knowledge Distillation for Detectors[(FGD, CVPR-2022)](https://arxiv.org/abs/2111.11837)
- One-to-one Self-teaching [(OST, TGRS-2023)](https://ieeexplore.ieee.org/abstract/document/10175627)
- Cross-Head Knowledge Distillation for Dense Object Detection[(CrossKD, CVPR-2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.pdf)

--- 
## Train & Test
Please refer to [main.py](main.py), <br>
and we have pre-defined some extra args for KD in [here](ultralytics/cfg/default.yaml).

---
## Time
2025.6.22 open the code

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
This code is built on [YOLOv8 (PyTorch)](https://github.com/ultralytics/ultralytics). We thank the authors for sharing the codes.

---
## Contact
If you have any questions, please contact me by email (kefanzhan@smail.xtu.edu.cn).
