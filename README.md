<div align="center">

# ✨LangScene-X: Reconstruct Generalizable 3D Language-Embedded Scenes with TriMap Video Diffusion✨

<p align="center">
<a href="https://liuff19.github.io/">Fangfu Liu</a><sup>1</sup>,
<a href="https://lifuguan.github.io/">Hao Li</a><sup>2</sup>,
<a href="https://github.com/chijw">Jiawei Chi</a><sup>1</sup>,
<a href="https://hanyang-21.github.io/">Hanyang Wang</a><sup>1,3</sup>,
<a href="https://github.com/liuff19/LangScene-X">Minghui Yang</a><sup>3</sup>,
<a href="https://github.com/liuff19/LangScene-X">Fudong Wang</a><sup>3</sup>,   
<a href="https://duanyueqi.github.io/">Yueqi Duan</a><sup>1</sup>
<br>
    <sup>1</sup>Tsinghua University, <sup>2</sup>NTU, <sup>3</sup>Ant Group     
</p>
<h3 align="center">ICCV 2025 🔥</h3>
<a href="https://github.com/liuff19/LangScene-X"><img src='https://img.shields.io/badge/arXiv-2507.02813-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://liuff19.github.io/LangScene-X"><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a><img src='https://img.shields.io/badge/License-MIT-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/chijw/LangScene-X"><img src='https://img.shields.io/badge/LangSceneX-huggingface-yellow'></a> &nbsp;&nbsp;&nbsp;&nbsp;

![Teaser Visualization](assets/teaser.png)

</div>

**LangScene-X:** We propose LangScene-X, a unified model that generates RGB, segmentation map, and normal map, enabling to reconstruct 3D field from sparse views input.

## 📢 News
- 🔥 [04/07/2025] We release "LangScene-X: Reconstruct Generalizable 3D Language-Embedded Scenes with TriMap Video Diffusion". Check our [project page](https://liuff19.github.io/LangScene-X) and [arXiv paper](https://arxiv.org/abs/2507.02813).

## 🌟 Pipeline

![Pipeline Visualization](assets/pipeline.png)

Pipeline of LangScene-X. Our model is composed of a TriMap Video Diffusion model which generates RGB, segmentation map, and normal map videos, an Auto Encoder that compresses the language feature, and a field constructor that reconstructs 3DGS from the generated videos. 


## 🎨 Video Demos from TriMap Video Diffusion

https://github.com/user-attachments/assets/55346d53-eb04-490e-bb70-64555e97e040

https://github.com/user-attachments/assets/d6eb28b9-2af8-49a7-bb8b-0d4cba7843a5

https://github.com/user-attachments/assets/396f11ef-85dc-41de-882e-e249c25b9961

## ⚙️ Setup

### 1. Clone Repository
```bash
git clone https://github.com/liuff19/LangScene-X.git
cd LangScene-X
```
### 2. Environment Setup

1. **Create conda environment**

```bash
conda create -n langscenex python=3.10 -y
conda activate langscenex
```
2. **Install dependencies**
```bash
conda install pytorch torchvision -c pytorch -y
pip install -e submodules/simple-knn
pip install -e submodules/segment-anything-langsplat
pip install -e auto-seg/submodules/segment-anything-1
pip install -e auto-seg/submodules/segment-anything-2
pip install -r requirements.txt
```

### 3. Model Checkpoints
The checkpoints of SAM, SAM2 and fine-tuned CogVideoX can be downloaded from our [huggingface repository](https://huggingface.co/chijw/LangScene-X).

## 💻Running

### Quick Start
You can start quickly by running the following scripts:
```bash
chmod +x quick_start.sh
./quick_start.sh <first_rgb_image_path> <last_rgb_image_path>
```
### Render
Run the following command to render from the reconstructed 3DGS field:
```bash 
python entry_point.py \
    pipeline.rgb_video_path="does/not/matter" \
    pipeline.normal_video_path="does/not/matter" \
    pipeline.seg_video_path="does/not/matter" \
    pipeline.data_path="does/not/matter" \
    gaussian.dataset.source_path="does/not/matter" \
    gaussian.dataset.model_path="output/path" \
    pipeline.selection=False \
    gaussian.opt.max_geo_iter=1500 \
    gaussian.opt.normal_optim=True \
    gaussian.opt.optim_pose=True \
    pipeline.skip_video_process=True \
    pipeline.skip_lang_feature_extraction=True \
    pipeline.mode="render"
```
You can also configurate by editting `configs/field_construction.yaml`.

## ✒️ TODO List
- [x] Per-scene Auto Encoder released
- [x] Fine-tuned CogVideoX checkpoints released
- [ ] Generalizable Auto Encoder (LQC)
- [ ] Improved TriMap Video Diffusion model

## 🔗Acknowledgement

We are thankful for the following great works when implementing LangScene-X:

- [CogVideoX](https://github.com/THUDM/CogVideo), [CogvideX-Interpolation](https://github.com/feizc/CogvideX-Interpolation), [LangSplat](https://github.com/minghanqin/LangSplat), [LangSurf](https://github.com/lifuguan/LangSurf), [VGGT](https://github.com/facebookresearch/vggt), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [SAM2](https://github.com/facebookresearch/sam2)

## 📚Citation

```bibtex
@misc{liu2025langscenexreconstructgeneralizable3d,
      title={LangScene-X: Reconstruct Generalizable 3D Language-Embedded Scenes with TriMap Video Diffusion}, 
      author={Fangfu Liu and Hao Li and Jiawei Chi and Hanyang Wang and Minghui Yang and Fudong Wang and Yueqi Duan},
      year={2025},
      eprint={2507.02813},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.02813}, 
}
```
