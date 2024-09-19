# UNIKD

This repository is the official implementation of **UNIKD**.

**[UNIKD: UNcertainty-filtered Incremental Knowledge Distillation for Neural Implicit Representation](https://dreamguo.github.io/projects/UNIKD/)**
<br/>
[Mengqi Guo](https://dreamguo.github.io/), [Chen Li](https://chaneyddtt.github.io/), [Hanlin Chen](https://hlinchen.github.io/), [Gim Hee Lee](https://www.comp.nus.edu.sg/~leegh/)
<br/>

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://dreamguo.github.io/projects/UNIKD/) [![arXiv](https://img.shields.io/badge/arXiv-2311.14603-b31b1b.svg)](https://arxiv.org/pdf/2212.10950)


## Abstract
> Recent neural implicit representations (NIRs) have achieved great success in the tasks of 3D reconstruction and novel view synthesis. However, they require the images of a scene from different camera views to be available for one-time training. This is expensive especially for scenarios with large-scale scenes and limited data storage. In view of this, we explore the task of incremental learning for NIRs in this work. We design a student-teacher framework to mitigate the catastrophic forgetting problem. Specifically, we iterate the process of using the student as the teacher at the end of each time step and let the teacher guide the training of the student in the next step. As a result, the student network is able to learn new information from the streaming data and retain old knowledge from the teacher network simultaneously. Although intuitive, naively applying the student-teacher pipeline does not work well in our task. Not all information from the teacher network is helpful since it is only trained with the old data. To alleviate this problem, we further introduce a random inquirer and an uncertainty-based filter to filter useful information. Our proposed method is general and thus can be adapted to different implicit representations such as neural radiance field (NeRF) and neural surface field. Extensive experimental results for both 3D reconstruction and novel view synthesis demonstrate the effectiveness of our approach compared to different baselines.

## 1. Installation

Pull repo.
```sh
git clone git@github.com:dreamguo/UNIKD.git
cd UNIKD
```

Create conda environment.
```sh
conda create -y -n UNIKD python=3.8
conda activate UNIKD

conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt
```

## 2. Usage

Download ICL and Replica datasets used in the paper [here](https://huggingface.co/datasets/dreamer001/UNIKD_dataset/blob/main/data.zip) and put them under `./data` folder. 


### Training.
Train on Replica Scan1 scene.
```sh
cd ./code
CUDA_VISIBLE_DEVICES=0 python training/exp_runner.py --conf confs/replica_mlp.conf --scan_id 1 --nepochs 2000 --training_type ours
```

Train on ICL Scan2 scene.
```sh
CUDA_VISIBLE_DEVICES=0 python training/exp_runner.py --conf confs/ICL_mlp.conf --scan_id 2 --gt_depth 1 --training_type ours
```


### Test.
Test on Replica Scan1 scene.
```sh
cd ./code
CUDA_VISIBLE_DEVICES=0 python training/exp_runner.py --conf confs/replica_mlp.conf --scan_id 1 --incre_timestamp <file_name> --training_type ours --infer 1
```

Test on ICL Scan2 scene.
```sh
CUDA_VISIBLE_DEVICES=0 python training/exp_runner.py --conf confs/ICL_mlp.conf --scan_id 2 --incre_timestamp <file_name> --training_type ours --infer 1
```


### Evaluation.
Evaluation on Replica scenes.
```sh
cd ./replica_eval
python evaluate_single_scene.py --input_mesh <path-to-ply-file>
```

Evaluation on ICL scenes.
```sh
cd ./ICL_eval
python evaluate_single_scene.py --input_mesh <path-to-ply-file>
```



## 3. Citation

If you make use of our work, please cite our paper:
```
@article{Guo2024UNIKD,
  author    = {Guo, Mengqi and Li, Chen and Chen, Hanlin and Lee, Gim Hee},
  title     = {UNIKD: UNcertainty-filtered Incremental Knowledge Distillation for Neural Implicit Representation},
  journal   = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
}
```

## 4. Ackowledgements

This project is built upon [MonoSDF](https://github.com/autonomousvision/monosdf). We use pretrained [Omnidata](https://omnidata.vision) for monocular depth and normal extraction. We thank all the authors for their great work and repos. 
