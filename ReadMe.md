# Enhancing the Self-Universality for Transferable Targeted Attacks 
This paper has been accepted by CVPR2023. [[Paper]](https://arxiv.org/abs/2209.03716).

If you use our method for attacks in your research, please consider citing
```
@inproceedings{wei2023enhancing,
  title={Enhancing the Self-Universality for Transferable Targeted Attacks},
  author={Wei, Zhipeng and Chen, Jingjing and Wu, Zuxuan and Jiang, Yu-Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

# Preparation
## Environment
```
conda create -n env_name python=3.9
source activate env_name
pip install -r requirements.txt
```
## Dataset
Download the ImageNet-compatible dataset from [cleverhans](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset) or [ZhengyuZhao](https://github.com/ZhengyuZhao/Targeted-Transfer).
## Model
The pre-trained ResNet50, DenseNet121, VGG16_bn, and Inception-v3 are from Pytorch.
## Variable
Need to specify some variables in [utils.py](utils.py).
|variable| description|
|:------:|:------:|
|OPT_PATH	|The path of saving outputs|
|NIPS_DATA|	The path of ImageNet-compatible dataset|


# Motivation
```
python motivation_run.py
```
Run cells in [local_motivation.ipynb](local_motivation.ipynb) and [feature_motivation.ipynb](feature_motivation.ipynb) to draw Figure 2(a) and (b), respectively.

# Performance Comparison
## Single-model transferable attacks
Perform the baseline attak: DTMI.
```
python main.py --gpu {gpu} --white_box {model} --loss_fn {loss_fn} --attack DTMI --target --MI --TI --DI --no-saveperts --file_tailor baseline
```
Perform the combinational attack of DTMI and SU
```
python main.py --gpu {gpu} --white_box {model} --loss_fn {loss_fn} --attack DTMI_Local_FeatureSimilarityLoss --target --MI --TI --DI --no-saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --file_tailor ours
```

## Ensemble model transferable attacks
Perform the baseline attak: DTMI.
```
python ensemble_main.py --gpu {gpu} --black_box {model} --loss_fn {loss_fn} --attack Ensemble_DTMI --target --MI --TI --DI --no-saveperts --file_tailor ensemble_baseline --batch_size 10
```
Perform the combinational attack of DTMI and SU
```
python ensemble_main.py --gpu {gpu} --black_box {model} --loss_fn {loss_fn} --attack ENSEMBLE_DTMI_Local_FeatureSimilarityLoss --target --MI --TI --DI --no-saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --file_tailor ensemble_ours --batch_size 5
```
| paramter      | values |
| :----------: | :-----------: |
| model   | 'inception_v3', 'resnet50', 'densenet121', 'vgg16_bn' |
|loss_fn | 'CE', 'Logit'|

## Combination with existing methods
ODI-TMI
```
python main.py --gpu {gpu} --white_box resnet50 --loss_fn Logit --attack TMI_ODI --target --MI --TI --DI --saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --batch_size 1 --part 200 --part_index {part_index} --file_tailor odi_tmi
```
ODI-TMI-SU
```
python main.py --gpu {gpu} --white_box resnet50 --loss_fn Logit --attack DTMI_Local_FeatureSimilarityLoss_ODI --target --MI --TI --DI --no-saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --batch_size 1 --part 200 --part_index {part_index} --file_tailor odi_tmi_su
```
The parameters ''part'' and ''part_index'' divide the dataset the specified number of parts. And the part with the specified index will be attacked.

DTMI-EMI
```
python main.py --gpu {gpu} --white_box resnet50 --loss_fn Logit --attack DTEMI --target --MI --TI --DI --no-saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --file_tailor dtmi_emi
```

DTMI-EMI-SU
```
python main.py --gpu {gpu} --white_box resnet50 --loss_fn Logit --attack DTMI_Local_FeatureSimilarityLoss_EMI --target --MI --TI --DI --no-saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --file_tailor dtmi_emi_su
```

DTMI-SI
```
python main.py --gpu {gpu} --white_box resnet50 --loss_fn Logit --attack DTMI_SI --target --MI --TI --DI --no-saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --batch_size 5 --file_tailor dtmi_si
```
DTMI-SI-SU
```
python main.py --gpu {gpu} --white_box resnet50 --loss_fn Logit --attack DTMI_Local_FeatureSimilarityLoss_SI --target --MI --TI --DI --no-saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --batch_size 5 --file_tailor dtmi_si_su
```
DTMI-Admix
```
python main.py --gpu {gpu} --white_box resnet50 --loss_fn Logit --attack DTMI_Admix --target --MI --TI --DI --no-saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --batch_size 1 --file_tailor dtmi_admix
```
DTMI-Admix-SU
```
python main.py --gpu {gpu} --white_box resnet50 --loss_fn Logit --attack DTMI_Local_FeatureSimilarityLoss_Admix --target --MI --TI --DI --no-saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --batch_size 1 --file_tailor dtmi_admix_su
```

# Ablation Study
## The scale parameter $s_l, s_{int}$
```
python main.py --gpu {gpu} --white_box densenet121 --loss_fn CE --attack DTMI_Local_FeatureSimilarityLoss --target --MI --TI --DI --no-saveperts --scale_start {start} --scale_interval {interval} --fsl_coef 1.0 --depth 3 --file_tailor scale_{start}_{interval}
```
We fix the weighted parameter $\lambda=1.0$, and the feature extraction layer as layer 3 of Table 1. 
| paramter      | values |
| :----------: | :-----------: |
| start: $s_l$      | 0.1,0.3,0.5,0.7       |
| interval: $s_{int}$   | 0.0,0.1,0.2,0.3        |
## The weighted parameter $\lambda$ and different layers
```
python main.py --gpu {gpu} --white_box densenet121 --loss_fn CE --attack DTMI_Local_FeatureSimilarityLoss --target --MI --TI --DI --no-saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef {coef} --depth {depth} --file_tailor coef_depth_{coef}_{depth}
```
| paramter      | values |
| :----------: | :-----------: |
| coef: $\lambda$      | $10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 10^{0}$   |
| depth: layers   | 1,2,3,4|
## Effect of components
No Local and Feature Similarity Loss
```
python main.py --gpu {gpu} --white_box {model} --loss_fn CE --attack DTMI --target --MI --TI --DI --no-saveperts --file_tailor no_local_fsl
```

Only Local
```
python main.py --gpu {gpu} --white_box {model} --loss_fn CE --attack DTMI_Local_FeatureSimilarityLoss --target --MI --TI --DI --no-saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0 --depth 3 --file_tailor wo_fsl
```


Local and Feature Similarity Loss
```
python main.py --gpu {gpu} --white_box {model} --loss_fn CE --attack DTMI_Local_FeatureSimilarityLoss --target --MI --TI --DI --no-saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --file_tailor local_fsl
```
| paramter      | values |
| :----------: | :-----------: |
| model   | 'inception_v3', 'resnet50', 'densenet121', 'vgg16_bn' |

## Effect of different regions
Center
```
python main.py --gpu {gpu} --white_box densenet121 --loss_fn CE --attack DTMI_Local_FeatureSimilarityLoss_Center --target --MI --TI --DI --no-saveperts --usecenter --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --file_tailor center
```
Corner
```
python main.py --gpu {gpu} --white_box densenet121 --loss_fn CE --attack DTMI_Local_FeatureSimilarityLoss_Center --target --MI --TI --DI --no-saveperts --no-usecenter --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --file_tailor corner
```
Uniform
```
python main.py --gpu {gpu} --white_box densenet121 --loss_fn CE --attack DTMI_Random_FeatureSimilarityLoss --target --MI --TI --DI --no-saveperts --scale_start 0.1 --scale_interval 0.0 --fsl_coef 0.001 --depth 3 --file_tailor uniform
```
