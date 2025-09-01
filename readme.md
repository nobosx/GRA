# GuidedRay Attack

This is the code repository of the paper "GuidedRay: Surging Decision-based Targeted Adversarial Attacks via Diversity-Oriented Sampling". Via this repository, users can reproduce the experimental results reported in our paper, such as conducting adversarial attack trials for different attack methods and several ablation experiments for our GuidedRay attack.

## Requirement

To run this code, you need to install PyTorch to run neural network models, as well as mmpretrain to load pretrained weights from mmlab.

The code has been tested on the following environment:

* python == 3.8.20
* pytorch == 1.10.1
* torchvision == 0.11.2
* cudatoolkit == 11.3.1
* numpy == 1.24.3
* tqdm == 4.66.5
* openmim == 0.3.9
* mmcv == 2.2.0
* mmpretrain == 1.2.0

If you are using Anaconda to arrange your Python environments, the following commands might work to prepare the code environment:

```shell
# Create a new python environment
conda create -n GuidedRay -y python=3.8
conda activate GuidedRay
# Install pytorch
conda install pytorch torchvision -c pytorch -y
# Install mmcv and mmpretrain
pip install mmcv mmpretrain
```

## The Code Structure

The code is written in Python.

An adversarial attack is composed of several modules, which are separately wrapped in different Python classes or files.

* An **Attacker** represents a specific attack method. All Attackers are organized in the folder `./attackers` and we have implemented our GuidedRay as well as all the baseline methods mentioned in our paper. One can inherit the `BasicAttacker` class can implement its `attack` method to build a customized Attacker.
* The file `data_sets.py` provides methods of loading different datasets.
* An undefended model is wrapped into the `VictimModel` class in `victim_model.py`. An model with a defense mechanism is wrapped into the `DefensiveModel` class in `defensive_model.py`.
* The class `AttackProcess` in `evaluator.py` can combine the above modules, implement the concrete procedure of adversarial attack trials and save corresponding attack records.
* The class `ResultParser` can parse attack records and get the results of ASR (Attack Success Rate), AvQ (Average Queries), MeQ (Median Queries), etc.
* Finally, there are several scripts, with file names starting by "attack\_" or "analyze\_", as user interfaces.

## Conducting Attack Experiments

There are some example commands to conduct attacks for 1,000 trials on the Cifar10 dataset and ResNet50 model, with a query limit of 10,000:

```shell
# Targed scenario
python ./attack_standard.py --dataset cifar10 --model resnet50 --method HSJA --norm inf --targeted 1 --test_num 1000 --max_query_count 10000
python ./attack_standard.py --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 10000 --pool_num 1 --augment_method gaussian_noise

# Untarged scenario
python ./attack_standard.py --dataset cifar10 --model resnet50 --method RayS --norm inf --targeted 0 --test_num 1000 --max_query_count 10000
python ./attack_standard.py --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 0 --test_num 1000 --max_query_count 10000 --augment_method mix
```

For the GuidedRay method, the parameter `--pool_num` is the number of adversarial examples used for guidance (i.e. the parameter $k$ in the paper) and `--augment_method` is the method used in the Adversarial Example Augmentation Framework, with `gaussian_noise` for GuidedRay-G and `mix` for GuidedRay-M.

To run attacks under other settings, just modify the corresponding parameters.

To run attacks on defensive models, one can run the following commands:

```shell
# Adversarial Training
python ./attack_defensive.py --dataset cifar10 --model resnet50 --defense_model adv_train --method Tangent --norm inf --targeted 1 --test_num 1000 --max_query_count 10000
python ./attack_defensive.py --dataset cifar10 --model resnet50 --defense_model adv_train --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 10000 --pool_num 1 --augment_method mix
# TRADES
python ./attack_defensive.py --dataset cifar10 --model resnet50 --defense_model TRADES --method Tangent --norm inf --targeted 1 --test_num 1000 --max_query_count 10000
python ./attack_defensive.py --dataset cifar10 --model resnet50 --defense_model TRADES --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 10000 --pool_num 1 --augment_method mix
```

To test GuidedRay attacks with different numbers of adversarial examples for guidance or with different image augmentation methods, run the following commands:

```shell
# Test different numbers of adversarial examples for guidance
python ./attack_standard.py --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 5000 --pool_num 1 --augment_method mix
python ./attack_standard.py --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 5000 --pool_num 3 --augment_method mix
python ./attack_standard.py --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 5000 --pool_num 5 --augment_method mix
python ./attack_standard.py --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 5000 --pool_num 10 --augment_method mix

# Test different image augmentation methods
# Gaussain noise
python ./attack_standard.py --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 5000 --pool_num 1 --augment_method gaussian_noise
# Random rotation
python ./attack_standard.py --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 5000 --pool_num 1 --augment_method rotation
# Random cropping
python ./attack_standard.py --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 5000 --pool_num 1 --augment_method crop
# Color jitter
python ./attack_standard.py --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 5000 --pool_num 1 --augment_method color_jitter
# Mixture strategy
python ./attack_standard.py --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 5000 --pool_num 1 --augment_method mix
```

When an attack experiment is completed, the attack records will be saved to the folder `./attack_records`. Then, one can run the following commands to see the attack results:

```shell
# Print the ASR for epsilon threshold 0.02 at different query limits of different attack methods
# Standard model
python ./analyze_results.py print_ASR --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 10000 --pool_num 1 --augment_method mix --query_limit 500 1000 3000 5000 10000 --epsilons 0.02
# Defensive model
python ./analyze_results.py print_ASR --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 10000 --pool_num 1 --augment_method mix --query_limit 1000 5000 10000 --epsilons 0.02  --defense_model adv_train

# Print AvQ and MeQ
python ./analyze_results.py print_early_stop_query --dataset cifar10 --model resnet50 --method GuidedRay --norm inf --targeted 1 --test_num 1000 --max_query_count 10000 --pool_num 1 --augment_method mix --query_limit 10000 --epsilons 0.02
```

## Testing the Diversity of New Directions (Section 6.4.2 in the paper)

We write additional scripts to test and analyze the diversity of new directions generated by RayS and GuidedRay.

The script `attack_gen_directions.py` can generate Ray directions for different attacks and analyze the diversity of these new directions. Please run the following commands:

```shell
python ./attack_gen_directions.py --dataset cifar10 --model resnet50 --test_num 1000 --num_directions 1000
python ./attack_gen_directions.py --dataset cifar100 --model resnet50 --test_num 1000 --num_directions 1000
python ./attack_gen_directions.py --dataset imagenet --model densenet121 --test_num 1000 --num_directions 1000
```

The script `attack_test_init.py` can test the effectiveness of the initialization phase of each method. Please run the following commands:

```shell
python ./attack_test_init.py --dataset cifar10 --model resnet50 --test_num 1000 --max_query_count 10000
python ./attack_test_init.py --dataset cifar100 --model resnet50 --test_num 1000 --max_query_count 10000
python ./attack_test_init.py --dataset imagenet --model densenet121 --test_num 1000 --max_query_count 10000
```

## Downloading the ImageNet Dataset

If you want to run attacks on the ImageNet dataset, please first download the  dataset file `imagenet.zip` from https://drive.google.com/file/d/1TI-Hp97pRCGANPTfr7kh2h56XbCvfS2z/view?usp=sharing and unzip it into the folder `./data`.
