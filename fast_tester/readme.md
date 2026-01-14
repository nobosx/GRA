# The Accuracy of Fast Test
To generate samples and use fast test method to check the adversarial direction, use the command:

```shell
python ./attack_standard.py --dataset cifar10 --model resnet50 --method fast_test --norm inf --targeted 1 --test_num 1000 --max_query_count 100 --pool_num 1 --augment_method gaussian_noise
```

And use the following command to analyse the accuracy:

```shell
python ./accuracy_analysis.py
```