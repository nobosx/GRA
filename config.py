DATASET_NAME = ["cifar10", "cifar100", "imagenet"]

MODEL_NAME = ["resnet50", "densenet121"]

SUPPORTED_DATAMODELS = ["cifar10-resnet50", "cifar100-resnet50", "imagenet-densenet121"]

SUPPORTED_ATTACK_METHOD = ["RayS", "GuidedRay", "HSJA", "bounce", "Tangent", "Sign_OPT_Linf", "RayST"]

SUPPORTED_DEFENSE_MODEL = ['adv_train', 'TRADES']

SUPPORTED_ADV_TRAIN_DATAMODELS = ['cifar10-resnet50']

SUPPORTED_TRADES_DATAMODELS = ['cifar10-resnet50']

INPUT_CHANNELS = {
    "cifar10": 3,
    "cifar100": 3,
    "imagenet": 3,
}

IMAGE_SIZE = {
    "cifar10": (32, 32),
    "cifar100": (32, 32),
    "imagenet": (224, 224),
}

CLASS_NUM = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet": 1000,
}