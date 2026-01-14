from config import SUPPORTED_DEFENSE_MODEL, CLASS_NUM, SUPPORTED_ADV_TRAIN_DATAMODELS, SUPPORTED_TRADES_DATAMODELS
from models.resnet import resnet
import torch
import os

class DefensiveModel(object):
    def __load_weight_from_pth_checkpoint(self, model, load_weights_path):
        assert (load_weights_path is not None) and os.path.exists(load_weights_path), f"Model weight path {load_weights_path} does not exist!"
        raw_state_dict = torch.load(load_weights_path, map_location='cpu')['state_dict']
        state_dict = dict()
        for key, val in raw_state_dict.items():
            new_key = key.replace('module.', '').replace('cnn.', "")  # FIXME why cnn?
            state_dict[new_key] = val
        model.load_state_dict(state_dict)

    def __load_cifar10_model(self, model_name, load_weights_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == 'resnet50':
            model = resnet(depth=50, num_classes=CLASS_NUM['cifar10'], block_name='BasicBlock')
            self.__load_weight_from_pth_checkpoint(model, load_weights_path)
            # Normalization for cifar10-resnet50
            self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
            self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
        return model

    def __init__(self, model_name, dataset_name, defense_model):
        assert defense_model in SUPPORTED_DEFENSE_MODEL, f"Defense model {defense_model} is not supported!"
        self.model_name = model_name
        self.dataset_name = dataset_name
        datamodel = dataset_name + "-" + model_name
        if defense_model == 'adv_train':
            assert datamodel in SUPPORTED_ADV_TRAIN_DATAMODELS, f"Data-model {datamodel} is not supported for adv_train defensive model!"
            load_weights_path = f"./models/defensive_models/adv_train/{datamodel}.pth.tar"
            if dataset_name == 'cifar10':
                self.model = self.__load_cifar10_model(model_name, load_weights_path)
        elif defense_model == 'TRADES':
            assert datamodel in SUPPORTED_TRADES_DATAMODELS, f"Data-model {datamodel} is not supported for TRADES defensive model!"
            load_weights_path = f"./models/defensive_models/TRADES/{datamodel}.pth.tar"
            if dataset_name == 'cifar10':
                self.model = self.__load_cifar10_model(model_name, load_weights_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

    def __preprocess(self, x):
        if self.mean is not None and self.std is not None:
            # Normalize the input data if needed
            x = (x - self.mean) / self.std
        return x

    def query_hard_label(self, x):
        """
        Query the hard-label output of the model.

        Args:
            x (torch.Tensor): Input data. Each pixel value of x should be in the range of [0, 1].

        Returns:
            torch.Tensor: The predicted labels.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        with torch.no_grad():
            x = self.__preprocess(x)
            z = self.model(x)
            _, predict_labels = torch.max(z, dim=1)
        return predict_labels.cpu()