from config import SUPPORTED_DATAMODELS, CLASS_NUM, INPUT_CHANNELS, IMAGE_SIZE
from mmpretrain import get_model
import torch

class VictimModel(object):
    def __load_cifar10_model(self, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "resnet50":
            model = get_model("resnet50_8xb16_cifar10", pretrained=True)
            # Normalization for cifar10-resnet50
            self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
            self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
        return model

    def __load_cifar100_model(self, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "resnet50":
            model = get_model("resnet50_8xb16_cifar100", pretrained=True)
            # Normalization for cifar100-resnet50
            self.mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1).to(device)
            self.std = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1).to(device)
        return model
    
    def __load_imagenet_model(self, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "densenet121":
            model = get_model("densenet121_3rdparty_in1k", pretrained=True)
            # Normalization for imagenet
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        elif model_name == "resnet50":
            model = get_model("resnet50_8xb32_in1k", pretrained=True)
            # Normalization for imagenet
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        return model

    def __load_inner_model(self, model_name, dataset_name):
        """
        Load the inner model for the specified dataset and model name.

        Args:
            model_name (str): Name of the model to load.
            dataset_name (str): Name of the dataset to load.

        Returns:
            The loaded model. This function will set the normalization parameters simultaneously.
        """
        datamodel = dataset_name + "-" + model_name
        assert datamodel in SUPPORTED_DATAMODELS, f"Unsupported datamodel: {datamodel}!"
        if dataset_name == "cifar10":
            return self.__load_cifar10_model(model_name)
        elif dataset_name == "cifar100":
            return self.__load_cifar100_model(model_name)
        elif dataset_name == "imagenet":
            return self.__load_imagenet_model(model_name)

    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model = self.__load_inner_model(model_name, dataset_name)
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