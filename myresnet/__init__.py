from IPython.utils.sysinfo import num_cpus
from PIL.features import modules

from .resnet import resnet18,resnet50,resnet34,resnet101

model_dist = {
    'resnet18' : resnet18,
    'resnet50' : resnet50,
    'resnet34' : resnet34,
    'resnet101': resnet101
}

def create_model(model_name,num_classes):
    return model_dist[model_name](num_classes = num_classes)