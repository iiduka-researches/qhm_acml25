from .resnet import resnet18
from .wideresnet import WideResNet28_10
from .vgg import vgg16_bn
from .ViT import ViT

def get_model(model_name, device):
    if model_name == "ResNet18":
        net = resnet18()
    elif model_name == "WideResNet-28-10":
        net = WideResNet28_10()
    elif model_name == "vgg-16":
        net = vgg16_bn()
    elif model_name == "ViT-T":
        net = ViT(img_size=32, patch_size = 4, num_classes=100, dim=192,
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=0.1, is_SPT="store_true", is_LSA="store_true")
    else:
        raise ValueError(f"Invalid model name '{model_name}'. Available options are: 'ResNet18', 'WideResNet-28-10'.")
    net = net.to(device)
    return net