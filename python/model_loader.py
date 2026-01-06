import os
import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes):
    """
    define and initialize ResNet18 model
    refine Fully Connected Layer to num_classes
    """
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def load_trained_model(model_path):
    """
    load weights and class info
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        device = "cpu"
        # cpu
        checkpoint = torch.load(model_path, map_location=device)

        if "class_names" in checkpoint:
            class_names = checkpoint["class_names"]
        else:
            # Except
            print("Wraning: using default logic.")
            # infer class_num with size of weight tensorflow
            last_layer_weight = checkpoint["model_state_dict"]["fc.weight"]
            num_classes_inferred = last_layer_weight.shape[0]
            class_names = [str(i) for i in range(num_classes_inferred)]

        num_classes = len(class_names)

        model = get_model(num_classes)

        model.load_state_dict(checkpoint["model_state_dict"])

        # fix Dropout, BatchNorm
        model.to(device)
        model.eval()

        return model, class_names

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}. Error: {e}")
