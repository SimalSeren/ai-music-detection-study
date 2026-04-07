from artifact_net import ArtifactNet
from resnet_spectrogram import ResNetSpectrogram
from simple_cnn import SimpleCNN


def create_model(model_name: str):
    model_name = model_name.lower()

    if model_name == "simplecnn":
        return SimpleCNN()
    if model_name == "resnet":
        return ResNetSpectrogram()
    if model_name == "artifactnet":
        return ArtifactNet()

    raise ValueError(f"Desteklenmeyen model: {model_name}")
