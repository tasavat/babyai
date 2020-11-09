import os
import torch

from .. import utils


def get_model_dir(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name)


def get_model_path(model_name):
    return os.path.join(get_model_dir(model_name), "model.pt")


def get_speaker_path(model_name):
    return os.path.join(get_model_dir(model_name), "speaker.pt")


def get_listener_path(model_name):
    return os.path.join(get_model_dir(model_name), "listener.pt")


def load_model(model_name, raise_not_found=True):
    path = get_model_path(model_name)
    try:
        model = torch.load(path)
        model.eval()
        return model
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No model found at {}".format(path))


def load_ec_model(model_name, raise_not_found=True):
    speaker_path = get_speaker_path(model_name)
    listener_path = get_listener_path(model_name)
    try:
        speaker = torch.load(speaker_path)
        listener = torch.load(listener_path)
        speaker.eval()
        listener.eval()
        return speaker, listener
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No model found at {}".format(path))
        return None, None


def save_model(model, model_name):
    path = get_model_path(model_name)
    utils.create_folders_if_necessary(path)
    torch.save(model, path)


def save_ec_model(speaker, listener, model_name):
    speaker_path = get_speaker_path(model_name)
    listener_path = get_listener_path(model_name)
    utils.create_folders_if_necessary(speaker_path)
    utils.create_folders_if_necessary(listener_path)
    torch.save(speaker, speaker_path)
    torch.save(listener, listener_path)
