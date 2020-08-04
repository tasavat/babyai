import os
import json
import numpy
import re
import torch
import babyai.rl

import gym
import babyai

from collections import deque
from json import JSONEncoder
from .. import utils


def get_vocab_path(model_name):
    return os.path.join(utils.get_model_dir(model_name), "vocab.json")


def get_simulated_env(model_name):
    env_name = model_name.split("_")[0]
    env = gym.make(env_name)
    return env


def get_pretrained_agent(model_name):
    env_name = model_name.split("_")[0]
    pretrained_model_name = env_name + "_pretrained"
    model = utils.load_model(pretrained_model_name, raise_not_found=True)
    if not model:
        raise FileNotFoundError(f"Model not found: {pretrained_model_name}")
    return model


def get_preprocessed_obs(model_name, obs_space):
    env_name = model_name.split("_")[0]
    pretrained_model_name = env_name + "_pretrained"
    obss_preprocessor = utils.ObssPreprocessor(pretrained_model_name, obs_space)
    return obss_preprocessor


class Vocabulary:
    def __init__(self, model_name):
        self.path = get_vocab_path(model_name)
        self.max_size = 100
        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))
        else:
            self.vocab = {}

    def __getitem__(self, token):
        if not (token in self.vocab.keys()):
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def save(self, path=None):
        if path is None:
            path = self.path
        utils.create_folders_if_necessary(path)
        json.dump(self.vocab, open(path, "w"))

    def copy_vocab_from(self, other):
        '''
        Copy the vocabulary of another Vocabulary object to the current object.
        '''
        self.vocab.update(other.vocab)


class ImageInstructionDict:
    def __init__(self):
        self.max_size = 256
        self.id_cache = deque(maxlen=self.max_size)
        self.img_instr = {}  # reset id cache every time training restarts

    def __getitem__(self, id):
        if id in self.img_instr.keys():
            return self.img_instr[id]
        return None

    def __setitem__(self, id, img_list):
        self.id_cache.append(id)
        self.img_instr[id] = img_list
        # remove old ids
        while len(self.id_cache) > self.max_size:
            remove_id = self.id_cache.popleft()
            if remove_id in self.img_instr.keys():
                self.img_instr.remove(remove_id)


class InstructionsPreprocessor(object):
    def __init__(self, model_name, load_vocab_from=None):
        self.model_name = model_name
        self.vocab = Vocabulary(model_name)

        path = get_vocab_path(model_name)
        if not os.path.exists(path) and load_vocab_from is not None:
            # self.vocab.vocab should be an empty dict
            secondary_path = get_vocab_path(load_vocab_from)
            if os.path.exists(secondary_path):
                old_vocab = Vocabulary(load_vocab_from)
                self.vocab.copy_vocab_from(old_vocab)
            else:
                raise FileNotFoundError('No pre-trained model under the specified name')

    def __call__(self, obss, device=None):
        raw_instrs = []
        max_instr_len = 0

        for obs in obss:
            tokens = re.findall("([a-z]+)", obs["mission"].lower())
            instr = numpy.array([self.vocab[token] for token in tokens])
            raw_instrs.append(instr)
            max_instr_len = max(len(instr), max_instr_len)

        instrs = numpy.zeros((len(obss), max_instr_len))

        for i, instr in enumerate(raw_instrs):
            instrs[i, :len(instr)] = instr

        instrs = torch.tensor(instrs, device=device, dtype=torch.long)
        return instrs


class ImgInstrPreprocessor(object):
    def __init__(self, model_name, obs_space=None):
        self.img_instr_dict = ImageInstructionDict()
        self.simulated_env = get_simulated_env(model_name)
        self.simulated_obs = None
        self.obss_preprocessor = get_preprocessed_obs(model_name, obs_space)
        self.pretrained_agent = get_pretrained_agent(model_name)
        self.pretrained_agent.eval()

    def __call__(self, obss, device=None):
        img_instrs = []
        for obs in obss:
            # retrieve image instruction from cache by env id
            if obs['id'] in self.img_instr_dict.img_instr.keys():
                img_instr = self.img_instr_dict[obs['id']]
            # generate new image instruction with pretrained agent
            else:
                img_instr = self._generate_img_instr(obs, device=device)
                self.img_instr_dict[obs['id']] = img_instr
            img_instrs.append(img_instr)

        img_instrs = torch.stack(img_instrs, dim=0)
        return img_instrs

    def _full_obs(self):
        full_obs = self.simulated_env.grid.encode()
        full_obs[self.simulated_env.agent_pos[0]][self.simulated_env.agent_pos[0]] = numpy.array([
            10,
            0,
            self.simulated_env.agent_dir
        ])
        return full_obs

    def _generate_img_instr(self, obs, device=None):
        # retrieve mission
        mission = obs["mission"]

        # regenerate env + play until success (reward > 0)
        success = False
        patience_cnt = 0
        while (not success) or patience_cnt < 3:
            # generate new env until both missions match
            while True:
                self.simulated_obs = self.simulated_env.reset()
                if self.simulated_obs['mission'] == mission:
                    img_instr = [self._full_obs()]  # first observation
                    break

            # simulate until done
            memory = torch.zeros(1, self.pretrained_agent.memory_size, device=device)
            mask = torch.ones(1, device=device)
            done = False
            while not done:
                preprocessed_obs = self.obss_preprocessor([self.simulated_obs], device=device)
                with torch.no_grad():
                    model_results = self.pretrained_agent(preprocessed_obs, memory * mask.unsqueeze(1))
                    dist = model_results['dist']
                    memory_ = model_results['memory']
                action = dist.sample()
                obs, reward, done, env_info = self.simulated_env.step(action.cpu().numpy())

                self.simulated_obs = obs
                memory = memory_
                mask = 1 - torch.tensor(done, device=device, dtype=torch.float)
                mask = torch.reshape(mask, (1,))

                if done and reward > 0:
                    success = True
                    img_instr.append(self._full_obs())  # finished observation
                patience_cnt += 1

        img_instr = torch.tensor(img_instr, device=device, dtype=torch.float)
        img_instr = torch.reshape(img_instr, (-1, img_instr.size()[1], img_instr.size()[2]))
        return img_instr


class RawImagePreprocessor(object):
    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        images = torch.tensor(images, device=device, dtype=torch.float)
        return images


class IntImagePreprocessor(object):
    def __init__(self, num_channels, max_high=255):
        self.num_channels = num_channels
        self.max_high = max_high
        self.offsets = numpy.arange(num_channels) * max_high
        self.max_size = int(num_channels * max_high)

    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        # The padding index is 0 for all the channels
        images = (images + self.offsets) * (images > 0)
        images = torch.tensor(images, device=device, dtype=torch.long)
        return images


class ObssPreprocessor:
    def __init__(self, model_name, obs_space=None, load_vocab_from=None):
        self.image_preproc = RawImagePreprocessor()
        self.instr_preproc = InstructionsPreprocessor(model_name, load_vocab_from)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": 147,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_


class IntObssPreprocessor(object):
    def __init__(self, model_name, obs_space, load_vocab_from=None):
        image_obs_space = obs_space.spaces["image"]
        self.image_preproc = IntImagePreprocessor(image_obs_space.shape[-1],
                                                  max_high=image_obs_space.high.max())
        self.instr_preproc = InstructionsPreprocessor(load_vocab_from or model_name)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": self.image_preproc.max_size,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_


class ImgInstrObssPreprocessor(object):
    def __init__(self, model_name, obs_space=None, load_vocab_from=None):
        self.image_preproc = RawImagePreprocessor()
        self.instr_preproc = ImgInstrPreprocessor(model_name, obs_space)
        self.obs_space = {
            "image": 147,
            "instr": 147,
        }

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_
