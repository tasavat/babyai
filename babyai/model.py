import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import babyai.rl
from babyai.rl.utils.supervised_losses import required_heads
from babyai.utils.model import load_model


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class ExpertControllerFiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)
        out = self.bn2(out)
        out = F.relu(out)
        return out


class ACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False, arch="cnn1",
                 aux_info=None):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim

        self.obs_space = obs_space

        if arch == "cnn1":
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=image_dim, kernel_size=(2, 2)),
                nn.ReLU()
            )
        elif arch.startswith("expert_filmcnn"):
            if not self.use_instr:
                raise ValueError("FiLM architecture can be used when instructions are enabled")

            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )
            self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        else:
            raise ValueError("Incorrect architecture name: {}".format(arch))

        # Define instruction embedding
        if self.use_instr:
            if self.lang_model in ['gru', 'bigru', 'attgru']:
                self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
                if self.lang_model in ['gru', 'bigru', 'attgru']:
                    gru_dim = self.instr_dim
                    if self.lang_model in ['bigru', 'attgru']:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim, gru_dim, batch_first=True,
                        bidirectional=(self.lang_model in ['bigru', 'attgru']))
                    self.final_instr_dim = self.instr_dim
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList([
                        nn.Conv2d(1, kernel_dim, (K, self.instr_dim)) for K in kernel_sizes])
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model == 'attgru':
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_instr and not "filmcnn" in arch:
            self.embedding_size += self.final_instr_dim

        if arch.startswith("expert_filmcnn"):
            if arch == "expert_filmcnn":
                num_module = 2
            else:
                num_module = int(arch[(arch.rfind('_') + 1):])
            self.controllers = []
            for ni in range(num_module):
                if ni < num_module-1:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim,
                        out_features=128, in_channels=128, imm_channels=128)
                else:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim, out_features=self.image_dim,
                        in_channels=128, imm_channels=128)
                self.controllers.append(mod)
                self.add_module('FiLM_Controler_' + str(ni), mod)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, instr_embedding=None):
        if self.use_instr and instr_embedding is None:
            instr_embedding = self._get_instr_embedding(obs.instr)
        if self.use_instr and self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M
            mask = (obs.instr != 0).float()
            # The mask tensor has the same length as obs.instr, and
            # thus can be both shorter and longer than instr_embedding.
            # It can be longer if instr_embedding is computed
            # for a subbatch of obs.instr.
            # It can be shorter if obs.instr is a subbatch of
            # the batch that instr_embeddings was computed for.
            # Here, we make sure that mask and instr_embeddings
            # have equal length along dimension 1.
            mask = mask[:, :instr_embedding.shape[1]]
            instr_embedding = instr_embedding[:, :mask.shape[1]]

            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

        if self.arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            for controler in self.controllers:
                x = controler(x, instr_embedding)
            x = F.relu(self.film_pool(x))
        else:
            x = self.image_conv(x)

        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_instr and not "filmcnn" in self.arch:
            embedding = torch.cat((embedding, instr_embedding), dim=1)

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions}

    def _get_instr_embedding(self, instr):
        lengths = (instr != 0).sum(1).long()
        if self.lang_model == 'gru':
            out, _ = self.instr_rnn(self.word_embedding(instr))
            hidden = out[range(len(lengths)), lengths-1, :]
            return hidden

        elif self.lang_model in ['bigru', 'attgru']:
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, final_states = self.instr_rnn(self.word_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            return outputs if self.lang_model == 'attgru' else final_states

        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))


class ACModelImgInstr(nn.Module, babyai.rl.RecurrentACModel):
    """ACModel with images as references for the given instruction instead of a natural language sequence"""
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=True, use_memory=False, arch="cnn1",
                 aux_info=None):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.aux_info = aux_info
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim

        self.obs_space = obs_space

        # Define image embedding
        if arch == "cnn1":
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=image_dim, kernel_size=(2, 2)),
                nn.ReLU()
            )
        elif arch.startswith("expert_filmcnn"):
            if not self.use_instr:
                raise ValueError("FiLM architecture can be used when instructions are enabled")

            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )
            self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        else:
            raise ValueError("Incorrect architecture name: {}".format(arch))

        # Define instruction embedding
        if self.use_instr:
            self.instr_cnn = nn.Sequential(
                nn.Conv2d(in_channels=3*2, out_channels=128, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=image_dim, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(image_dim, instr_dim),
            )
            self.final_instr_dim = self.instr_dim

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_instr and not "filmcnn" in arch:
            self.embedding_size += self.final_instr_dim

        if arch.startswith("expert_filmcnn"):
            if arch == "expert_filmcnn":
                num_module = 2
            else:
                num_module = int(arch[(arch.rfind('_') + 1):])
            self.controllers = []
            for ni in range(num_module):
                if ni < num_module-1:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim,
                        out_features=128, in_channels=128, imm_channels=128)
                else:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim, out_features=self.image_dim,
                        in_channels=128, imm_channels=128)
                self.controllers.append(mod)
                self.add_module('FiLM_Controler_' + str(ni), mod)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, instr_embedding=None):
        if self.use_instr and instr_embedding is None:
            instr = torch.transpose(torch.transpose(obs.instr, 1, 3), 2, 3)
            instr_embedding = self._get_instr_embedding(instr, inject_dummy=False)
            # [dummy]
            # instr_embedding = self._get_instr_embedding(obs.instr, inject_dummy=True)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

        if self.arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            for controller in self.controllers:
                x = controller(x, instr_embedding)
            x = F.relu(self.film_pool(x))
        else:
            x = self.image_conv(x)

        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_instr and not "filmcnn" in self.arch:
            embedding = torch.cat((embedding, instr_embedding), dim=1)

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions}

    def _get_instr_embedding(self, instr, inject_dummy=False):
        instr_embedding = self.instr_cnn(instr)
        if inject_dummy:
            instr_embedding = torch.zeros_like(instr_embedding)
        return instr_embedding


class SpeakerModel(ACModelImgInstr):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=True, use_memory=False, arch="cnn1", aux_info=None,
                 vocab_size=None, max_len=None, instr_gru_num_layers=2, force_eos=True,
                 pretrained_model_path=None
                 ):
        super().__init__(obs_space, action_space,
                         image_dim=image_dim, memory_dim=memory_dim, instr_dim=instr_dim,
                         use_instr=use_instr, use_memory=False, arch=arch, aux_info=aux_info)
        # remove actor-critic heads
        self.actor = None
        self.critic = None

        # update state
        if pretrained_model_path:
            self.load_pretrained_state(pretrained_model_path)
            print(f"update pretrained state: {pretrained_model_path}")

        # inject instruction generator module
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.force_eos = force_eos
        self.instr_gru_num_layers = instr_gru_num_layers
        self.instr_gru_hidden_size = instr_dim
        self.instr_gru = nn.ModuleList([
            nn.GRUCell(input_size=self.embedding_size, hidden_size=self.instr_gru_hidden_size)
            if i == 0 else
            nn.GRUCell(input_size=self.instr_gru_hidden_size, hidden_size=self.instr_gru_hidden_size)
            for i in range(self.instr_gru_num_layers)
        ])
        self.hidden_to_output = nn.Linear(self.instr_gru_hidden_size, vocab_size)
        self.vocab_embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.sos_embedding = nn.Parameter(torch.zeros(self.embedding_size), requires_grad=False)
        self.reset_sos_embeddings()

    def reset_sos_embeddings(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def load_pretrained_state(self, pretrained_model_path):
        pretrained_model = load_model(pretrained_model_path, raise_not_found=True)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.state_dict()
        # update only relevant state
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, obs, memory, instr_embedding=None):
        if self.use_instr and instr_embedding is None:
            instr = torch.transpose(torch.transpose(obs.instr, 1, 3), 2, 3)
            instr_embedding = self._get_instr_embedding(obs.instr, inject_dummy=False)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

        if self.arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            for controller in self.controllers:
                x = controller(x, instr_embedding)
            x = F.relu(self.film_pool(x))
        else:
            x = self.image_conv(x)

        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_instr and not "filmcnn" in self.arch:
            embedding = torch.cat((embedding, instr_embedding), dim=1)

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        # instruction gru
        prev_hidden = [embedding]
        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.instr_gru_num_layers - 1)])
        input = torch.stack([self.sos_embedding] * embedding.size(0))

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.instr_gru):
                h_t = layer(input, prev_hidden[i])
                prev_hidden[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))

            input = self.vocab_embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        # mask after EOS
        max_k = sequence.size(1)
        zero_mask = sequence == 0
        lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
        lengths.add_(1).clamp_(max=max_k)
        not_eosed = torch.stack([(i < lengths).float() for i in range(max_k)], dim=1).type('torch.cuda.LongTensor')

        sequence = sequence * not_eosed

        return {'logits': logits, 'value': sequence, 'entropy': entropy,
                'memory': memory, 'extra_predictions': extra_predictions}


class ListenerModel(ACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False,
                 arch="cnn1", aux_info=None):
        super().__init__(obs_space, action_space,
                         image_dim=image_dim, memory_dim=memory_dim, instr_dim=instr_dim,
                         use_instr=use_instr, lang_model=lang_model, use_memory=use_memory,
                         arch=arch, aux_info=aux_info)
