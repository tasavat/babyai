import torch
import numpy
from abc import ABC, abstractmethod
from collections import defaultdict

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.baselines import MeanBaseline
from babyai.rl.utils.supervised_losses import ExtraInfoCollector


class BaseECAlgo(ABC):
    """The base class for RL algorithms."""
    def __init__(self, envs, speaker, listener,
                 num_frames_per_proc, discount, lr, gae_lambda,
                 entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                 s_preprocess_obss, l_preprocess_obss, reshape_reward, aux_info):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        speaker : torch.Module
            the speaker model
        listener : torch.Module
            the listener model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        s_preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the speaker model can handle
        l_preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the listener model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        aux_info : list
            a list of strings corresponding to the name of the extra information
            retrieved from the environment for supervised auxiliary losses
        """
        # Store parameters

        self.env = ParallelEnv(envs)
        self.speaker = speaker
        self.listener = listener
        self.speaker.train()
        self.listener.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.s_preprocess_obss = s_preprocess_obss
        self.l_preprocess_obss = l_preprocess_obss
        self.reshape_reward = reshape_reward
        self.aux_info = aux_info

        # Store helpers values
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values
        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])

        self.message = None
        self.messages = []
        self.message_logit = None
        self.message_logits = []
        self.message_entropy = None
        self.message_entropies = []
        self.s_rewards = []  # reward at the end of the episode
        self.replace_message = False
        self.instructions = [None]*(shape[0])

        self.s_memory = torch.zeros(shape[1], self.speaker.memory_size, device=self.device)
        self.s_memories = torch.zeros(*shape, self.speaker.memory_size, device=self.device)
        self.l_memory = torch.zeros(shape[1], self.listener.memory_size, device=self.device)
        self.l_memories = torch.zeros(*shape, self.listener.memory_size, device=self.device)

        self.s_mask = torch.ones(shape[1], device=self.device)
        self.s_masks = torch.zeros(*shape, device=self.device)
        self.l_mask = torch.ones(shape[1], device=self.device)
        self.l_masks = torch.zeros(*shape, device=self.device)

        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        if self.aux_info:
            self.aux_info_collector = ExtraInfoCollector(self.aux_info, shape, self.device)

        # Initialize log values
        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """
        # reset variables
        self.messages = []
        self.message_logits = []
        self.message_entropies = []
        self.s_rewards = []

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            # speaker's turn
            if self.message is None or self.replace_message:
                s_preprocessed_obs = self.s_preprocess_obss(self.obs, device=self.device)
                model_results = self.speaker(s_preprocessed_obs, self.s_memory * self.s_mask.unsqueeze(1))
                message = model_results['value']
                logits = model_results['logits']
                entropy = model_results['entropy']
                self.message = message
                self.message_logit = logits
                self.message_entropy = entropy
                self.replace_message = False

            # listener's turn
            l_preprocessed_obs = self.l_preprocess_obss(self.obs, self.message, device=self.device)
            with torch.no_grad():
                model_results = self.listener(l_preprocessed_obs, self.l_memory * self.l_mask.unsqueeze(1))
                dist = model_results['dist']
                value = model_results['value']
                l_memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']

            action = dist.sample()

            obs, reward, done, env_info = self.env.step(action.cpu().numpy())
            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)

            # Update experiences values
            self.obss[i] = self.obs
            self.obs = obs

            self.l_memories[i] = self.l_memory
            self.l_memory = l_memory
            self.l_masks[i] = self.l_mask
            self.l_mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)

            self.instructions[i] = self.message
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            if self.aux_info:
                self.aux_info_collector.fill_dictionaries(i, env_info, extra_predictions)

            # Update log values
            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for j, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[j].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[j].item())
                    self.log_num_frames.append(self.log_episode_num_frames[j].item())

                    self.replace_message = True
                    self.messages.append(self.message[j])
                    self.message_logits.append(self.message_logit[j])
                    self.message_entropies.append(self.message_entropy[j])
                    self.s_rewards.append(self.rewards[i][j])

            self.log_episode_return *= self.l_mask
            self.log_episode_reshaped_return *= self.l_mask
            self.log_episode_num_frames *= self.l_mask

        # Add advantage and return to experiences
        l_preprocessed_obs = self.l_preprocess_obss(self.obs, self.message, device=self.device)
        with torch.no_grad():
            next_value = self.listener(l_preprocessed_obs, self.l_memory * self.l_mask.unsqueeze(1))['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.l_masks[i+1] if i < self.num_frames_per_proc - 1 else self.l_mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk
        s_exps = DictList()
        s_exps.has_data = False
        if len(self.messages) > 0:
            s_exps.has_data = True
            s_exps.message = torch.stack(self.messages)
            s_exps.message_logit = torch.stack(self.message_logits)
            s_exps.message_entropy = torch.stack(self.message_entropies)
            s_exps.s_reward = torch.stack(self.s_rewards)

        l_exps = DictList()
        l_exps.obs = [
            self.obss[i][j]
            for j in range(self.num_procs)
            for i in range(self.num_frames_per_proc)
        ]
        l_exps.instruction = torch.stack(self.instructions).reshape(-1).unsqueeze(1)

        # In comments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality
        # T x P x D -> P x T x D -> (P * T) x D
        l_exps.memory = self.l_memories.transpose(0, 1).reshape(-1, *self.l_memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        l_exps.mask = self.l_masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        l_exps.action = self.actions.transpose(0, 1).reshape(-1)
        l_exps.value = self.values.transpose(0, 1).reshape(-1)
        l_exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        l_exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        l_exps.returnn = l_exps.value + l_exps.advantage
        l_exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        if self.aux_info:
            l_exps = self.aux_info_collector.end_collection(l_exps)

        # Preprocess experiences
        # s_exps.obs = self.s_preprocess_obss(s_exps.obs, device=self.device)
        l_exps.obs = self.l_preprocess_obss(l_exps.obs, l_exps.instruction, device=self.device)

        # Log some values
        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return s_exps, l_exps, log

    @abstractmethod
    def update_parameters(self):
        pass


class PPOReinforceAlgo(BaseECAlgo):
    def __init__(self, envs, speaker, listener, num_frames_per_proc=None,
                 discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256,
                 s_preprocess_obss=None, l_preprocess_obss=None, reshape_reward=None, aux_info=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, speaker, listener,
                         num_frames_per_proc, discount, lr, gae_lambda,
                         entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                         s_preprocess_obss, l_preprocess_obss, reshape_reward, aux_info)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.s_optimizer = torch.optim.Adam(self.speaker.parameters(), 1e-2, (beta1, beta2), eps=adam_eps)
        self.l_optimizer = torch.optim.Adam(self.listener.parameters(), lr, (beta1, beta2), eps=adam_eps)
        self.batch_num = 0

        self.baselines = defaultdict(MeanBaseline)
        self.length_cost = 0

    def update_parameters(self):
        # Collect experiences

        s_exps, l_exps, logs = self.collect_experiences()
        '''
        exps is a DictList with the following keys ['obs', 'memory', 'mask', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.obs is a DictList with the following keys ['image', 'instr']
        exps.obj.image is a (n_procs * n_frames_per_proc) x image_size 4D tensor
        exps.obs.instr is a (n_procs * n_frames_per_proc) x (max number of words in an instruction) 2D tensor
        exps.memory is a (n_procs * n_frames_per_proc) x (memory_size = 2*image_embedding_size) 2D tensor
        exps.mask is (n_procs * n_frames_per_proc) x 1 2D tensor
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        '''
        if s_exps.has_data:
            message = s_exps.message
            s_log_prob = s_exps.message_logit
            s_entropy = s_exps.message_entropy
            loss = -1 * s_exps.s_reward

            message_length = self._find_lengths(message)
            length_loss = message_length.float() * self.length_cost
            # print("message")
            # print(message)
            # print("message length")
            # print(message_length)

            # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
            s_effective_entropy = torch.zeros(s_entropy.size(0), device=self.device)
            # the log prob of the choices made by S before and including the eos symbol - again, we don't
            # care about the rest
            s_effective_log_prob = torch.zeros(s_log_prob.size(0), device=self.device)

            for i in range(message.size(1)):
                not_eosed = (i < message_length).float()
                s_effective_entropy += s_entropy[:, i] * not_eosed
                s_effective_log_prob += s_log_prob[:, i] * not_eosed
            s_effective_entropy = s_effective_entropy / message_length.float()

            policy_length_loss = ((length_loss - self.baselines['length'].predict(length_loss)) * s_effective_log_prob).mean()
            policy_loss = ((loss.detach() - self.baselines['loss'].predict(loss.detach())) * s_effective_log_prob).mean()
            weighted_entropy = s_effective_entropy.mean() * self.entropy_coef

            optimized_loss = policy_length_loss + policy_loss - weighted_entropy
            # if the receiver is deterministic/differentiable, we apply the actual loss
            optimized_loss += loss.mean()

            # optimize
            self.s_optimizer.zero_grad()
            optimized_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.speaker.parameters(), self.max_grad_norm)
            self.s_optimizer.step()

            # update baselines
            self.baselines['loss'].update(loss)
            self.baselines['length'].update(length_loss)

        # update listener
        for _ in range(self.epochs):
            # Initialize log values
            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []

            '''
            For each epoch, we create int(total_frames / batch_size + 1) batches, each of size batch_size (except
            maybe the last one. Each batch is divided into sub-batches of size recurrence (frames are contiguous in
            a sub-batch), but the position of each sub-batch in a batch and the position of each batch in the whole
            list of frames is random thanks to self._get_batches_starting_indexes().
            '''

            for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches
                # Initialize batch values
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory
                memory = l_exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = l_exps[inds + i]

                    # Compute loss
                    model_results = self.listener(sb.obs, memory * sb.mask)
                    dist = model_results['dist']
                    value = model_results['value']
                    memory = model_results['memory']
                    extra_predictions = model_results['extra_predictions']

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values
                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch
                    if i < self.recurrence - 1:
                        l_exps.memory[inds + i + 1] = memory.detach()

                # Update batch values
                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic
                self.l_optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.listener.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.listener.parameters(), self.max_grad_norm)
                self.l_optimizer.step()

                # Update log values
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())

        # Log some values
        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["loss"] = numpy.mean(log_losses)

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    @staticmethod
    def _find_lengths(messages: torch.Tensor) -> torch.Tensor:
        """
        :param messages: A tensor of term ids, encoded as Long values, of size (batch size, max sequence length).
        :returns A tensor with lengths of the sequences, including the end-of-sequence symbol <eos> (in EGG, it is 0).
        If no <eos> is found, the full length is returned (i.e. messages.size(1)).

        >>> messages = torch.tensor([[1, 1, 0, 0, 0, 1], [1, 1, 1, 10, 100500, 5]])
        >>> lengths = find_lengths(messages)
        >>> lengths
        tensor([3, 6])
        """
        max_k = messages.size(1)
        zero_mask = messages == 0
        # a bit involved logic, but it seems to be faster for large batches than slicing batch dimension and
        # querying torch.nonzero()
        # zero_mask contains ones on positions where 0 occur in the outputs, and 1 otherwise
        # zero_mask.cumsum(dim=1) would contain non-zeros on all positions after 0 occurred
        # zero_mask.cumsum(dim=1) > 0 would contain ones on all positions after 0 occurred
        # (zero_mask.cumsum(dim=1) > 0).sum(dim=1) equates to the number of steps that happened after 0 occured (including it)
        # max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1) is the number of steps before 0 took place

        lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
        lengths.add_(1).clamp_(max=max_k)

        return lengths
