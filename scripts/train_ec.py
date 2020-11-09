"""
Script to train the agent through EC-reinforcement learning.
"""

import os
import logging
import csv
import json
import gym
import time
import datetime
import torch
import numpy as np
import subprocess

import babyai
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.model import SpeakerModel, ListenerModel
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import ModelAgent


# Parse arguments
parser = ArgumentParser()
parser.add_argument("--algo", default='ppo_reinforce',
                    help="algorithm to use (default: ppo_reinforce)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--reward-scale", type=float, default=20.,
                    help="Reward scale multiplier")
parser.add_argument("--gae-lambda", type=float, default=0.99,
                    help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--ppo-epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")
parser.add_argument("--vocab_size", type=int, default=10,
                    help="size of vocabulary (default: 10)")
parser.add_argument("--max_len", type=int, default=5,
                    help="message's maximum length(default: 5)")
parser.add_argument("--suffix", type=str, default=None,
                    help="suffix to model's name (default: None)")
parser.add_argument("--speaker_pretrained_model", type=str, default=None,
                    help="path to speaker pretrained model (default: None)")
parser.add_argument("--speaker_pretrained_EC_model", type=str, default=None,
                    help="path to speaker pretrained EC model (default: None)")
parser.add_argument("--listener_pretrained_model", type=str, default=None,
                    help="path to speaker pretrained model (default: None)")
parser.add_argument("--listener_pretrained_EC_model", type=str, default=None,
                    help="path to speaker pretrained model (default: None)")
parser.add_argument("--episodes", type=int, default=100000,
                    help="number of training episodes")
args = parser.parse_args()

utils.seed(args.seed)


def main():
    # Generate environments
    envs = []
    for i in range(args.procs):
        env = gym.make(args.env)
        # env = utils.FullyObsWrapper(env)
        env.seed(100 * args.seed + i)
        envs.append(env)

    # Define model name
    suffix = args.suffix or datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    instr = args.instr_arch if args.instr_arch else "noinstr"
    mem = "mem" if not args.no_mem else "nomem"
    model_name_parts = {
        'env': args.env,
        'algo': args.algo,
        'arch': args.arch,
        'instr': instr,
        'mem': mem,
        'seed': args.seed,
        'info': '',
        'coef': '',
        'suffix': suffix,
        'vocab_size': args.vocab_size,
        'max_len': args.max_len
    }
    default_model_name = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_V{vocab_size}_L{max_len}_{suffix}".format(**model_name_parts)
    if args.pretrained_model:
        default_model_name = default_model_name + '_PretrainedWith_' + args.pretrained_model
    args.model = args.model.format(**model_name_parts) if args.model else default_model_name

    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    # Define obss preprocessor
    speaker_obss_preprocessor = utils.SpeakerObssPreprocessor(args.model)
    listener_obss_preprocessor = utils.ListenerObssPreprocessor(args.model)

    # Define actor-critic model
    speaker, listener = utils.load_ec_model(args.model, raise_not_found=False)
    
    if speaker is None:
        if args.pretrained_model:
            speaker, listener = utils.load_ec_model(args.pretrained_model, raise_not_found=True)
        else:
            speaker = SpeakerModel(speaker_obss_preprocessor.obs_space, envs[0].action_space,
                                   args.image_dim, args.memory_dim, args.instr_dim,
                                   not args.no_instr, not args.no_mem, args.arch,
                                   vocab_size=args.vocab_size, max_len=args.max_len,
                                   pretrained_model_path=args.speaker_pretrained_model,
                                   pretrained_model_EC_path=args.speaker_pretrained_EC_model)
            listener = ListenerModel(listener_obss_preprocessor.obs_space, envs[0].action_space,
                                     args.image_dim, args.memory_dim, args.instr_dim,
                                     not args.no_instr, args.instr_arch, not args.no_mem, args.arch,
                                     pretrained_model_path=args.listener_pretrained_model,
                                     pretrained_model_EC_path=args.listener_pretrained_EC_model)
    
    utils.save_ec_model(speaker, listener, args.model)

    if torch.cuda.is_available():
        speaker.cuda()
        listener.cuda()

    # Define actor-critic algo
    reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
    algo = babyai.rl.PPOReinforceAlgo(envs, speaker, listener, args.frames_per_proc,
                                      args.discount, args.lr, args.beta1, args.beta2, args.gae_lambda,
                                      args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                      args.optim_eps, args.clip_eps, args.ppo_epochs, args.batch_size,
                                      speaker_obss_preprocessor, listener_obss_preprocessor, reshape_reward)

    # When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
    # Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure that
    # the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

    utils.seed(args.seed)

    # Restore training status
    status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
    if os.path.exists(status_path):
        with open(status_path, 'r') as src:
            status = json.load(src)
    else:
        status = {'i': 0,
                  'num_episodes': 0,
                  'num_frames': 0}

    # Define logger and Tensorboard writer and CSV writer
    header = (["update", "episodes", "frames", "FPS", "duration"]
              + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["success_rate"]
              + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"]
              + ["s_policy_loss", "s_policy_length_loss", "s_entropy", "s_loss", "s_grad_norm"])
    if args.tb:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(utils.get_log_dir(args.model))
    csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
    first_created = not os.path.exists(csv_path)
    # we don't buffer data going in the csv log, cause we assume
    # that one update will take much longer that one write to the log
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(header)

    # Log code state, command, availability of CUDA and model
    babyai_code = list(babyai.__path__)[0]
    try:
        last_commit = subprocess.check_output(
            'cd {}; git log -n1'.format(babyai_code), shell=True).decode('utf-8')
        logger.info('LAST COMMIT INFO:')
        logger.info(last_commit)
    except subprocess.CalledProcessError:
        logger.info('Could not figure out the last commit')
    try:
        diff = subprocess.check_output(
            'cd {}; git diff'.format(babyai_code), shell=True).decode('utf-8')
        if diff:
            logger.info('GIT DIFF:')
            logger.info(diff)
    except subprocess.CalledProcessError:
        logger.info('Could not figure out the last commit')
    logger.info('COMMAND LINE ARGS:')
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))
    logger.info(speaker)
    logger.info(listener)

    # Train model
    total_start_time = time.time()
    best_success_rate = 0
    best_mean_return = 0
    test_env_name = args.env

    consecutive_success = 0
    success_rate_criteria = 0.90

    # [adjust] training stop condition
    while (status['num_episodes'] < args.episodes) and (consecutive_success < 3):
        # Update parameters
        update_start_time = time.time()
        logs = algo.update_parameters()
        update_end_time = time.time()

        status['num_frames'] += logs["num_frames"]
        status['num_episodes'] += logs['episodes_done']
        status['i'] += 1

        # Print logs
        if status['i'] % args.log_interval == 0:
            total_ellapsed_time = int(time.time() - total_start_time)
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = datetime.timedelta(seconds=total_ellapsed_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            success_per_episode = utils.synthesize(
                [1 if r > 0 else 0 for r in logs["return_per_episode"]])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            data = [status['i'], status['num_episodes'], status['num_frames'],
                    fps, total_ellapsed_time,
                    *return_per_episode.values(),
                    success_per_episode['mean'],
                    *num_frames_per_episode.values(),
                    logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                    logs["loss"], logs["grad_norm"],
                    logs["s_policy_loss"], logs["s_policy_length_loss"], logs["s_entropy"], logs["s_optimized_loss"], logs["s_grad_norm"]
                    ]

            format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                          "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                          "pL {:.3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | "
                          "s_pL {:.3f} | s_pLL {:.3f} | s_H {:.3f} | s_L {:.3f} | s_gN {:.3f} | ")

            logger.info(format_str.format(*data))
            if args.tb:
                assert len(header) == len(data)
                for key, value in zip(header, data):
                    writer.add_scalar(key, float(value), status['num_frames'])

            csv_writer.writerow(data)

        # Save obss preprocessor vocabulary and model
        if args.save_interval > 0 and status['i'] % args.save_interval == 0:
            with open(status_path, 'w') as dst:
                json.dump(status, dst)
                utils.save_ec_model(speaker, listener, args.model)

        # update consecutive success
        success_per_episode = utils.synthesize([1 if r > 0 else 0 for r in logs["return_per_episode"]])
        if success_per_episode['mean'] >= success_rate_criteria:
            consecutive_success += 1
        else:
            consecutive_success = 0


if __name__ == "__main__":
    main()
