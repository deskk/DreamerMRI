import os
import sys
import ruamel.yaml as yaml
import wandb

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "external", "dreamerv3"))

import jax
import elements
import embodied
from dreamerv3.agent import Agent
from dreamerv3.main import make_logger, make_replay, make_stream

from scripts.real_medical_env import RealMedicalEnv, DreamerV3Wrapper

def make_env(config, index=0):
    dataset_dir = "/local/scratch/scratch-hd/desmond/duke_micro_subset/preprocessed"
    # Deterministically locks the batch architecture over a solitary matrix
    env = RealMedicalEnv(data_dir=dataset_dir, debug_patient_id="Breast_MRI_001")
    env = DreamerV3Wrapper(env)
    
    import numpy as np
    env.obs_space = {
        'obs_3d': elements.Space(np.uint8, (64, 64, 64)),
        'image': elements.Space(np.uint8, (64, 64, 3)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool)
    }
    env.act_space = {
        'action': elements.Space(np.float32, (3,), -1.0, 1.0),
        'reset': elements.Space(bool)
    }
    
    env = embodied.wrappers.UnifyDtypes(env)
    return env

def main():
    print("--- Memory Overfit (N=1) Interaction Test ---")
    devices = jax.devices()
    print(f"[JAX DEBUG] Detected devices: {devices}")
    
    # Enable comprehensive intrinsic world model video synchronizations mapped into Weight and Biases
    wandb.init(entity="s26_aai_finalproject", project="Medical-Dreamer", name="N1-Overfit-Check-A6000", sync_tensorboard=True)

    folder = elements.Path(os.path.join(root_dir, 'external', 'dreamerv3', 'dreamerv3'))
    configs_yaml = folder / 'configs.yaml'
    configs_data = yaml.YAML(typ='safe').load(configs_yaml.read())
    config = elements.Config(configs_data['defaults'])
    
    config = config.update({
        'task': 'overfit_medical',
        'run.steps': 15000,            
        'run.log_every': 250,
        'run.train_ratio': 64,         
        'batch_size': 16,
        'batch_length': 16,
        # Native telemetry routing
        'logger.outputs': ['terminal', 'tensorboard', 'jsonl', 'wandb']
    })

    logdir = elements.Path(os.path.expanduser('~/logdir/overfit_medical'))
    config = config.update({'logdir': str(logdir)})

    print("--- Firing N=1 Online Agent Memorization Loop ---")
    try:
        from functools import partial as bind
        
        args = elements.Config(
            **config.run,
            logdir=config.logdir,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            report_length=config.report_length,
            consec_train=config.consec_train,
            consec_report=config.consec_report,
            replay_context=config.replay_context,
            replica=0,
            replicas=1,
        )

        def make_agent(config):
            env = make_env(config, 0)
            obs_space = env.obs_space
            # Explicit embodied API translation dropping internal states
            act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
            env.close()
            agent = Agent(obs_space, act_space, elements.Config(
                **config.agent,
                logdir=config.logdir,
                seed=config.seed,
                jax=config.jax,
                batch_size=config.batch_size,
                batch_length=config.batch_length,
                replay_context=config.replay_context,
                report_length=config.report_length,
                replica=0,
                replicas=1,
            ))
            # HACK: Preserve submodule integrity by strictly duck-typing instance parameters 
            # to hide the C=64 matrix from the intrinsic W&B video generation loop.
            agent.dec.imgkeys = [k for k in agent.dec.imgkeys if k != 'obs_3d']
            return agent

        embodied.run.train(
            bind(make_agent, config),
            bind(make_replay, config, 'replay'),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            args)
            
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
