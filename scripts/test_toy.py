import os
import sys
import ruamel.yaml as yaml

# Ensure local custom modules and dreamerv3 are resolvable
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "external", "dreamerv3"))

import jax
import elements
import embodied
from dreamerv3.agent import Agent
from dreamerv3.main import make_logger, make_replay, make_stream

from scripts.toy_env import ToyMedicalEnv, DreamerV3Wrapper

def make_env(config, index=0):
    env = ToyMedicalEnv()
    env = DreamerV3Wrapper(env)
    
    # Explicitly bind the embodied.Space dictionary configurations for the framework
    import numpy as np
    env.obs_space = {
        'image': embodied.Space(np.uint8, (64, 64, 64)),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool)
    }
    env.act_space = {
        'action': embodied.Space(np.float32, (3,), -1.0, 1.0),
        'reset': embodied.Space(bool)
    }
    
    env = embodied.wrappers.UnifyDtypes(env)
    return env

def main():
    print("--- 3D Toy Sphere Sanity Check ---")
    devices = jax.devices()
    print(f"[JAX DEBUG] Detected devices: {devices}")
    
    if len(devices) == 0 or "cpu" in str(devices[0]).lower():
        print("[WARNING] JAX is not utilizing a GPU/TPU! Check CUDA wheels on A100.")
    else:
        print("[SUCCESS] JAX seamlessly bound to the Accelerator Tensor Cores!")

    # Native dreamerv3 configuration parse routine
    folder = elements.Path(os.path.join(root_dir, 'external', 'dreamerv3', 'dreamerv3'))
    configs_yaml = folder / 'configs.yaml'
    configs_data = yaml.YAML(typ='safe').load(configs_yaml.read())
    config = elements.Config(configs_data['defaults'])
    
    config = config.update({
        'task': 'toy_medical',
        'run.steps': 10000,            # Very aggressive boundary just to verify OOM bounds
        'run.log_every': 500,
        'run.train_ratio': 64,         
        'batch_size': 16,
        'batch_length': 16,            # Reduced RSSM unroll sequence length for memory safety             
        'encoder.mlp_keys': '$^',
        'decoder.mlp_keys': '$^',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
    })

    logdir = elements.Path('~/logdir/toy_medical').expand()
    config = config.update({'logdir': str(logdir)})

    print("--- Firing Online Agent Interaction Loop ---")
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
            act_space = env.act_space
            env.close()
            return Agent(obs_space, act_space, elements.Config(
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

        embodied.run.train(
            bind(make_agent, config),
            bind(make_replay, config, 'replay'),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            args)
            
    except Exception as e:
        print(f"Exception raised tracking inner loops: {e}")

if __name__ == '__main__':
    main()
