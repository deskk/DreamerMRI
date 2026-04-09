import os
import sys

# Ensure custom modules and official dreamerv3 are securely located
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "external", "dreamerv3"))

import dreamerv3
from env import MedicalEnv, DreamerV3Wrapper

def make_medical_env(config, logger, step, *args, **kwargs):
    """
    Factory function required by dreamerv3 to instantiate environments natively.
    """
    env = MedicalEnv()
    # Wrap it to strictly enforce the dreamerv3 expected dictionary structure: 
    # returning obs['image'], obs['reward'], obs['is_first'], obs['is_terminal']
    env = DreamerV3Wrapper(env)
    return env

def main():
    print("Initializing Official DreamerV3 3D Medical Pipeline...")
    
    # 1. Load DreamerV3 Configuration and Override for our 3D flattened tensor input
    config = dreamerv3.configs['defaults']
    config = config.update({
        'task': 'medical',
        'run.steps': 1000000,
        'run.log_every': 100,
        'run.train_ratio': 32,
        'batch_size': 16,
        # Enforce that the tokenizer natively points to our Continuous flattened 3D image dict
        'encoder.mlp_keys': '$^',
        'decoder.mlp_keys': '$^',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
    })
    
    logdir = os.path.expanduser('~/logdir/medical_mri_dreamerv3')
    
    # 2. Setup DreamerV3 global bindings (JAX/TF/Torch backend initializations)
    dreamerv3.setup(logdir, config)
    
    # 3. Initialize metrics, step counters, inside the official loop
    step = dreamerv3.Counter()
    logger = dreamerv3.Logger(step, config.logdir)

    train_env = make_medical_env(config, logger, step)
    eval_env = make_medical_env(config, logger, step)

    # 4. Start the DreamerV3 Online RL Training Loop
    print("Starting DreamerV3 Online RL Loop Native Interaction...")
    # Based on danijar/dreamerv3 entry point APIs
    try:
        from dreamerv3.train import train
        train(make_medical_env, make_medical_env, config, logdir)
    except Exception as e:
        print("Note: If the dreamerv3 core loops are missing, ensure it's installed via pip install dreamerv3.")
        print(f"Exception details: {e}")

if __name__ == '__main__':
    main()
