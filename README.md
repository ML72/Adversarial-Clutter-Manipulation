# Adversarial Cluttered Manipulation

## Environment Setup

This cluttered manipulation environment runs on Isaac Lab, which has to first be installed. At a high level, the overall installation procedure involves first installing Isaac Sim via pip, and then building Isaac Lab from source.

This procedure works for both Linux Ubuntu and Windows. This documentation is mostly based on the [official Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).

### Part 1: Install Isaac Sim

1. Create an environment running Python 3.10. Conda and venv both work. To create a conda environment named `clutter_manip`: 

    ```
    conda create -n clutter_manip python=3.10
    conda activate clutter_manip
    ```

2. Install a CUDA-enabled PyTorch 2.5.1 build. Run the following for CUDA 12:

    ```
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    ```

3. Install the Isaac Sim package:

    ```
    pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
    ```

    Optionally, install Isaac Sim cached extension dependencies. These are not included in the main python package and should be downloaded upon demand at runtime, but installing them manually avoids future problems with downloads from the registers.

    ```
    pip install isaacsim-extscache-physics==4.5 isaacsim-extscache-kit==4.5 isaacsim-extscache-kit-sdk==4.5 --extra-index-url https://pypi.nvidia.com
    ```

4. **Windows only:** If there is an error in any of the previous step related to Windows Long Path support, enable long path support by opening Powershell in Administrator mode and running the following command (separate from the current terminal):

    ```
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
    ```

5. To verify the installation, run the following command. The first run may take a few minutes as dependent extensions are pulled from the registry. Additionally, the first run will prompt users to accept the Nvidia Omniverse License Agreement.

    ```
    # Running the simulator
    isaacsim

    # Running a specific experience file
    isaacsim omni.isaac.sim.python.kit
    ```

6. **Windows only:** If there is an error in the previous step related to "`fbgemm.dll` or one of its dependencies", the issue is likely caused by a missing `libomp140.x86_64.dll` file.

    To optionally verify that a missing `libomp140.x86_64.dll` is causing the issue, install and extract `Dependencies_x64_Release.zip` from [source](https://github.com/lucasg/Dependencies/releases), run `DependenciesGui.exe` from the extracted folder, and open the `fbgemm.dll` file specified in the error message.

    To fix the issue, install an `libomp140.x86_64.dll` file. There are many possible sources to install from, but we found [dllme](https://www.dllme.com/dll/files/libomp140_x86_64/versions) to work.
    
    After `libomp140.x86_64.dll` is installed, put it *either under the same folder as `fbgemm.dll` or under `Windows/System32`*. We recommend the former for sake of security. Note that with default conda settings, `fbgemm.dll` should be under `<path-to-conda>/anaconda3/envs/<env-name>/Lib/site-packages/torch/lib`.

### Part 2: Build Isaac Lab

1. **Linux only:** Install dependencies:

    ```
    sudo apt install cmake build-essential
    ```

2. From the root of this repository, iterate over all extensions in `source/extensions` and install them using pip.

    On Linux:
    ```
    ./isaaclab.sh --install
    ```

    On Windows:
    ```
    isaaclab.bat --install
    ```

3. To verify the installation, run the following command. Feel free to also run other standalone script under `scripts/demos` to verify that Isaac Lab is working properly.

    ```
    python scripts/tutorials/00_sim/create_empty.py
    ```

### Part 3: Install Custom SKRL Version

1. We want to install our custom SKRL module, which contains adversarial training code. First uninstall the existing version of SKRL:

    ```
    pip uninstall skrl
    ```

2. Install the library in editable mode:

    ```
    cd libraries/skrl_custom
    pip install -e .["torch"]
    ```

## Running Experiments

### Project Structure

We use a custom version of SKRL for RL. The entrypoints for training and evaluation are as follows:

- Train entrypoint: `scripts/reinforcement_learning/skrl/train.py`
- Eval entrypoint: `scripts/reinforcement_learning/skrl/play.py`

Important files (non-exhaustive list):

- `source/isaaclab/isaaclab/envs/adversarial_manager_based_rl_env.py`: Contains the `adversarial_reset` method, which is support for adversarial positioning on the Isaac Lab side
- `libraries/skrl_custom/skrl/trainers/torch/base.py`: Contains the `single_agent_train` method, which contains the adversarial training loop
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/`: Contains Isaac Lab environment configuration files
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/skrl_ppo_cfg.yaml`: Contains configuration for SKRL training, such as timesteps, LR, checkpoint intervals, etc.

During training, log files are saved to the `logs/skrl/franka_lift/` folder.

During eval, save files are conventionally saved to the `results/skrl/` folder.

### Running Experiments

Arguments (see entrypoint source code for full details/options):

- `--task`: The Isaac Lab training task, specified by Hydra
- `--num_envs`: Number of environments to run in parallel
- `--checkpoint`: Path to model checkpoint to resume training or eval model
- `--headless`: This flag is passed to disable initialization of the simulation UI
- `--enable_cameras`: This flag is passed for environments needing a camera
- `--positioning` (train only): Adversarial positioning strategy
- `--max_episodes` (eval only): Number of episodes to run, total number of rollouts is `max_episodes * num_envs`
- `--save_file` (eval only): File to save position, reward, and success data to

Example train commands for pre-training models on state-based observations in the simple environment (remember to increase timesteps in configuration accordingly):

```
isaaclab.bat -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Lift-Cube-Franka-Simple-v0 --num_envs 256 --headless --positioning domain_rand --name simple_domainrand_pt --seed 42
isaaclab.bat -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Lift-Cube-Franka-Simple-v0 --num_envs 256 --headless --positioning domain_rand_restricted --name simple_domainrandrestrict_pt --seed 42
```

Example eval commands for evaluating the models trained above:

```
isaaclab.bat -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Lift-Cube-Franka-Simple-v0 --num_envs 50 --checkpoint logs/skrl/franka_lift/simple_domainrand_pt/checkpoints/best_agent.pt --max-episodes 40 --save-file simple_domainrand_pt.json --headless
isaaclab.bat -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Lift-Cube-Franka-Simple-v0 --num_envs 50 --checkpoint logs/skrl/franka_lift/simple_domainrandrestrict_pt/checkpoints/best_agent.pt --max-episodes 40 --save-file simple_domainrandrestrict_pt.json --headless
```

Example eval command for creating a video (saved inside the `logs/skrl/franka_lift/` folder):

```
isaaclab.bat -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Lift-Cube-Franka-Simple-v0 --num_envs 4 --checkpoint logs/skrl/franka_lift/simple_domainrand_pt/checkpoints/best_agent.pt --headless --video --video_length 20000
```

Example train commands for fine-tuning a pretrained model, assuming the specified model exists under the `pretrained/` folder:

```
isaaclab.bat -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Lift-Cube-Franka-Simple-v0 --num_envs 256 --headless --positioning pure_adversary --name simple_pt-dr_ft-pureadv --checkpoint pretrained/simple_domainrand_pretrain.pt --seed 1
isaaclab.bat -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Lift-Cube-Franka-Simple-v0 --num_envs 256 --headless --positioning domain_rand --name simple_pt-dr_ft-dr --checkpoint pretrained/simple_domainrand_pretrain.pt --seed 1
```

### Tensorboard Logs

We use Tensorboard for logging statistics during training. To start up Tensorboard logs:

```
tensorboard --logdir=logs\skrl\franka_lift
```
