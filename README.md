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

### Part 3: Install Submodule SKRL Version

1. We want to install our custom SKRL module, which contains adversarial training code. First uninstall the existing version of SKRL:

    ```
    pip uninstall skrl
    ```

2. Initialize the submodules by running the following commands:

    ```
    git submodule update --init --recursive
    ```

3. Install the submodule in editable mode:

    ```
    cd submodules/skrl-adv
    pip install -e .["torch"]
    ```
