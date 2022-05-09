# Primo + LinnOS

This directory contains code copied from the
[LinnOS Artifact](https://www.chameleoncloud.org/experiment/share/15?s=409ab137f20e4cd38ae3dd4e0d4bfa7chttps://www.chameleoncloud.org/experiment/share/15?s=409ab137f20e4cd38ae3dd4e0d4bfa7c) and adapted to use Primo. 
As stated in the paper, the original experiments in Chameleon Cloud are unreliable due to the unstable and random SSD I/O accesses (also argued by the authors). Consequently, we shift our implementation and evaluation on a bare-metal server from CloudLab.


## Getting Started

**Evaluation Time Estimation: 2 hours.**

**Note the artifact process need to reboot the server several times.**

### I. Start experiment.

Note ***Creat Experiment Profile*** and ***Start Experiment*** can be found in the upper left corner of the CloudLab page menu.

1. To request a specified machine from CloudLab, users need to ***Creat Experiment Profile*** first. We provide `CloudLab_profile.py`, you can select **Upload File** to create the profile.

2. Enter ***Start Experiment*** page and select the created profile. In the **3. Finalize** step, Cluster need to select ***CloudLab Utah***. Other settings remain default. The server will be kept for 16 hours and please finish the evaluation as soon as possible.

### II. Server setup.

3. After the server is provisioned, users can add ssh key from ***Manage SSH Keys*** page (the upper right corner of the CloudLab page menu) to access the server locally.

    ```bash
    ssh yourusername@pcap1.utah.cloudlab.us
    ```

4. Enter the server, and install `conda` environment.

    ```bash
    mkdir LinnOS
    cd LinnOS
    wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
    bash Miniconda3-py37_4.10.3-Linux-x86_64.sh

    (Select 'yes' for: Do you wish the installer to initialize Miniconda3 by running conda init? [yes|no])
    ```

5. To make the main disk space enough for experiment, we need to repartition the system drive `sda`:

    ```bash
    sudo fdisk /dev/sda
    p
    d
    1
    d
    2
    d
    3
    d
    n
    p
    1
     (Default for First sector and Last sector)
    Y(Select 'Y' for: Do you want to remove the signature? [Y]es/[N]o)
    w
    sudo reboot
    ```

    Wait for 5~10 minutes and reconnect the server.

    ```bash
    sudo resize2fs /dev/sda1
    df -h  (You can check /dev/sda1  should have 1.5T space.)
    ```

6. Install JupyterLab and Bash support for evaluation part.

    ```bash
    cd LinnOS
    rm -rf Miniconda3-py37_4.10.3-Linux-x86_64.sh
    conda install jupyterlab
    pip install bash_kernel gdown seaborn
    python -m bash_kernel.install
    conda install pandas matplotlib --yes
    ```

7. Download artifact files from Google Drive to the server and prepare the environment.

    ```bash
    gdown 1Y2DkU2rbpGijTUa0fCFcgzYEq5iwLtQr # linux-5.4.8-primo.tar.gz
    gdown 1VW3tokiiBa_sKVlobfRtn73bstHC-UKX # linux-5.4.8-linnos.tar.gz
    gdown 1axMULigH5G8m3gIRGZ0RycPpES6ADs47 # LinnOSWriterReplayer.tar.xz
    ```

    ```bash
    tar -xf LinnOSWriterReplayer.tar.xz
    tar -zxf linux-5.4.8-linnos.tar.gz
    tar -zxf linux-5.4.8-primo.tar.gz
    ```

    ```bash
    sudo apt update
    sudo apt -y install build-essential libncurses-dev bison flex libssl-dev libelf-dev
    ```

8. Start JupyterLab and subsequent experiments will be performed in `Primo_LinnOS.ipynb`.

    ```bash
    tmux new-session -t jupyter
    bash generatefiles.sh > writelog 2>&1 &
    sudo ~/miniconda3/bin/jupyter lab --allow-root
    
    (Click the link to access JupyterLab in browser.)
    ```


### III. Performance evaluation.
    
Details refer to `Primo_LinnOS.ipynb`. Run each code block to obtain results traces for different algorithms.


**IMPORTANT**: Two Linux kernels will be installed and reboot for two times during the evaluation. After reboot, run the following script to continue the experiment.

```bash
uname -mrs  # Check the kernel version whether changed correctly.
tmux new-session -t jupyter
bash generatefiles.sh > writelog 2>&1 &
sudo ~/miniconda3/bin/jupyter lab --allow-root

(Click the link to access JupyterLab in browser.)
```

**IMPORTANT**: Change to Primo kernel need to edit `/etc/default/grub` as stated in the notebook

### IV. Results visualization.
    
Enter `ResultsPlot.ipynb`. Run each code block to check the results.

### Optional: Replace the pre-trained model (Time Estimate: 6 manual hours).

We provide pre-trained models for both LinnOS and Primo for more convenient evaluation. If you want to reproduce the full pipeline of LinOS and Primo implementation, including trace collection, neural network and PrDT model training, quantization and integrate into Linux kernel. 

Please refer to `Model_Retrain.ipynb`. All related files are provided.
