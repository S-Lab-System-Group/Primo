# Primo + Pensieve

[Pensieve](https://github.com/hongzimao/pensieve) is a system that generates adaptive bitrate algorithms using reinforcement learning.

We provide the source code of the simulation and the deployment in reality. 

## Getting Started

**Evaluation Time Estimation: 2.5 hours.**

### I. Test with all baselines. (15 minutes)

```bash
cd Pensieve/baselines
sh test_all.sh
```

Note since there are high randomness of Pensieve result across different machines, we skip it by default.

Results will be stored in `./results` and `./results_fcc` respectively. We provide pretrained models of Pensieve, Metis and Primo for reproduction.


```bash
cd ..
jupyter lab
```

Open `Result_Plot.ipynb` and run all cells within the notebook to visualize results.

### II. Primo distill engine. (15 minutes)

Primo provide `Distill Engine` for RL model replacement.

```bash
python primo_pensieve.py -hsdpa
```

This will replace the pretrained Primo model automatically in `primo_results`.


### III. Metis model generation. (2 hours)

The [Metis](https://github.com/transys-project/metis/tree/master/interpret-pensieve) model reproduction follows its default setting.


```bash
cd baselines/metis
python metis.py -hsdpa
```

This will replace the pretrained Metis model automatically in `metis_model`.

You can follow the `I. Test with all baselines.` to test trained Primo and Metis performance again.


### Optional:  Deployment in reality. (2 hours)

Please refer to [DEPLOY.md](./DEPLOY.md). We provide details on how to deploy Primo in reality. 

Note this evaluation requires a Linux desktop (recommend Ubuntu 20.04) with GUI, and use Chrome to play the video and record latency & memory information.