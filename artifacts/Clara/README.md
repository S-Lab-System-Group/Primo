# Primo + Clara

[Clara](https://github.com/824728350/Clara) is an automated tool that improves the productivity of SmartNIC NF offloading workflow by generating offloading insights. 

We adopt Primo to optimize Clara system. Each folder contains a standalone module of Clara.

+ `CP_compute_prediction`: Generate compute prediction offloading insights, predict number of compute instructions after offloading into SmartNIC.

+ `AI_algo_id`: Generate algorithm identification offloading insights, find crc and lpm accelerating opportunities hidden in click elements.

+ `MS_scale_out`: Generate scale out analysis offloading insights, find the "best" number of calls to use for high throughput without sacrificing latency.




## Getting Started

**Evaluation Time Estimation: 20 minutes.**

All baseline algorithms keep the same configuration as Clara's original setting.

### I. Clara-MS.

a. (**Figure 9**) Obtain baseline algorithm results.
```bash
cd MS_scale_out/baselines
sh clara_ms_baseline.sh
cd ..
```

b.  Open `Primo_Clara_MS.ipynb` and run all cells within the notebook to evaluate Primo performance, Primo model interpretation (**Figure 8**) and monotonic constraint effect .


### II. Clara-AI.

a. (**Figure 10**) Obtain baseline algorithm results.
```bash
cd ../AI_algo_id/baselines
sh clara_ai_baseline.sh
cd ..
```

b.  Open `Primo_Clara_AI.ipynb` and run all cells within the notebook to evaluate Primo performance and counterfactual explanation effect.

### III. Clara-CP.

a. (**Figure 11**) Obtain baseline algorithm results.
```bash
cd ../CP_compute_prediction/baselines
sh clara_cp_baseline.sh
cd ..
```

b.  Open `Primo_Clara_CP.ipynb` and run all cells within the notebook to evaluate Primo performance.

