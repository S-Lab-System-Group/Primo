
```
    ____       _               
   / __ \_____(_)___ ___  ____ 
  / /_/ / ___/ / __ `__ \/ __ \
 / ____/ /  / / / / / / / /_/ /
/_/   /_/  /_/_/ /_/ /_/\____/ 
                   
```
# Primo

Primo is a unified framework that assists developers to design practical learning-augmented systems with interpretable models. To learn more about how Primo works, please refer our [ATC '22 paper](TODO) ***"Primo: Practical Learning-Augmented Systems with Interpretable Models"***.


## Installation

```
pip install pyprimo
```

## Artifacts

This folder contains the artifact for the ATC paper. Specifically, we provide the Primo implementation for each case study (`Section 4: LinnOS`, `Section 5: Clara`, `Section 6: Pensieve`):


|  | System Scenario | ML Algorithm | Offical Repo |
|---|---|---|---|
| LinnOS ([OSDI   '20](https://www.usenix.org/conference/osdi20/presentation/hao)) | Flash Storage I/O | DNN | [LinnOS Code](https://www.chameleoncloud.org/experiment/share/15?s=409ab137f20e4cd38ae3dd4e0d4bfa7chttps://www.chameleoncloud.org/experiment/share/15?s=409ab137f20e4cd38ae3dd4e0d4bfa7c) |
| Clara ([SOSP   '21](https://dl.acm.org/doi/10.1145/3477132.3483583)) | SmartNIC Offloading | Mixture(LSTM, GBDT, SVM) | [Clara Code](https://github.com/824728350/Clara) |
| Pensieve ([SIGCOMM   '17](https://dl.acm.org/doi/10.1145/3098822.3098843)) | Adaptive Video Streaming | RL | [Pensieve Code](https://github.com/hongzimao/pensieve) |

You can use these scripts to learn how to leverage Primo model in different system scenarios and reproduce the results presented in our paper.



## Acknowledgements

Primo build on top of many great open-source repositories, including

[interpret](https://github.com/interpretml/interpret) | [scikit-learn](https://github.com/scikit-learn/scikit-learn) |  [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) | [viper](https://github.com/obastani/viper)


## Citation

```
@inproceedings{Primo,
  title     = {Primo: Practical Learning-Augmented Systems with Interpretable Models},
  author    = {Qinghao Hu and Harsha Nori and Peng Sun and Yonggang Wen and Tianwei Zhang},
  booktitle = {2022 {USENIX} Annual Technical Conference},
  publisher = {{USENIX} Association},
  year      = {2022},
  series    = {{USENIX} {ATC} '22}
}
```