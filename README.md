# ga-mmwave
This repository contains datasets, demo code, and results that complement academic material published in the IEEE Antennas and Wireless Propagation Letters.</br>
**Using these datasets requires appropriate citations.**

[S.B.F Gomes, N. Simmons, P. C. Sofotasios, M. D. Yacoub, and S. L. Cotton, "Genetic Algorithms for Channel Parameter Estimation in Indoor Millimeter-Wave Environments", published in the IEEE Antennas and Wireless Propagation Letters, 14 Sept. 2023.](https://ieeexplore.ieee.org/abstract/document/10251547/)

## Dependencies

This program is written in Python 3.9 and uses some dependencies listed in requirements.txt:

`pip install -r requirements.txt`

## How to use

Just run the demo main file:

`python main.py`

It will create/use a folder called runs on the project root and plot the figures inside for each call of main.py. 

To set the arguments, please go to _arguments.py_ file.

This demo does not save the estimations anywhere. </br> To see the estimations for each
algorithm, look at the command prompt.

## Citation

If feel inspired, please consider cite:

```@ARTICLE{10251547,
  author={Gomes, Samuel Borges Ferreira and Simmons, Nidhi and Sofotasios, Paschalis C. and Yacoub, Michel Daoud and Cotton, Simon L.},
  journal={IEEE Antennas and Wireless Propagation Letters}, 
  title={Channel Parameter Estimation in Millimeter-Wave Propagation Environments Using Genetic Algorithm}, 
  year={2024},
  volume={23},
  number={1},
  pages={24-28},
  keywords={Fading channels;Genetic algorithms;Parameter estimation;Channel estimation;Sociology;Wireless communication;Probability density function;Channel measurements;genetic algorithms;meta-heuristic algorithms;millimeter-wave communications},
  doi={10.1109/LAWP.2023.3315422}}
```




