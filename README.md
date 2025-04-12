# Content 
This respository contains data and code related to the publication [A quantitative characterization of the heterogeneous response of glioblastoma U-87 MG cell line to temozolomide](https://www.biorxiv.org/content/10.1101/2024.05.28.596108v1)
The project main experiments consist of micrograph time-lapses where of cells (potentially different cell lines) grown under different conditions (for instance using increasing drug doses) in 96-well plates.
The output data was consists mostly in statistical observables obtained from those time-lapses. The code and minimal input data (mostly tracks and cell outline segmentation) required to reproduce this data and the plots is provided here.


## Structure of the repository
The data and code is organised as follows:
- `figures/` contains all figures from the paper as fig_xx.svg
    * `figures/plots`: contains plots required to generate the figures
- `data/` Data pertains to all types of data required to generate the plots, this can be experimental or intermediate with the following convention: `data/type_of_experiment_1/condition/intermediate_data_file{.xml|.csv|npy|etc}`
  for instance `data/transfection/2/10um/fluo_B4.bin` relates to the second transfection experiment, 10uM TMZ drug and contains the preprocssed fluorescence data.
- `figures/` contains notebooks for all the figures.
- `analysis/utils/`  routines that are called by each of the figure notebooks have been outsourced there


## Tracking analysis

Manual tracking was performed using trackmate [1] and Mastodon  (https://mastodon.readthedocs.io)  on specific experiments and wells, with or without fluorescence imaging. Tracks can be downloaded separately by downloading the release `revision tracks` on github. 

## Segmentation

Most experiments were segmented using Cellpose.

## Tracking
Fluorescence data was extracted from tracks that were generate by Mastodon/trackmate  either fully manually or by using the LAP tracker on spots generated from Cellpose label images.

## Simulations

This repository uses python and julia. We recommend [`miniforge`](https://github.com/conda-forge/miniforge) for python and [`juliaup`](https://github.com/JuliaLang/juliaup) for julia. Please find the installation instruction on their respective GitHub repo and documentation. You will need a software to view and run jupyter notebooks. We recommend [VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

After the install follow these steps:
- Check that `conda`, and `mamba` are working from your terminal.
  - `mamba --version`

- Install julia v1.9 using `juliaup`.
  - `juliaup add 1.9`
  - `juliaup default 1.9`

- Clone this repo.
  - `git clone git@github.com:Dioscuri-Centre/phenotypic_heterogeneity.git dioscuri`

- Change directory to the root directory of the repo.
  - `cd dioscuri`

- Setup python environment.
  - `mamba env create`
  - `mamba activate dioscuri`

- Setup julia environment.
  - `julia`
  - `]`
  - `activate .`
  - `instantiate`
  - `precompile`

- Now Jupyter notebooks `.ipynb` and julia scripts `.jl` can be run.
  - For example, to run julia inference script run following
  - `cd simulations`
  - `julia --project=.. inference.jl` (~5 min)
  - then run `analyze_exp33PD.ipynb` in VS Code.



# References 

[1] Ershov, D., Phan, M.-S., Pylvänäinen, J. W., Rigaud, S. U., Le Blanc, L., Charles-Orszag, A., … Tinevez, J.-Y. (2022). TrackMate 7: integrating state-of-the-art segmentation algorithms into tracking pipelines. Nature Methods, 19(7), 829–832. doi:10.1038/s41592-022-01507-1

