# Summary

This respository contains data and code related to the project `Phenotypic heterogeneity in cancer chemotherapy`
The project main experiments consist of micrograph time-lapses where of cells (potentially different cell lines) grown under different conditions (for instance using increasing drug doses) in 96-well plates.
The output data was consists mostly in statistical observables obtained from those time-lapses. The code and minimal input data (mostly tracks and cell outline segmentation) required to reproduce this data and the plots is provided here.


# Structure of the repository
The data and code is organised as follows:

* Tracking

Manual tracking was performed using trackmate [1] on specific experiments and wells, with or without fluorescence imaging. Associated to the actual images a detailed analysis allows to obtain information on motility, correlation between speed, division or fluorescence and so forth.

* Segmentation

A chosen set of experiments were automatically segmented using Cellpose [2], Boundary-Preserving Mask R-CNN [3], Omnipose [4]



[1] Ershov, D., Phan, M.-S., Pylvänäinen, J. W., Rigaud, S. U., Le Blanc, L., Charles-Orszag, A., … Tinevez, J.-Y. (2022). TrackMate 7: integrating state-of-the-art segmentation algorithms into tracking pipelines. Nature Methods, 19(7), 829–832. doi:10.1038/s41592-022-01507-1
[2] Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature methods, 18(1), 100-106.
[3] Cheng, Tianheng, Xinggang Wang, Lichao Huang, and Wenyu Liu. “Boundary-Preserving Mask R-CNN.” arXiv:2007.08921 [Cs], July 17, 2020. http://arxiv.org/abs/2007.08921.
[4]Cutler, Kevin J., Carsen Stringer, Teresa W. Lo, Luca Rappez, Nicholas Stroustrup, S. Brook Peterson, Paul A. Wiggins, and Joseph D. Mougous. “Omnipose: A High-Precision Morphology-Independent Solution for Bacterial Cell Segmentation.” Nature Methods 19, no. 11 (November 2022): 1438–48. https://doi.org/10.1038/s41592-022-01639-4.

# How to run the code
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
  - `mamba env create -f pyenv.yml`
  - `mamba activate dioscuri`

- Setup julia environment.
  - `julia`
  - `]`
  - `activate .`
  - `instantiate`
  - `precompile`

- Now Jupyter notebooks `.ipynb` and julia scripts `.jl` can be run.
  - For example, to run julia inference script run following
  - `cd code`
  - `julia --project=.. inference.jl` (~5 min)
  - then run `analyze_exp33PD.ipynb` in VS Code.