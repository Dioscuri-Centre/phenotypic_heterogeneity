# Summary

This respository contains data and code related to the project `Phenotypic heterogeneity in cancer chemotherapy`
The project main experiments consist of micrograph time-lapses where of cells (potentially different cell lines) grown under different conditions (for instance using increasing drug doses) in 96-well plates.
The output data was consists mostly in statistical observables obtained from those time-lapses. The code and minimal input data (mostly tracks and cell outline segmentation) required to reproduce this data and the plots is provided here.


# Structure of the repository
The data and code is organised as follows:

## Tracking

Manual tracking was performed using trackmate [1] on specific experiments and wells, with or without fluorescence imaging. Associated to the actual images a detailed analysis allows to obtain information on motility, correlation between speed, division or fluorescence and so forth.
The notebook to be run is located in [the Tracking folder](Tracking/generate_figures.ipynb)

## Segmentation

Experiment #1 was segmented using PointRend [2] using a model trained on U-87MG cells. Fluorescence data extracted fron tracks was processed with Segment anything [3] using the weights from micro-sam [4]. The notebook that is required is in [the Segmentation folder](Segmentation/segmentation.ipynb)
Currently running the processing pipeline depends on installing [detectron2](https://detectron2.readthedocs.io/).


[1] Ershov, D., Phan, M.-S., Pylvänäinen, J. W., Rigaud, S. U., Le Blanc, L., Charles-Orszag, A., … Tinevez, J.-Y. (2022). TrackMate 7: integrating state-of-the-art segmentation algorithms into tracking pipelines. Nature Methods, 19(7), 829–832. doi:10.1038/s41592-022-01507-1

[2] Kirillov, Alexander, Yuxin Wu, Kaiming He, and Ross Girshick. “PointRend: Image Segmentation as Rendering.” arXiv, February 16, 2020. https://doi.org/10.48550/arXiv.1912.08193.

[3] Kirillov, Alexander, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, et al. “Segment Anything.” arXiv, April 5, 2023. https://doi.org/10.48550/arXiv.2304.02643.

[4] Archit, Anwai, et al. Segment Anything for Microscopy. 22 Aug. 2023. Bioinformatics, https://doi.org/10.1101/2023.08.21.554208.

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
  - `cd code`
  - `julia --project=.. inference.jl` (~5 min)
  - then run `analyze_exp33PD.ipynb` in VS Code.
