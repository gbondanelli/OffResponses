# OffResponses
Supplementary code for the paper "Network dynamics underlying OFF responses in he auditory cortex" by Giulio Bondanelli, Thomas Deneux, Brice Bathellier and Srdjan Ostojic (2020)

Data folder:

- off_responses_trialavg.npy: Gaussian smoothed trial-averaged OFF responses (2343 neurons x 102 timepoints x 16 stimuli)
- off_responses_single_trials.npy: Gaussian smoothed single-trial OFF responses (2343 neurons x 102 timepoints x 16 stimuli x 20 trials)
- infos: additional information (column 1: cell number; column 2: mouse number; column 3: session number)

Modules folder:

- contains functions used in the scripts .py

Instructions: run the .py files to generate the following sample figures

- example_dynamics_recurrent -> Fig.3B-E
- corr_r0_peak_recurrent -> Fig.3F
- fit_recurrent_model -> Fig.4B
- predictions_on_r0_U_V -> Fig.4E-F
- corr_r0_peak_data -> Fig.4G
- orthogonal_transient_channels -> Fig.4H right
- fit_single_cell_model -> Fig.5B-C
- corr_r0_peak_vs_numstim -> Fig.5F
- dynamics_variability -> Fig.6B

