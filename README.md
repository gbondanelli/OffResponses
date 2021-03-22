# OffResponses
Supplementary code for the paper "Network dynamics underlying OFF responses in he auditory cortex" by Giulio Bondanelli, Thomas Deneux, Brice Bathellier and Srdjan Ostojic, eLife (2021)

Data folder:

- off_responses_trialavg.npy: Gaussian smoothed trial-averaged OFF responses (2343 neurons x 102 timepoints x 16 stimuli)
- off_responses_single_trials.npy: Gaussian smoothed single-trial OFF responses (2343 neurons x 102 timepoints x 16 stimuli x 20 trials)
- infos.npy: additional information (column 1: cell number; column 2: mouse number; column 3: session number)

Modules folder:

- contains functions used in the scripts .py

Instructions: run the .py files to generate the following sample figures

- Fig.3C: fit_single_cell_model
- Fig.3H: fit_recurrent_model
- Fig.5B,C: predictions_on_r0_U_V 
- Fig.5D: corr_r0_peak_data
- Fig.5E: corr_r0_peak_vs_numstim
- Fig.6: example_dynamics_recurrent
- Fig.7A: connectivity_overlaps
- Fig.7B: connectivity_vs_subspace_overlaps
- Fig.7C: comparison_fit
- Fig.8B: dynamics_variability

