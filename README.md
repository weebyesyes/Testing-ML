# Testing-ML
Repository contains a small selection of code from Kaggle for the NeurIPS Open Polymer Prediction 2025.

## Model overview

The main idea overall is: convert polymer SMILES into molecular descriptors, train separate target-specific XGBoost models for each property, remove weak and redundant features, and then blend the models using holdout-based calibration for more robust final predictions.

More specifically, this pipeline:
- cleans and canonicalizes SMILES strings,
- integrates supplementary external datasets,
- builds descriptor-based tabular features (mainly Mordred / RDKit-based),
- performs repeated feature selection and correlation pruning,
- trains separate models for `Tg`, `FFV`, `Tc`, `Density`, and `Rg`,
- uses cross-validation and holdout blending to improve generalization.

I also experimented with a graph neural network (GNN) pipeline, and the final submission averaged predictions from the descriptor-based XGBoost pipeline and the GNN ensemble.
