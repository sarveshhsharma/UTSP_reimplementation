# UTSP_reimplementation
This is a replica of the model described in Unsupervised Learning for Solving the Travelling Salesman Problem, but implemented using a Graph Attention Network (GAT) instead.

## Instructions for Using the Model
Run train.py to initiate the training process and learn the necessary parameters.

After training, a model file containing the trained parameters will be saved in the Saved_models folder.

Navigate to load_model.py, load the trained model from the Saved_models directory, and perform predictions.

The following outputs will be generated:
* Heatmap
* Predicted path
* Predicted path length
