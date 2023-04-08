# GDL_Submission
Code for the Geometric Deep Learning mini-project

For a quick demo run:

best_model_for('Squirrel') or best_model_for('Chameleon') from best_model.py



The model's code is in EFGNN.py

MultiHopTransform.py just augments the data with some precomputed stuff, like CSR adjacencies, etc.

train contains a train function

AttLayer adds a learnable attention-based filter, but it doesn't seem to bring any performance benefits at this point.
