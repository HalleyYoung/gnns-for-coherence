# gnns-for-coherence
An examination of the use of several methods (including a novel approach with graph convolutional networks).

Here are the steps to run:

1) Obtain the NYT annotated corpus (https://catalog.ldc.upenn.edu/LDC2008T19).
2) Run getnyt.py
3) Run conceptnet_nytsent_script.py
4) To run RNN for sentence ordering or quality classification, run one of the scripts in the RNNBaseline folder.
5) To run a GCN, run the corresponding "depframe" file in the generate_graph folder, and then run the corresponding "graphclass" file in the GCN folder.
6) To run the coreference baselines, run the files in the coref_baseline folder.
