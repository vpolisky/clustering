#A custom GDBSCAN algorithm implementation and a tool for comparison of different clustering algorithms

## Custom GDBSCAN
DBSCAN implementation accepting different norms and different types of density checks. Currently supported is Euclidean 
and Manhattan distance and point count based density check.

## Algorithm Comparison
Runs multiple clustering algorithms - in particular, the custom DBSCAN implementation - on artificial non-convex data 
with different levels of noise.

The evaluation produces text output in terminal and plots.

To run:

- create and activate a virtualenv from Python 3.6+
- `pip install -r requirements.txt`
- `python evaluation.py`
- after some time, plots will start showing up
- close plot to continue
