## Experiments playbook

All scripts are in folder `code`

### Requirements
This code requires Gurobi (with a valid license) to be installed to the python environment used - this could be done at https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html

### Tight cuts

Tu run the experiment, do
```angular2html
python tight_cuts.py
```

The parameters `n` (ground set size), `m` (number of iterations/points), `outer` (number of runs) can be changes in the 
script.

To generate plots for the experiment, dp
```angular2html
python plot_tight_cuts.py
```

### Online learning

#### Over permutahedron

To run the experiment, do
```angular2html
python online_learning_permutahedron.py
```

The parameters `n` (ground set size), `T` (number of iterations in the online problem), `outer` (number of runs), `a` 
(number of permutations used to generate loss function), `b` (swap distance between any two permutations used to 
generate loss function) can be changed within the script.

To generate runtime plot for OMD-AFW variants,
```angular2html
python plot_omd_runtime.py
```

To generate regret plot for OMD-AFW variants,
```angular2html
python plot_omd_regret.py
```

To generate AFW iterates plot for OMD-AFW variants,
```angular2html
python plot_omd_iterates.py
```

To generate regret/time plot for OMD-AFW variants,
```angular2html
python plot_omd_regret_over_time.py
```

#### Over a noncardinality based submodular function

To run the experiment, do
```angular2html
python online_learning_noncardinality_submodular.py
```

The parameters `n` (ground set size), `T` (number of iterations in the online problem), `outer` (number of runs), `a` 
(number of permutations used to generate loss function), `b` (swap distance between any two permutations used to 
generate loss function), `p` (probability associated with generating random graph for submodular function) 
can be changed within the script.

To generate runtime plot for OMD-AFW variants,
```angular2html
python plot_omd_runtime.py
```

To generate regret plot for OMD-AFW variants,
```angular2html
python plot_omd_regret.py
```

To generate AFW iterates plot for OMD-AFW variants,
```angular2html
python plot_omd_iterates.py
```

To generate regret/time plot for OMD-AFW variants,
```angular2html
python plot_omd_regret_over_time.py
```
