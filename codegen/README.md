# Solver code generation

### Setup

Install Fatrop and Blasfeo.

### Usage

1. Copy `compiled_solver.c` file into this folder, that is generated when running examples with `compile_solver=True`.
2. Compile in `/build` folder.
3. Copy resulting library to `/lib` folder. Make sure the name `load_compiled_solver` is set correctly in the examples.