# Solver code generation

### Setup

Install Fatrop and Blasfeo. Make sure the Blasfeo include directory is correct in CMakeLists.txt.

### Usage

1. Copy `compiled_solver.c` file into this folder, that is generated when running examples with `compile_solver=True`.
2. Compile in `/build` folder.
3. Copy resulting library to `/lib` folder. Make sure the name `load_compiled_solver` is set correctly in the examples.


### Euler

Careful: Need to compile Fatrop and Blasfeo locally, since they cannot be installed on the server. Then their directories need to be included in CMakeLists.txt.

To compile the solver:

1. Load cmake module:
```bash
module load stack/2024-06 gcc/12.2.0 cmake/3.27.7
```
2. Run batch job:
```bash
sbatch --time=1:00:00 --mem-per-cpu=16G --wrap="make"
```