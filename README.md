# Alpha Zero with Visual Transformer
CSC413 Final Course project for the University of Toronto. We applied a ViT or Visual Transformer to AlphaZero's self-learning algorithm.
- AlphaZero Julia library was sourced from https://github.com/jonathan-laurent/AlphaZero.jl
- We implemented the ViT at AlphaZero.jl/src/networks/architectures/vit.jl
- We also fixed a bug in the Pons Benchmark script at AlphaZero.jl/games/connect-four/scripts/pons_benchmark.jl

Note: data in AlphaZero.jl/games/connect-four/benchmark is sourced from http://blog.gamesolver.org/solving-connect-four/02-test-protocol/

To train this algorithm, first clone this repository. Ensure that Julia 1.8.1+ is installed.
We have the model parameters stored in the AlphaZero.jl/games/connect-four/ directory. To pick a model to train, simply rename the desired model params-[**model_name**].jl to params.jl 

Then, within the top level AlphaZero.jl directory, run
```
export GKSwstype=100  # To avoid an occasional GR bug
julia --project -e 'import Pkg; Pkg.instantiate()'
julia --project -e 'using AlphaZero; Scripts.train("connect-four")'
```