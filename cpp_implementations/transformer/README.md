# C++ Transformer Implementation

## Details

- Rotary Positional Embeddings: GPT2-NeoX style rotary positional embeddings implemented
- KV caching: Key and Value caching is used to avoid recalculating attention scores
- `transformer/components` contains all of the components used in implementing
- `transformer/driver.cpp` is the main file, edit this to edit transformer layers, dimensions, etc

## Build instructions

- Install [libtorch](https://pytorch.org/cppdocs/installing.html)
- To build:
  - Edit `transformer/builds/build.sh` and point `-DCMAKE_PREFIX_PATH` to the absolute file path to libtorch
  - Run `transformer/builds/build.sh`
- To run executable: Run `transformer/builds/transformer`

## Future TODOs

- It would be cool to import model weights from Pythia into this implementation
- Unfortunately, I have not figured out an easy way to get tokenizers into C++ to do the above
