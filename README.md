# Sucuri-GPUNode
This repository contains the Sucuri version with GPUNode using Numba.


# Sucuri
Sucuri is a Dataflow Programming library in Python which helps developers program their application in parallel through a Dataflow graph.

Sucuri main repository: https://bitbucket.org/flatlabs/

For more information:<br/>
<pre>
@inproceedings{alves2014minimalistic,
  title={A minimalistic dataflow programming library for python},
  author={Alves, Tiago AO and Goldstein, Brunno F and Fran{\c{c}}a, Felipe MG and Marzulo, Leandro AJ},
  booktitle={Computer Architecture and High Performance Computing Workshop (SBAC-PADW), 2014 International Symposium on},
  pages={96--101},
  year={2014},
  organization={IEEE}
}
</pre>

# Sucuri with GPUNode
Inside the Sucuri folder, you have the original sucuri examples and the GPU Node implementation inside pyDF folder alongside the Sucuri version.

Inside the Blackscholes and Convolutional Separable folder, there are examples of how to use the two versions of GPUNode and a version only using the Numba without Sucuri. The Kernels implementations are based on: https://github.com/fernandoc1/Benchmarking-CUDA

Obs: Some versions of Numpy are not working with this version of Sucuri, so you must convert to list before returning to a node like the BlackScholes and Convolutional examples.

## Requirements:
Python 2.7 for Sucuri
Numba for Sucuri with GPUNode

To install Numba:
> Install Anaconda.<br/>
> conda install numba<br/>
> conda install cudatoolkit<br/>
