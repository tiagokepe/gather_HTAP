# gather_HTAP
Emulating AVX512 gather instructions for Hybrid Transactional/Analytical Processing (HTAP).
The workloads are based on the paper: [Gather-scatter DRAM: in-DRAM Address Translation to Improve the Spatial Locality of Non-Unit Strided Accesses](https://dl.acm.org/doi/10.1145/2830772.2830820) 

The workloads set up is the following:

*In-memory databases

*Row-oriented layout

*Single table with 8 columns of 8-bytes (double)

*Tuple size of 64-bytes, fitting in one cache line

*1 milling tuples


## Analityc Workload

Sum two columns using AVX512 gather instructions to load the columns and writing the result into a new array


## Getting started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

The next command will generate an executable withou printfs: analytic_realease.

```bash
$ make
```

To generate an executable with debug outputs, run:

```bash
$make debug

```

To clean up the project from old object and executable files, run:

```bash
$make clean

```
