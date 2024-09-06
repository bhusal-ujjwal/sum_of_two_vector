# Sum of two vector using cuda:
## Implementation of an algorithm that will take two vectors (A and B) and sum the corresponding elements in these vectors into the third vector C. The algorithm should work with vectors of variable length containing integer numbers. (All three vectors will have the same dimension.) The algorithm will be implemented in three ways:
1) Sum the vectors by using the CPU.
2) Sum the vectors by using the GPU and manually managed memory.
3) Sum the vectors by using the GPU and memory managed automatically.

- Execute algorithms 100 times for various vector dimensions (start at 100 000 elements and use 10 000 as an increment). For each execution, measure the time consumed by the first algorithm (except memory allocation) and the time consumed by other algorithms (including operations specific to these implementations: memory allocation on the GPU, data copy, etc.).

- Print out the measured times, including the vector dimension, into the standard output, separated by a comma.

- Run your application and redirect the output into the data.csv file. Open the file in MS Excel and produce a graph comparing the times across the dimensions.
