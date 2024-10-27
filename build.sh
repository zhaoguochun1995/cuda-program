export EXTRA_ARGS=' --verbose '
nvcc 01_helloworld.cu -o 01_helloworld ${EXTRA_ARGS}
nvcc 02_vector_add.cu -o 02_vector_add ${EXTRA_ARGS}
nvcc 03_matrix_add.cu -o 03_matrix_add ${EXTRA_ARGS}