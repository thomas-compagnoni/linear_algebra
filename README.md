# Linear Algebra

The project replicates the main alghoritms used in linear algebra.
The function follow the contents of the course in Linear Algebra (18.06) teached by professor Gilbert Strang at MIT, freely available on [YouTube](https://youtube.com/playlist?list=PL49CF3715CB9EF31D).

Even though these functions are available in all major programming software and have been perfected for calculation speed and accuracy, I attempted to write them from scratch in order to practise my Python skills and better understand the concepts covered in the course.

## List of functions

- **PLU**: Lower–Upper (LU) decomposition or factorization
- **rref**: Return the Reduced Row Echelon Form of a matrix
- **inverse**: Return a matrix inverse using the Gauss-Jordan method
- **column_space**: Compute the column space of a matrix; C(A)
- **nullspace**: Compute the nullspace of a matrix; N(A)
- **row_space**: Compute the row space of a matrix; C(A.T)
- **left_nullspace**: Compute the left nullspace a matrix; N(A.T)
- **solve**: Solve the system Ax = b using the Gaussian elimination
- **orthogonality**: Test whether two matrices are orthogonal
- **projection_matrix**: Return the projection matrix of a matrix A
- **project_vector**: Project a vector onto a subspace
- **gram_schmidt**: Transform columns of a matrix A into orthonormal vectors
- **determinant**: Compute the determinant as the product of the diagonal of the reduced form U
- **determinant_cofactor**: Compute the determinant using the Cofactor formula
- **inverse_cofactor**: Return the inverse of a matrix using cofactors
- **solve_cramer**: Solve the system Ax = b using the Cramer alghoritm
- **eigenvectors**: Compute the eigenvectors of a matrix given its eigenvalues
- **matrix_power**: Compute the power of a matrix A using eigenvalues
- **matrix_exponential**: Compute the exponential of a matrix A using eigenvalues


### Note on eigenvalues

In a separate script, I attempted to develop a function that calculates the eigenvalues using the QR factorization, where Q is the Gram-Schimdt decomposition of A. However, because it deals with complex numbers and other large problems, this is one of the most complex algorithms ever created.
I kept it separate since the version I created only works on specific matrices. I'll endeavour to finish it, and I'll keep the repository updated with any new developments.
