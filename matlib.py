# -*- coding: utf-8 -*-
"""
file      matlib.py
author  Ernesto P. Adorio
          UPDEPP (UP Clarkfield)
          ernesto.adorio@gmail.com
revisions
    Ver 0.0.1 initial release
               2009.01.16 added matdots,matrandom, isiterable
    Ver 0.0.2  2009.10.12 added vec2colmat,vec2rowmat,  mat2vec, matSelectCols,
                          matDelCols, matInsertConstCol,matreadstring
    Ver 0.0.3  2009.12.22-25 added matSubmat, matRows, matCols,
                          matmatt, matxtx, matvectmatvec,matsplitbycolumn
                          revised matvec
"""
def matreadstring(s, sep = " "):
    """
    Each line must be complete row.
    comment lines start with a #.
    version 0.0.1
    no varnames yet!
    """
    M = []
    for line in s.split("\n"):
       if line:
          if not line.startswith("#"):
             M.append([float(x) for x in (line.split(sep))])
    return M
def matsplitbycolumn(X, col=-1):
    """
    Splits a matrix X by column.
    """
    if col == -1 or col == len(X[0])-1:
       # X2 will be a vector not a matrix.
       X1 = [x[:-1] for x in X]
       X2 = [x[-1] for x in X]
    else:
       #X2 is a matrix.
       X1 = [x[:col] for x in X]
       X2 = [X[col:-1] for x in X]
    return X1, X2
def tabular(hformat, headers, bformat, M):
    # added nov.29, 2008.
    # prints the table data.
    nrows = len(M)
    ncols = len(M[0])
    # print headers.
    for j, heading in enumerate(headers):
        print hformat[j] % heading,
    print
    # print the body.
    for i, row in enumerate(M):
        for j, col in enumerate(M[i]):
            print bformat[j] % M[i][j],
        print
    print
def vecadd(X,Y):
    n = len(X)
    if n != len(Y):
       return None
    return [x + y for x,y in zip(X,Y)]
def vecsub(X, Y):
    n = len(X)
    if n != len(Y):
       raise ArgumentError, "incompatible vectors in vecsub."
    return [x - y for x,y in zip(X,Y)]
def eye(m, n= None):
    if n is None:
        n = m
    B= [[0]* n for i in range(m)]
    for i in range(m):
        B[i][i] = 1.0
    return B
matiden = eye
def vec2colmat(X):
    """
    Retuns a 1 column matrix out of array or vector X.
    """
    return ([[x] for x in X])
def vec2rowmat(X):
    """
    Retuns a 1 row matrix out of array or vector X.
    """
    return ([[x for x in X]])
def mat2vec(M, column=0):
    """
    Returns a vector from a column of matrix M.
    """
    try:
      M[0][column]
      return [m[column] for m in M]
    except:
      return M
def matvectmatvec(x,M):
    """
    Computes x^t M x where x is a column vector.
    Result is a scalar.
    """
    return dot(x,matvec(M,x))
def matSelectCols(M, jindices):
    """
    Extracts a submatrix from M with col indices in jindices.
    Negative indices are properly handled too by Python, no need for
    adjustment.
    """
    N = []
    n = len(M)
    for i in range(n):
        N.append([M[i][j] for j in jindices])
    return N
matCols = matSelectCols
def matRows(M, iindices):
    """
    Returns selected rews of M as a matrix.
    """
    return [M[i:] for i in iindices]
def matDelCols(M, colindices):
    """
    M - input matrix
    colindices = array indices of columns to delete.
    Returns matrix from M with columns deleted.
    """
    ncols   = len(M[0])
    nindices = len(colindices)
    # Adjust for negative indices.
    for j in range(nindices):
        if colindices[j] < 0:
           colindices[j] += ncols
    jindices = []
    for j in range(ncols):
        if j not in colindices:
           jindices.append(j)
    return matSelectCols(M, jindices)
def matSubmat(M, rindices, jindices):
    # Returns a submatrix selected from M.
    N = []
    for i in rindices:
        N.append([M[i][j] for j in jindices])
    return N
def matInsertConstCol(X, column, c = 1.0, inplace= True):
    """
    Inserts a constant column to vector or matrix X at position column.
    NEVER forget that indicing starts at zero.
    """
    if not inplace:
       Xcopy = [x[:] for x in X]
    else:
       Xcopy = X
    for i in range(len(X)):
       Xcopy[i].insert(column, c)
    return Xcopy
def matxtx(X):
    # returns the matrix of the coefficients of the
    # normal equations for least squares computation.
    # This should be more efficient than calling a
    # matrix multiplication on t(X) and X.
    # same as function call mattmat(X, X)
    m = len(X)
    n = len(X[0])
    M = [[0.0]* n for i in range(n)]
    for i in range(n):
        for j in range(i, n):
            dot = 0.0
            for r  in range(m):
                dot += X[r][i] * X[r][j]
            M[i][j] = dot
            if i != j:
               M[j][i] = dot
    return M
def matzero(m, n = None):
    """
    Returns an m by n zero matrix.
    """
    if n is None:
        n = m
    return [[0]* n for i in range(m)]
def matdiag(D):
    """
    Returns a diagonal matrix with diagonal elements
    from D.
    """
    n = len(D)
    A = [[0] * n for i in range(n)]
    for i in range(n):
        A[i][i] = D[i]
    return A
def diag(A):
    """
    Returns diagonal elements of A as a matrix
    with 1 column
    """
    return [[A[i][i]] for i in range(len(A))]
    # Here is a version which returns a vector.
    # return A[i][i] for i in rang(len(A))
def matcol(X, j):
    # Returns the jth column of matrix X.
    nrows = len(X)
    return [X[i][j] for i in range(nrows)]
def trace(A):
    """
    Returns the trace of a matrix.
    """
    return sum([A[i][i] for i in range(len(A))])
def matadd(A, B):
    """
    Returns C = A + B.
    """
    try:
        m = len(A)
        if m != len(B):
            return None
        n = len(A[0])
        if n != len(B[0]):
            return None
        C = matzero(m, n)
        for i in range(m):
            for j in range(n):
                C[i][j] = A[i][j] + B[i][j]
        return C
    except:
        return None
def matsub(A, B):
    """
    returns C = A - B.
    """
    try:
        m = len(A)
        if m != len(B):
            return None
        n = len(A[0])
        if n != len(B[0]):
            return None
        C = matzero(m, n)
        for i in range(m):
            for j in range(n):
                C[i][j] = A[i][j] - B[i][j]
        return C
    except:
        return None
def matcopy(A):
    B = []
    for a in A:
       B.append(a[:])
    return B
def matkmul(A, k):
    """
    Multiplies each element of A by k.
    """
    B = matcopy(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            B[i][j] *= k
    return B
def transpose(A):
    """
    Returns the transpose of A.
    """
    m,n = matdim(A)
    At = [[0] * m for j in range(n)]
    for i in range(m):
        for j in range(n):
            At[j][i] = A[i][j]
    m,n = matdim(At)
    return At
matt = transpose
mattrans = transpose
def matdim(A):
    # Returns the number of rows and columns of A.
    if hasattr(A, "__len__"):
       m = len(A)
       if hasattr(A[0], "__len__"):
          n = len(A[0])
       else:
          n = 0
    else:
       m = 0  # not a matrix!
       n = 0
    return (m, n)
def matprod(A, B):
    """
    Computes the product of two matrices.
    2009.01.16 Revised for matrix or vector B.
    A and B are matrices. If one of them is a vector,
    it must be transformed into a matrix with one row
    or one column.
    """
    m, n = matdim(A)
    p, q = matdim(B)
    if n!= p:
       return None
    try:
       if iter(B[0]):
          q = len(B[0])
    except:
       q = 1
    C = matzero(m, q)
    for i in range(m):
        for j in range(q):
            t = sum([A[i][k] * B[k][j] for k in range(p)])
            C[i][j] = t
    return C
matmul = matprod
def matvec(A, y):
    """
    Returns the product of matrix A with vector y.
    Revision:
       dec. 22, 2009: this version should work with one column matrices y.
    """
    m = len(A)
    n = len(A[0])
    try:
      y[0][0]
      out = [0] * m
      for i in range(m):
        for j in range(n):
            out[i] += A[i][j] * y[j][0]
      return out
    except:
      out = [0] * m
      for i in range(m):
        for j in range(n):
            out[i] += A[i][j] * y[j]
      return out
def mattvec(A, y):
    """
    Returns the vector A^t y.
    """
    At = transpose(A)
    return matvec(At, y)
def dot(X, Y):
    """
    Dot product of vectors X and Y.
    """
    return sum(x* y for (x,y) in zip(X,Y))
def matdots(X):
    # Added Jan 16, 2009.
    # Returns the matrix of dot products of the column vectors
    # This is the same as X^t X.
    (nrow, ncol) = matdim(X)
    M = [[0.0] * ncol for i in range(ncol)]
    for i in range(ncol):
        for j in range(i+1):
            dot = sum([X[p][i]* X[p][j] for p in range(ncol)])
            M[i][j] = dot
            if i != j:
               M[j][i] = M[i][j]
    return M
def mattmat(A, B):
    """
    Returns the product [transpose(A) B]
    if B = A, use matxtx instead.
    """
    AtB = matprod(transpose(A), B)
    return AtB
def matmatt(A,B):
    """
    Returns A B^t
    added dec,22,2009
    """
    return matmul (A, mattrans(B))
def matrandom(nrow, ncol = None):
    # Added Jan. 16, 2009
    if ncol is None:
       ncol = nrow
    R = []
    for i in range(nrow):
        R.append([random.random() for j in range(ncol)])
    return R
def matunitize(X, inplace = False):
    # Added jan. 16, 2009
    # Transforms each vector in X to have unit length.
    if not inplace:
       V = [x[:] for x in X]
    else:
       V = X
    nrow = len(X)
    ncol = len(X[0])
    for j in range(ncol):
        recipnorm = sum([X[j][j]**2 for j in range(ncol)])
        for i in range(nrow):
            V[i][j] *= recipnorm
    return V
def matprint(A,format= "%8.4f"):
    #prints the matrix A using format
    if hasattr(A, "__len__"):
      for i,row in enumerate(A):
        try:
          if iter(row):
             for c in row:
               print format % c,
             print
        except:
           print row
    else:
        print "Not a matrix!"
    print # prints a blank line after matrix
def mataugprint(A,Y, format= "%8.4f"):
    #prints the augmented matrix A|Y using format
    try:
        ycols = len(Y[0])
    except:
        ycols = 1
    for i,row in enumerate(A):
        for c in row:
           print format % c,
        print "|",
        if ycols == 1:
           print format % Y[i]
        else:
           for y in Y[i]:
               print format % Y[i],
    print
def gjinv(AA,inplace = False):
    """
    Determines the inverse of a square matrix BB by Gauss-Jordan reduction.
    """
    n = len(AA)
    B = eye(n)
    if not inplace:
        A = [row[:] for row in AA]
    else:
        A = AA
    for i in range(n):
        #Divide the ith row by A[i][i]
        m = 1.0 / A[i][i]
        for j in range(i, n):
            A[i][j] *= m  # # this is the same as dividing by A[i][i]
        for j in range(n):
            B[i][j] *= m
        #lower triangular elements.
        for k in range(i+1, n):
            m = A[k][i]
            for j in range(i+1, n):
                A[k][j] -= m * A[i][j]
            for j in range(n):
                B[k][j] -= m * B[i][j]
        #upper triangular elements.
        for k in range(0, i):
            m = A[k][i]
            for j in range(i+1, n):
                A[k][j] -= m * A[i][j]
            for j in range(n):
                B[k][j] -= m * B[i][j]
    return B
matinverse = gjinv
matinv = gjinv
def Test():
    X = [1,1,1]
    print dot(X, X)
    AA = [[1,2,3],
          [4,5,8],
          [9,7,6]]
    BB = eye(3)
    print "Identity matrix eye(3):"
    matprint(BB)
    print "inputs:"
    print AA
    print "product"
    matprint(matprod(AA, AA))
    print "inverse of AA:"
    BB = gjinv(AA)
    matprint(BB)
    print "product of AA and its inverse:"
    matprint(matprod(AA ,BB))
if __name__ == "__main__":
    Test()