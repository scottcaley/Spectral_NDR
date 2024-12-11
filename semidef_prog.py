import cvxpy as cp
import numpy as np
from sklearn.neighbors import KDTree


#KNN by KD tree, since it's faster in predicting.

# The notations are the same as in the paper, where p is the dimension of x_i's, and m that of y_i's, and n is the number of x_i's.
m,p,n=10,20,500
#X, the original data
X = np.random.randint(low=-100,high=100, size=(n, p))
# Build KD tree
kdtree = KDTree(X)

# eta is the matrix of eta_{ij}'s
eta=[[0 for _1 in range(n)] for _2 in range(n)]
for i in range(n):
    distance,indices = kdtree.query([X[i,:]], k=5)
    for k in indices[0]:
        eta[i][k]=1
    
#Y, the output data from LLE
Y = np.random.randint(low=-100,high=100, size=(n, m))
print(Y)
C = np.random.randn(n, n)


# Define and solve the CVXPY problem.
# Create a symmetric matrix variable.
P = cp.Variable((m,m), symmetric=True)
# creating the objective function
objective=0
for i in range(n):
    #setup s_i
    numer=0
    denom=0
    for j1 in range(n):
        for j2 in range(n):
            if j1!=j2:
                vecY=Y[j1,:]-Y[j2,:]
                vecX=X[j1,:]-X[j2,:]
                numer+=2*eta[i][j1]*eta[i][j2]*cp.quad_form(vecY, P)*(np.linalg.norm(vecX)**2)
                denom+=2*eta[i][j1]*eta[i][j2]*(np.linalg.norm(vecX)**4)
    s=numer/denom
    for j1 in range(n):
        for j2 in range(n):
            if j1!=j2:
                vecY=Y[j1,:]-Y[j2,:]
                vecX=X[j1,:]-X[j2,:]
                expression=eta[i][j1]*eta[i][j2]*(cp.quad_form(vecY, P)-s*np.linalg.norm(vecX)**2)**2
                objective+=expression
# The operator >> denotes matrix inequality.
constraints = [P >> 0]
constraints += [cp.trace(P) == 1]
prob = cp.Problem(objective, constraints)
prob.solve()
print("The optimal value is", prob.value)
print("A solution P is")
print(P.value)