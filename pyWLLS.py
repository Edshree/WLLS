# ====================================================
# This code contains the weighted least
# squares method suitable for gamma spectrum
# analysis
#
# Author @ Aiyun Sun
# Email: say17@nuaa.edu.cn
# date 05/25/2021
# ====================================================
import numpy as np

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
class WLLS:
    def __init__(self,A,b):
        self.A = A
        self.b = b
        self.m,n = np.shape(A)
        self.w_mat = np.zeros((self.m, self.m))
    def solve(self):
        error_std = np.sqrt(self.b)
        for i in range(self.m):
            if error_std[i]!=0:
                self.w_mat[i,i]=1/error_std[i]
        # AW = np.dot(self.A.T,self.w_mat).T
        AW = np.dot(self.w_mat,self.A)
        bW = np.dot(self.b,self.w_mat)
        ana = np.linalg.lstsq(AW, bW)[0]
        return ana
    def solve_algebra(self):
        error_std = np.sqrt(self.b)
        for i in range(self.m):
            if error_std[i]!=0:
                self.w_mat[i,i]=1/error_std[i]
        WA = np.dot(self.w_mat,self.A)
        temp1 = np.linalg.inv(np.dot(WA.T,WA))
        temp2 = np.dot(temp1,WA.T)
        Wb = np.dot(self.b,self.w_mat)
        ana = np.dot(temp2,Wb)
        return ana
    def solve_bj(self):
        error_std = self.b
        for i in range(self.m):
            if error_std[i]!=0:
                self.w_mat[i,i]=1/error_std[i]
        WA = np.dot(self.w_mat,self.A)
        temp1 = np.linalg.inv(np.dot(WA.T,WA))
        temp2 = np.dot(temp1,WA.T)
        Wb = np.dot(self.b,self.w_mat)
        ana = np.dot(temp2,Wb)
        return ana

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
class WLLS2:
    def __init__(self,A,b):
        self.A = A
        self.b = b
        self.m,n = np.shape(A)
        self.w_mat = np.zeros((self.m, self.m))
    def solve(self):
        error_std = np.sqrt(self.b)
        for i in range(self.m):
            if error_std[i]!=0:
                self.w_mat[i,i]=1/error_std[i]
        AW = np.dot(self.A.T,self.w_mat).T
        bW = np.dot(self.b,self.w_mat)
        ana = np.linalg.lstsq(AW, bW)[0]
        return ana

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
class LLS:
    def __init__(self,A,b):
        self.A = A
        self.b = b
    def solve(self):
        ana = np.linalg.lstsq(self.A,self.b)[0]
        return ana
