import numpy as np
from scipy.sparse import csr_matrix
from datetime import datetime as dt
import time
from py_pcha.furthest_sum import furthest_sum
import torch
from torch.autograd import Variable



def SAA(X,L,beta, noc, I=None, U=None, delta=0, verbose=False, conv_crit=1E-6, maxiter=500):
    """Return archetypes of dataset.

    Note: Commonly data is formatted to have shape (examples, dimensions).
    This function takes input and returns output of the transposed shape,
    (dimensions, examples).

    Parameters
    ----------
    X : Tensor.2darray
        Data matrix in which to find archetypes

    noc : int
        Number of archetypes to find

    I : 1d-array
        Entries of X to use for dictionary in C (optional)

    U : 1d-array
        Entries of X to model in S (optional)


    Output
    ------
    XC : Tensor.2darray
        I x noc feature matrix (i.e. XC=X[:,I]*C forming the archetypes)

    S : Tensor.2darray
        noc x length(U) matrix, S>=0 |S_j|_1=1

    C : Tensor.2darray
        noc x length(U) matrix, S>=0 |S_j|_1=1

    SSE : float
        Sum of Squared Errors

    varexlp : float
        Percent variation explained by the model
    """
    def S_update(S, XCtX, CtXtXC, muS, SST, SSE, niter,beta,L):
        """Update S for one iteration of the algorithm."""
        noc, J = S.shape
        e = torch.ones((noc, 1)).double().cuda()
        for k in range(niter):
            SSE_old = SSE

            g = (torch.matmul(CtXtXC, S) - XCtX +beta*torch.matmul(S, L))/ (SST / J)   # change this line 
            
            g = g - e * torch.sum(g * S, axis=0)

            S_old = S
            
            line_search_it=0 
            while True:

                S = (S_old - g * muS).clip(min=0)
            
                S = S / torch.matmul(e, torch.sum(S, axis=0).view(1,S.shape[1]))    
                SSt = torch.matmul(S , S.T)

                SSE= SST - 2 * torch.sum(XCtX * S) + torch.sum(CtXtXC * SSt) + beta*torch.sum( torch.matmul(S, L)* S)


                if SSE <= SSE_old * (1 + 1e-9):
                    muS = muS * 1.2
                    break
                else:
                    muS = muS / 2

                if(line_search_it>=100):  
                    break  
                line_search_it+=1 

        return S, SSE, muS, SSt

    def C_update(beta,S,L,X, XSt, XC, SSt, C, delta, muC, mualpha, SST, SSE, niter=1):
        """Update C for one iteration of the algorithm."""
        J, nos = C.shape

        if delta != 0:
            alphaC =torch.sum(C, axis=0)
            C = torch.matmul(C, torch.diag(1 / alphaC))

        e = torch.ones((J, 1)).double().cuda();
        XtXSt = torch.matmul(X.T, XSt)
        
        for k in range(niter):

            # Update C
            SSE_old = SSE
            g = (torch.matmul(X.T, torch.matmul(XC, SSt)) - XtXSt) / SST

            if delta != 0:
                g = torch.matmul(g, torch.diag(alphaC))
                
            g = g - e * torch.sum(g* C, axis=0)

            C_old = C


            line_search_it=0
            while True:
                C = (C_old - muC * g).clip(min=0)

                nC = torch.sum(C, axis=0) + 1e-16 
                
                C = torch.matmul(C, torch.diag(1 / nC))
                
                if delta != 0:
                    Ct = torch.matmul(C,torch.diag(alphaC) ) 
                else:
                    Ct = C

                XC = torch.matmul(X, Ct)

                CtXtXC = torch.matmul(XC.T, XC)

                SSE = SST - 2 * torch.sum(XC * XSt) + torch.sum(CtXtXC * SSt) + beta*torch.sum( torch.matmul(S, L)* S) # change this line 
            
                if SSE <= SSE_old * (1 + 1e-9):
                    muC = muC * 1.2
                    break
                else:
                    muC = muC / 2

                if(line_search_it>=100): 
                    break
                line_search_it+=1  


            # Update alphaC
            SSE_old = SSE

            if delta != 0:
                
                g = (torch.diag(CtXtXC * SSt).T / alphaC - torch.sum(C * XtXSt)) / (SST * J)
                
                alphaC_old = alphaC
                line_search_it=0
                while True:
                    alphaC = alphaC_old - mualpha * g
                    alphaC[alphaC < 1 - delta] = 1 - delta
                    alphaC[alphaC > 1 + delta] = 1 + delta
                    
                    XCt = torch.matmul(XC, torch.diag(alphaC / alphaC_old)) #np.dot(XC, np.diag(alphaC / alphaC_old))
                    CtXtXC = torch.matmul(XCt.T, XCt)
                    
                    SSE = SST - 2 * torch.sum(XCt * XSt) + torch.sum(CtXtXC * SSt) + beta*torch.sum( torch.matmul(S, L)* S)# change this line 
                    
                    if SSE <= SSE_old * (1 + 1e-9):
                        mualpha = mualpha * 1.2
                        XC = XCt
                        break
                    else:
                        mualpha = mualpha / 2

                    if(line_search_it>=100): ## added by CHF 
                        break ## added by CHF 
                    line_search_it+=1 ## added by CHF 
        if delta != 0:
            C = torch.matmul(C,torch.diag(alphaC) ) #C * np.diag(alphaC)

        #import pdb;pdb.set_trace();
        return C, SSE, muC, mualpha, CtXtXC, XC

    N, M = X.shape
    

    if I is None:
        I = range(M)
    if U is None:
        U = range(M)


    SST = torch.sum(X[:, U] * X[:, U])

    try:
       i = furthest_sum(X[:, I], noc, [int(np.ceil(len(I) * np.random.rand()))])
    except IndexError:
       class InitializationException(Exception): pass
       raise InitializationException("Initialization does not converge. Too few examples in dataset.")

    

    j = range(noc)
    C = csr_matrix((np.ones(len(i)), (i, j)), shape=(len(I), noc)).todense()
    C= torch.tensor(C).cuda()

    XC = torch.matmul(X[:, I], C)


    muS, muC, mualpha = 1, 1, 1 

    # Initialise S
    XCtX = torch.matmul(XC.T, X[:, U]) # C^T X^T X
    CtXtXC = torch.matmul(XC.T, XC) # C^T X^T X C
    
    S = -np.log(np.random.random((noc, len(U))))
    S = S / np.dot(np.ones((noc, 1)), np.mat(np.sum(S, axis=0)))
    S= torch.tensor(S).cuda()


    SSt = torch.matmul(S, S.T)

    SSE = SST - 2 * torch.sum(XCtX * S) + torch.sum(CtXtXC * SSt) + beta*torch.sum( torch.matmul(S, L)* S)  # ||X-XCS||_F   # change this line
    
    S, SSE, muS, SSt = S_update(S, XCtX, CtXtXC, muS, SST, SSE, 25,beta,L)

    # Set SAA parameters
    iter_ = 0
    dSSE = np.inf
    t1 = dt.now()

    varexpl= (SST - (SST - 2 * torch.sum(XCtX * S) + torch.sum(CtXtXC * SSt)))/SST


    if verbose:
        print('\nSupervised Archetypal Analysis')
        print('A ' + str(noc) + ' component model will be fitted')
        print('To stop algorithm press control C\n')

    dheader = '%10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s' % ('Iteration', 'Expl. var.', 'Cost func.', 'Delta SSEf.', 'muC', 'mualpha', 'muS', ' Time(s)   ')
    dline = '-----------+------------+------------+-------------+------------+------------+------------+------------+'


    while np.abs(dSSE) >= conv_crit * np.abs(SSE.cpu()) and iter_ < maxiter: #and varexpl.cpu() < 0.9999:
        if verbose and iter_ % 100 == 0:
            print(dline)
            print(dheader)
            print(dline)
        told = t1
        iter_ += 1
        SSE_old = SSE

        # C (and alpha) update

        XSt = torch.matmul(X[:, U], S.T)
        C, SSE, muC, mualpha, CtXtXC, XC = C_update(beta,S,L,X[:, I], XSt, XC, SSt, C, delta, muC, mualpha, SST, SSE, 10)

        # S update
        XCtX = torch.matmul(XC.T, X[:, U])

        S, SSE, muS, SSt = S_update(
            S, XCtX, CtXtXC, muS, SST, SSE, 10,beta,L
        )


        # Evaluate and display iteration
        dSSE = (SSE_old - SSE).cpu()
        t1 = dt.now()

        if iter_ % 1 == 0:
            time.sleep(0.000001)
            varexpl = (SST- (SST - 2 * torch.sum(XCtX * S) + torch.sum(CtXtXC * SSt)))/SST #(SST - SSE) / SST
            if verbose:
                print('%10.0f | %10.4f | %10.4e | %10.4e | %10.4e | %10.4e | %10.4e | %10.4f \n' % (iter_, varexpl.cpu(), SSE.cpu(), dSSE/np.abs(SSE.cpu()), muC, mualpha, muS, (t1-told).seconds))

    varexpl= (SST - (SST - 2 * torch.sum(XCtX * S) + torch.sum(CtXtXC * SSt)))/SST
    if verbose:

        print(dline)
        print(dline)
        print('%10.0f | %10.4f | %10.4e | %10.4e | %10.4e | %10.4e | %10.4e | %10.4f \n' % (iter_, varexpl.cpu(), SSE.cpu(), dSSE/np.abs(SSE.cpu()), muC, mualpha, muS, (t1-told).seconds))
    
    ind, vals = zip(
        *sorted(enumerate(torch.sum(S, axis=1)), key=lambda x: x[0], reverse=1)
    )

    S = S[ind, :]
    C = C[:, ind]
    XC = XC[:, ind]

    return XC, S, C, SSE, varexpl