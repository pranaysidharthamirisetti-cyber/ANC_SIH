import numpy as np

class NLMSFilter:
    def __init__(self,taps=128,step=0.3,eps=1e-8):
        self.taps=taps
        self.step=step
        self.eps=eps
        self.w=np.zeros(taps,dtype=np.float32)
        self.x=np.zeros(taps,dtype=np.float32)

    def process(self,primary,reference):
        L=min(len(primary),len(reference))
        out=np.zeros(L,dtype=np.float32)
        for n in range(L):
            self.x[1:]=self.x[:-1]
            self.x[0]=reference[n]
            y=np.dot(self.w,self.x)
            e=primary[n]-y
            out[n]=e
            p=np.dot(self.x,self.x)+self.eps
            self.w += (self.step/p)*e*self.x
        return out
