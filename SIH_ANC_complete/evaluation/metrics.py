import numpy as np

def snr_db(clean,test,eps=1e-12):
    L=min(len(clean),len(test))
    clean,test=np.asarray(clean[:L], dtype=np.float64),np.asarray(test[:L], dtype=np.float64)
    noise=clean-test
    return 10*np.log10((np.sum(clean**2)+eps)/(np.sum(noise**2)+eps))

def si_snr_db(clean,test,eps=1e-8):
    L=min(len(clean),len(test))
    clean=np.array(clean[:L], dtype=np.float64, copy=True)
    test=np.array(test[:L], dtype=np.float64, copy=True)
    clean-=clean.mean()
    test-=test.mean()
    target=np.dot(test,clean)/(np.dot(clean,clean)+eps)*clean
    noise=test-target
    return 10*np.log10((np.sum(target**2)+eps)/(np.sum(noise**2)+eps))
