import numpy as np

def snr_db(clean,test,eps=1e-12):
    L=min(len(clean),len(test))
    clean,test=clean[:L],test[:L]
    noise=clean-test
    return 10*np.log10((np.sum(clean**2)+eps)/(np.sum(noise**2)+eps))

def si_snr_db(clean,test,eps=1e-8):
    L=min(len(clean),len(test))
    clean,test=clean[:L],test[:L]
    clean-=clean.mean()
    test-=test.mean()
    target=np.dot(test,clean)/(np.dot(clean,clean)+eps)*clean
    noise=test-target
    return 10*np.log10((np.sum(target**2)+eps)/(np.sum(noise**2)+eps))
