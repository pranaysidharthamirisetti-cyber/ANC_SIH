import torch

def complex_l1_loss(Y, S):
    return torch.mean(torch.abs(Y.real-S.real) + torch.abs(Y.imag-S.imag))

def si_snr(est, ref, eps=1e-8):
    est = est-est.mean(dim=-1, keepdim=True)
    ref = ref-ref.mean(dim=-1, keepdim=True)
    proj = (torch.sum(est*ref, dim=-1, keepdim=True) /
            (torch.sum(ref*ref, dim=-1, keepdim=True)+eps))*ref
    noise = est-proj
    return 10*torch.log10(
        (torch.sum(proj*proj, dim=-1)+eps) /
        (torch.sum(noise*noise, dim=-1)+eps)
    )

def time_domain_si_snr_loss(est, ref):
    return -si_snr(est, ref).mean()
