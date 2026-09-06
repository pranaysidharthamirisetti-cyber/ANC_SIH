import torch
from config import *
from models.crn import ComplexCRN

def main():
    ckpt=torch.load(CHECKPOINT_PATH,map_location="cpu")
    n_fft = ckpt.get("n_fft", N_FFT)
    freq_bins = ckpt.get("freq_bins", n_fft // 2 + 1)
    model=ComplexCRN(ckpt.get("hidden",HIDDEN), freq_bins=freq_bins)
    state=ckpt["model_state"]
    # Extract the CRN weights from SpeechEnhancer.net
    state={k.replace("net.","",1):v for k,v in state.items() if k.startswith("net.")}
    model.load_state_dict(state)
    model.eval()

    dummy=torch.randn(1,2,freq_bins,126)
    out=ROOT/"deployment"/"crn_mask.onnx"
    torch.onnx.export(
        model,dummy,out,opset_version=17,
        input_names=["stft_real_imag"],output_names=["complex_mask"],
        dynamic_axes={"stft_real_imag":{0:"batch",3:"time"},
                      "complex_mask":{0:"batch",3:"time"}}
    )
    print("Exported:",out)

if __name__=="__main__":
    main()
