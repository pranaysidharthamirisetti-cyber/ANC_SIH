"""Hardware-specific real-time integration skeleton.

Use a real audio stream/callback for the selected USB/I2S audio interface.
Do not claim real-time performance until end-to-end latency is measured.
"""

import argparse

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/best.pt")
    a=p.parse_args()
    print("Checkpoint:",a.checkpoint)
    print("Connect your selected audio interface to a streaming callback:")
    print("audio frame -> STFT -> model -> iSTFT -> output.")
    print("Then add the reference microphone -> NLMS path and benchmark latency.")

if __name__=="__main__":
    main()
