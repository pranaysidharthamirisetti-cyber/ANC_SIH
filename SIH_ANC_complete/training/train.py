import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from models.model import SpeechEnhancer
from training.dataset import PairedAudioDataset, collate_fn
from training.losses import complex_l1_loss
from training.validation import validate

def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = PairedAudioDataset(TRAIN_NOISY, TRAIN_CLEAN)
    val_ds = PairedAudioDataset(VAL_NOISY, VAL_CLEAN)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Generate the dataset first.")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = SpeechEnhancer(HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best = float("inf")

    for epoch in range(1, EPOCHS+1):
        model.train()
        running = 0.0
        for X, S in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            X, S = X.to(device), S.to(device)
            optimizer.zero_grad()
            Y, _ = model(X)
            loss = complex_l1_loss(Y, S)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item()

        train_loss = running/max(len(train_loader),1)
        val_loss = validate(model, val_loader, device)
        print(f"train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best:
            best = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "hidden": HIDDEN,
                "sample_rate": SAMPLE_RATE,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "win_length": WIN_LENGTH,
                "freq_bins": N_FFT // 2 + 1,
            }, CHECKPOINT_PATH)
            print("Saved:", CHECKPOINT_PATH)

if __name__ == "__main__":
    main()
