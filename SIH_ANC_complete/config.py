from pathlib import Path

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 128

HIDDEN = 64

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

LMS_TAPS = 128
LMS_STEP = 1e-4
NLMS_TAPS = 128
NLMS_STEP = 0.3
NLMS_EPS = 1e-8

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CLEAN_DIR = DATA / "clean_speech"
NOISE_DIR = DATA / "defence_noise"

TRAIN_NOISY = DATA / "generated_dataset/train/noisy"
TRAIN_CLEAN = DATA / "generated_dataset/train/clean"
VAL_NOISY = DATA / "generated_dataset/val/noisy"
VAL_CLEAN = DATA / "generated_dataset/val/clean"
TEST_NOISY = DATA / "generated_dataset/test/noisy"
TEST_CLEAN = DATA / "generated_dataset/test/clean"

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best.pt"
OUTPUT_DIR = ROOT / "output"
INPUT_DIR = ROOT / "input"
