import os, sys

# Ensure src directory is in sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import preprocess
import train
import evaluate

def run():
    print("=" * 60)
    print("STEP 1: PREPROCESSING RAW DATA...")
    print("=" * 60)
    preprocess.main()
    
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING MODELS (NAIVE BAYES & LOGISTIC REGRESSION)...")
    print("=" * 60)
    train.main()
    
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATING MODELS & GENERATING METRICS/PLOTS...")
    print("=" * 60)
    evaluate.main()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE! ALL ARTIFACTS GENERATED SUCCESSFULLY.")
    print("=" * 60)

if __name__ == "__main__":
    run()

