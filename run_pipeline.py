import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import preprocess, train, evaluate

if __name__ == "__main__":
    preprocess.main()
    train.main()
    evaluate.main()
    print("Pipeline complete. See outputs/ for results.")
