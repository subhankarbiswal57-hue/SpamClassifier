import sys, os

# Ensure src is in sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from predict import predict_message

def main():
    print("=" * 60)
    print("      SMS / TEXT SPAM CLASSIFIER - INTERACTIVE CLI")
    print("=" * 60)
    print("Type a message to classify as 'Spam' or 'Not Spam'.")
    print("Type 'examples' to run built-in test cases.")
    print("Type 'quit' or 'exit' to exit.")
    print("-" * 60)
    
    while True:
        try:
            msg = input("\nEnter message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break
            
        if not msg:
            continue
            
        if msg.lower() in ["quit", "exit", "q"]:
            print("Exiting Spam Classifier CLI. Goodbye!")
            break
            
        if msg.lower() == "examples":
            test_cases = [
                ("WINNER!! You have won a 1,000 cash prize! Call 09050000321 now to claim your code. Valid 12hrs only.", "Spam"),
                ("Hey Mom, are we still meeting for lunch at 1pm tomorrow?", "Not Spam"),
                ("URGENT! Your mobile number won 5000 pounds bonus. Claim immediately by texting YES.", "Spam"),
                ("Can you send me the lecture notes for CS101 when you get home?", "Not Spam"),
                ("Congratulations! Urgent alert: your loan is approved. Click link to verify.", "Spam")
            ]
            print("\n--- Running Built-in Example Predictions ---")
            for text, expected in test_cases:
                pred, prob = predict_message(text)
                status = "[PASS]" if pred == expected else "[FAIL]"
                print(f"Message : \"{text}\"")
                print(f"  -> Prediction: {pred} (Confidence: {prob:.2%}) | Expected: {expected} {status}\n")
            continue
            
        label, prob = predict_message(msg)
        print(f"-> Prediction: {label} (Confidence: {prob:.2%})")

if __name__ == "__main__":
    main()

