import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_message

if __name__ == "__main__":
    while True:
        msg = input("Enter message (or 'quit'): ")
        if msg.lower() == "quit":
            break
        label, prob = predict_message(msg)
        print(f"-> {label} (confidence: {prob})")
