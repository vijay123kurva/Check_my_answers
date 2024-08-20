import tensorflow as tf
import transformers
from flask import Flask, render_template, request
import numpy as np
from huggingface_hub import from_pretrained_keras
import tensorflow_hub as hub

app = Flask(__name__)

class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data."""

    def __init__(
            self,
            sentence_pairs,
            labels,
            batch_size=32,
            shuffle=True,
            include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # Encode the sentence pairs.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=128,
            return_attention_mask=True,
            return_token_type_ids=True,
            padding='max_length',
            return_tensors="tf",
        )

        # Convert encoded features to numpy arrays.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Return features and labels if required.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

# Load the model globally
try:
    model = from_pretrained_keras("keras-io/bert-semantic-similarity")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define labels
labels = ["Contradiction", "Perfect", "Neutral"]

def check_similarity(sentence1, sentence2):
    if model is None:
        raise RuntimeError("Model not loaded")

    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )
    try:
        probs = model.predict(test_data[0])[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {}

    labels_probs = {labels[i]: float(probs[i]) for i, _ in enumerate(labels)}
    return labels_probs

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/checkMyAnswer")
def checkMyAnswer():
    return render_template("checkMyAnswer.html")

@app.route("/answers")
def answers():
    return render_template("answers.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        student_ans = request.form.get('student_ans')
        model_ans = request.form.get('model_ans')
        print(f"Student answer: {student_ans}")
        print(f"Model answer: {model_ans}")
        text = check_similarity(student_ans, model_ans)
        print(f"Prediction: {text}")

        if not text:
            return "Error in prediction", 500

        con_val = int(text.get("Contradiction", 0) * 100)
        per_val = int(text.get("Perfect", 0) * 100)
        neu_val = int(text.get("Neutral", 0) * 100)

        result_dict = {
            'Contradiction': con_val,
            'Perfect': per_val,
            'Neutral': neu_val,
            'student_ans': student_ans,
            'model_ans': model_ans,
            'marks': per_val
        }
        return render_template('answers.html', dict=result_dict)
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

if __name__ == '__main__':
    app.run()
