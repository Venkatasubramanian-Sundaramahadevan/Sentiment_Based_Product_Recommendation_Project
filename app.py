from flask import Flask, request, render_template
from model import SentimentRecommender

app = Flask(__name__)

sent_reco_model = SentimentRecommender()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the username as input
    user_name_input = request.form['username'].lower()
    sent_reco_output = sent_reco_model.top5_recommendations(user_name_input)

    if not (sent_reco_output is None):
        return render_template("index.html", output=sent_reco_output)
    else:
        return render_template("index.html",
                               message_display="User Name doesn't exist. Please provide a valid user!")


if __name__ == '__main__':
    app.run()
