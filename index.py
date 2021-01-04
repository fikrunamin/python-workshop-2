from flask import Flask
import nltk

app = Flask(__name__)


@app.route('/')
def hello_world():
    text = "Hello World"
    text = nltk.word_tokenize(text)
    text = ', '.join(text)
    return text


if __name__ == '__main__':
    app.run()
