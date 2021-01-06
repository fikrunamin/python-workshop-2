from flask import Flask

import random
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)


@app.route('/')
def hello_world():
    text = "hello world"
    text = nltk.word_tokenize(text)
    text = ', '.join(text)
    return text


@app.route('/train')
def train_data():
    lemmatizer = WordNetLemmatizer()

    words = []
    classes = []
    documents = []
    bigram_words = []

    ignore_words = [',', '|']

    combined_intents = open('intents.json').read()

    # interactive_chat = open('interactive_chat_intents.json').read()
    # disease = open('disease_intents.json').read()
    # symptom = open('symptom_intents.json').read()

    # interactive_chat = json.load(interactive_chat)
    # disease = json.load(disease)
    # symptom = json.load(symptom)

    # intents =

    intents = json.loads(combined_intents)
    stopWords = set(stopwords.words('english'))

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # print(pattern)
            w = nltk.word_tokenize(pattern)
            w = [word for word in w if word.isalnum()]
            w = [lemmatizer.lemmatize(
                word.lower()) for word in w]
            filtered_words = []
            for wrd in w:
                if wrd not in stopWords:
                    filtered_words.append(wrd)
            w = filtered_words
            bw = nltk.bigrams(w)
            bw = map(lambda x: ' '.join(x), list(bw))
            w.extend(list(bw))
            words.extend(w)
            documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    print(words)
    print(classes)
    print(documents)

    words = [lemmatizer.lemmatize(w.lower())
             for w in words if w not in ignore_words]

    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))

    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique lemmatized words", words)

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    training = []
    output_empty = [0] * len(classes)
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(
            word.lower())for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    print("Training data created")

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(np.array(train_x), np.array(train_y),
                     epochs=300, batch_size=5, verbose=1)
    model.save('chatbot_model.h5', hist)

    print("model created")
    return "Model Created"


if __name__ == '__main__':
    app.run(debug=True)
