from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import random
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model
import numpy as np
import pickle
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import mysql.connector

app = Flask(__name__)

CORS(app)


@app.route('/')
def hello_world():
    nltk.download()
    text = "hello world"
    text = nltk.word_tokenize(text)
    text = ', '.join(text)
    return text


@app.route('/train', methods=["POST"])
def train_data():
    lemmatizer = WordNetLemmatizer()

    words = []
    classes = []
    documents = []
    bigram_words = []

    ignore_words = [',', '|']

    intents = request.json.get('data')
    print(intents)

    stopWords = set(stopwords.words('english'))

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # print(pattern)
            w = nltk.word_tokenize(pattern)
            w = [word.replace("\\", "").replace("\"", "").replace(
                "\'", "").strip() for word in w if word.isalnum()]
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
    return "Model created"


@app.route('/disease', methods=['GET'])
@cross_origin(supports_credentials=True)
def open_mysql():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="workshop-2"
    )

    mycursor = mydb.cursor()

    # mycursor.execute("SELECT * FROM diseases where id=%i" % id)
    mycursor.execute("SELECT * FROM diseases")
    myresult = mycursor.fetchall()

    row_headers = [x[0] for x in mycursor.description]
    json_data = []
    for result in myresult:
        json_data.append(dict(zip(row_headers, result)))

    return jsonify(json_data)


@app.route('/get_response', methods=['GET','POST'])
def create_response():
    lemmatizer = WordNetLemmatizer()
    model = load_model('chatbot_model.h5')
    # Ubah Intents jadi intent dari php
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))

    detected_tags = []
    symptoms_list = []
    detected_rules = []

    # Symptoms, rules, dan diseases ambil dari database

    symptoms = [
        'Hard to chew',  # 1
        'Swelling  or inflammation  of the gums',  # 2
        'Shaky Teeth',  # 3
        'Swelling of the jaw',  # 4
        'Fever',  # 5
        'Swollen lymph nodes around jaw or neck',  # 6
        'Bad Breath',  # 7
        'Pain or tenderness around the gums',  # 8
        'Severe pain for several days after tooth extraction',  # 9
        'Bones seen in socket',  # 10
        'Teeth feel painful and sensitive',  # 11
        'Eroded tooth',  # 12
        'Headache',  # 13
        'Insomnia or feeling restless',  # 14
        'The sound of teeth crunching during sleep',  # 15
        'Gums bleed easily',  # 16
        'The shape of the gum is round',  # 17
        'The consistency of the gums becomes soft',  # 18
        'Gum or suppurating teeth',  # 19
        'Tooth aches or throbbing',  # 20
        'Redness on the corners of the mouth',  # 21
        'The corner of the mouth feel painful',  # 22
        'Scaly mouth corners',  # 23
        'Ulcer (wound in the corner of the mouth)',  # 24
        'Dentin Seen',  # 25
        'Cavity',  # 26
        'Infected pulp/inflammation of the pulp',  # 27
        'Throbbing pain without stimulation',  # 28
        'White spots on teeth',  # 29
        'White patches on tongue',  # 30
        'White patches on the oral cavity',  # 31
        'Plaque deposits',  # 32
        'There is Tartar',  # 33
        'Tooth decay',  # 34
        'Pulp is numb',  # 35
        'The pulp chamber is open',  # 36
        'Red gum'  # 37
    ]

    rules = [
        ['1', '2', '3'],  # Disease 1
        ['7', '1', '4', '8', '5', '6'],  # disease 2
        ['7', '9', '10'],
        ['11', '12'],
        ['11', '13', '14', '15'],
        ['2', '16', '17', '18'],
        ['2', '5', '6', '19', '20'],
        ['7', '1', '4', '8', '2'],
        ['21', '22', '23', '24'],
        ['26', '25', '11'],
        ['25', '26', '27', '28'],
        ['26', '29'],
        ['7', '30', '31'],
        ['7', '16', '32', '33'],
        ['26', '34', '35', '36'],
        ['7', '16', '2', '19', '37']
    ]

    diseases = [
        'Periodontal Abscess',
        'Peripical Abscess',
        'Alveolar Osteitis',
        'Dental Abrasion',
        'Bruxism',
        'Gingivitis',
        'Infected Teeth',
        'Pain at the rear teeth',
        'Angular Ceilitis',
        'Caries Media',
        'Caries Profunda',
        'Caries Superficial',
        'Candidiasis',
        'Calculus (dental)',
        'Pulp Necrosis',
        'Periodontitis',
    ]

    symptoms = [" ".join(x.lower().split()) for x in symptoms]
    diseases = [" ".join(x.lower().split()) for x in diseases]

    def clean_up_sentence(sentence):
        stopWords = set(stopwords.words('english'))
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [word for word in sentence_words if word.isalnum()]
        sentence_words = [lemmatizer.lemmatize(
            word.lower()) for word in sentence_words]

        filtered_words = []
        for w in sentence_words:
            if w not in stopWords:
                filtered_words.append(w)

        bigrm = nltk.bigrams(filtered_words)

        result = map(lambda x: ' '.join(x), list(bigrm))

        filtered_words.extend(list(result))
        print(filtered_words)
        return filtered_words

    def bow(sentence, words, show_details=True):
        sentence_words = clean_up_sentence(sentence)
        bags = []
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag = [0]*len(words)
                    bag[i] = 1
                    bags.append(bag)
                    if show_details:
                        print('found in bag: %s' % w)
        return(np.array(bags))

    def predict_class(sentence):
        bags = bow(sentence, words)
        if(len(bags) == 0):
            return 0
        results = []
        for bag in bags:
            res = model.predict(np.array([bag]))[0]
            ERROR_THRESHOLD = 0.25
            PROBABILITY_THRESHOLD = 0.8
            result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
            result.sort(key=lambda x: x[1], reverse=True)
            results.append(result)

        print("results", results)
        return_list = []
        for result in results:
            for r in result:
                if (r[1] > PROBABILITY_THRESHOLD):
                    return_list.append(
                        {"intent": classes[r[0]], "probability": str(r[1])})
        # print("r", r)
        # print("results", results)
        # print("res", res)
        # print("symptoms", symptoms)
        print("return list", return_list)
        return return_list

    def getPrediction(ints):
        print('ints', ints)
        data_intents = {}
        for intent in ints:
            tag = intent['intent'].lower()
            if(tag != ''):  # dump every 'tag' which determines which intents the user's input match into the list symptoms_list
                detected_tags.append(tag)
                if any(tag in symptom for symptom in symptoms):
                    if(tag not in symptoms_list):
                        symptoms_list.append(tag)
                        detected_rules.append(symptoms.index(tag) + 1)

                detected_diseases = []
                for detected_rule in detected_rules:
                    for rule in rules:
                        if str(detected_rule) in rule:
                            detected_diseases.append(
                                diseases[rules.index(rule)])
                            temp = list(set(detected_diseases))

                if 'temp' in locals():
                    detected_disease_probabilities = []
                    for index, disease in enumerate(temp, start=1):
                        disease_index = diseases.index(disease)
                        rules_list = rules[disease_index]
                        total_rules = len(rules_list)
                        matched_rules = 0
                        for rule in rules_list:
                            for detected_rule in detected_rules:
                                if str(detected_rule) in rule:
                                    matched_rules += 1
                        rule_probability = matched_rules / total_rules
                        # detected_disease_probabilities.append(
                        #     {'disease': disease, 'probability': rule_probability * detected_diseases.count(disease)/len(detected_diseases), 'index': index})  # Algorithm for Probability goes here
                        detected_disease_probabilities.append(
                            {'disease': disease, 'probability': rule_probability, 'index': diseases.index(disease)})  # Algorithm for Probability goes here
                    detected_disease_probabilities = sorted(
                        detected_disease_probabilities, key=lambda x: x['probability'], reverse=True)

            data_intents['detected_tags'] = detected_tags
            data_intents['symptoms_list'] = symptoms_list
            data_intents['detected_rules'] = detected_rules
            data_intents['detected_diseases'] = detected_diseases

        print("Tags: ", detected_tags)
        print("Symptoms: ", symptoms_list)
        print("Rules: ", detected_rules)
        print("Detected Disease: ", detected_diseases)

        if 'detected_disease_probabilities' in locals():
            data_intents['temp'] = temp
            data_intents['detected_disease_probabilities'] = detected_disease_probabilities
            print("Diseases: ", temp)
            print("Probabilities: ", detected_disease_probabilities)

        return data_intents

    def getSuggestion(ints):
        prediction = getPrediction(ints)
        index_highest_disease_probability = prediction['detected_disease_probabilities'][0]['index']
        for rule in rules[index_highest_disease_probability]:
            if (int(rule) not in detected_rules):
                result = "Mmm, based on my data, people that have your symptoms are also have " + symptoms[int(
                    rule) - 1] + ", do you also feel it? symptoms no: " + rule
                print(result)
                user_input = input().lower().strip()
                if 'yes' in user_input:
                    ints.append(predict_class(symptoms[int(
                        rule) - 1], model))
        result = "I have diagnosed your symptoms and I guess you are having "
        result += prediction['detected_disease_probabilities'][0]['disease'] + ". Probability : " + str(prediction[0]['probability'])
        return [result, prediction, True]

    def getResponse(ints):
        prediction = getPrediction(ints)
        list_of_intents = intents['intents']
        result = ints[-1]['intent']
        for i in list_of_intents:
            if(i['tag'] == result):
                if(len(symptoms_list) < 3):
                    result = random.choice(i['responses'])
                elif(result == "no_other_symptoms" or len(symptoms_list) >= 3):
                    index_highest_disease_probability = prediction[
                        'detected_disease_probabilities'][0]['index']
                    for rule in rules[index_highest_disease_probability]:
                        if(int(rule) not in detected_rules):
                            result = "Mmm, based on my data, people that have your symptoms are also have " + symptoms[int(
                                rule) - 1] + ", do you also feel it? symptoms no: " + rule
                            print(result)
                            user_input = input().lower().strip()
                            if 'yes' in user_input:
                                ints.append(predict_class(symptoms[int(
                                    rule) - 1], model))
                    result = "I have diagnosed your symptoms and I guess you are having "
                    result += prediction[
                        'detected_disease_probabilities'][0]['disease'] + ". Probability : " + str(prediction[
                            'detected_disease_probabilities'][0]['probability'])
                    return [result, prediction, True]
                    # for ds in detected_disease_probabilities:
                    #     if ds['probability'] >= 0.3:
                    #         result += ds['disease'] + ", "
        #         else:
        #             print('3')
        #             print(i['responses'])
        # response = random.choice(i['responses'])
        return [result, prediction, False]

    def chatbot_response(msg):
        ints = predict_class(msg)
        if(ints == 0):
            sorry = ['Sorry, I don\'t understand what you mean.',
                     'Can you type in understandable words?']
            return(random.choice(sorry))
        res = getResponse(ints)
        return(res)

    print("You can start interact with the chatbot now.")

    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="workshop-2"
    )

    mycursor = mydb.cursor()

    mycursor.execute(
        "SELECT * FROM `chats` LEFT JOIN chat_sessions ON chats.session = chat_sessions.session WHERE chats.sender = 'me' AND chat_sessions.session = '%s'" % request.form['session'])
    myresult = mycursor.fetchall()

    row_headers = [x[0] for x in mycursor.description]
    json_data = []
    for result in myresult:
        json_data.append(dict(zip(row_headers, result)))

    final_response = ""
    for data_db in json_data:
    # while True:
        user_input = data_db['message']
        # user_input = "Umm, I'm not feeling well today. I am having cavity on my teeth, and my dentin is seen and my pulp felt infected."
        print(user_input)
        user_input = user_input.lower().strip()
        # user_input = "i have bad breathe, fever, cannot sleep, headache"
        if(user_input != ""):
            print("You: ============================================================================>>>", user_input)
            response = chatbot_response(user_input)
            print("Bot: ============================================================================>>>", response)
            final_response = response
    
    return final_response


if __name__ == '__main__':
    app.run(debug=True)
