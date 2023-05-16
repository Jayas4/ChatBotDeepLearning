import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Charger les données JSON à partir du fichier
with open('data.json', 'r') as file:
    data = json.load(file)
    
# Extraction des questions, réponses et catégories
questions = []
reponses = []
categories = []
for entry in data['data']:
    questions.append(entry['question'])
    reponses.append(entry['reponse'])
    categories.append(entry['categorie'])

# Tokenization des questions et réponses
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + reponses)

# Nombre total de mots dans le vocabulaire
vocab_size = len(tokenizer.word_index) + 1

# Transformation des questions en séquences numériques
questions_sequences = tokenizer.texts_to_sequences(questions)

# Transformation des réponses en séquences numériques
reponses_sequences = tokenizer.texts_to_sequences(reponses)

# Padding des séquences pour qu'elles aient la même longueur
max_sequence_length = max(len(seq) for seq in questions_sequences + reponses_sequences)
padded_questions_sequences = pad_sequences(questions_sequences, maxlen=max_sequence_length, padding='post')
padded_reponses_sequences = pad_sequences(reponses_sequences, maxlen=max_sequence_length, padding='post')

# Transformation des catégories en vecteurs one-hot
num_categories = len(set(categories))
category_mapping = {category: i for i, category in enumerate(set(categories))}
category_labels = np.array([category_mapping[category] for category in categories])
category_labels = np.eye(num_categories)[category_labels]

# Création du modèle
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_sequence_length))
model.add(LSTM(100))
model.add(Dense(num_categories, activation='softmax'))

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
model.fit(padded_questions_sequences, category_labels, epochs=10, batch_size=16)

# Sauvegarde du modèle
model.save('chatbot_model.h5')

# Sauvegarde du tokenizer
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as file:
    file.write(tokenizer_json)

# Sauvegarde du mapping des catégories
category_mapping_json = json.dumps(category_mapping)
with open('category_mapping.json', 'w') as file:
    file.write(category_mapping_json)
