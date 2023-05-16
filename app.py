import tkinter as tk
import json
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Charger le tokenizer à partir du fichier
with open('tokenizer.json', 'r') as file:
    tokenizer_json = file.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# Charger le mapping des catégories à partir du fichier
with open('category_mapping.json', 'r') as file:
    category_mapping_json = file.read()
category_mapping = json.loads(category_mapping_json)

# Charger le modèle à partir du fichier
model = load_model('chatbot_model.h5')

# Fonction pour obtenir une réponse à partir d'une question
def get_response(question):
    # Transformer la question en séquence numérique
    question_sequence = tokenizer.texts_to_sequences([question])
    padded_question_sequence = pad_sequences(question_sequence, maxlen=max_sequence_length, padding='post')
    
    # Faire une prédiction avec le modèle
    prediction = model.predict(padded_question_sequence)[0]
    
    # Trouver la catégorie prédite
    predicted_category_index = np.argmax(prediction)
    predicted_category = list(category_mapping.keys())[list(category_mapping.values()).index(predicted_category_index)]
    
    # Trouver une réponse aléatoire dans la catégorie prédite
    category_responses = [reponse for i, reponse in enumerate(reponses) if categories[i] == predicted_category]
    return np.random.choice(category_responses)

# Fonction pour afficher la réponse dans la fenêtre
def display_response():
    question = question_entry.get()
    response = get_response(question)
    response_text.config(text=response)

# Création de la fenêtre
window = tk.Tk()
window.title("Chatbot")

# Label pour la question
question_label = tk.Label(window, text="Question:")
question_label.pack()

# Entrée pour la question
question_entry = tk.Entry(window, width=50)
question_entry.pack()

# Bouton pour obtenir la réponse
response_button = tk.Button(window, text="Obtenir une réponse", command=display_response)
response_button.pack()

# Label pour la réponse
response_label = tk.Label(window, text="Réponse:")
response_label.pack()

# Label pour afficher la réponse
response_text = tk.Label(window, text="")
response_text.pack()

# Variables pour les données
questions = []
reponses = []
categories = []

# Charger les données JSON à partir du fichier
with open('data.json', 'r') as file:
    data = json.load(file)

# Extraction des questions, réponses et catégories
for entry in data['data']:
    questions.append(entry['question'])
    reponses.append(entry['reponse'])
    categories.append(entry['categorie'])

# Padding des séquences pour qu'elles aient la même longueur
max_sequence_length = max(len(seq) for seq in tokenizer.texts_to_sequences(questions + reponses))

# Lancer la fenêtre
window.mainloop()
