from flask import Flask, render_template, request

app = Flask(__name__)

#--------------------------------------
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load healthcare data from a JSON file
with open("healthcare_data.json", "r") as file:
    healthcare_data = json.load(file)

# Predefined list of symptoms and associated diseases
symptom_to_disease = healthcare_data.get("symptom_to_disease", {})

# Function to get a random disease for a given symptom
def get_random_disease(symptom):
    possible_diseases = symptom_to_disease.get(symptom, [])
    if possible_diseases:
        return random.choice(possible_diseases)
    else:
        return "Unknown"

# Start the conversation
print("Healthcare Chatbot: Hello! I'm here to help you with your health concerns.")

# Vectorize symptoms for machine learning
symptoms = list(symptom_to_disease.keys())
vectorizer = TfidfVectorizer()
symptom_vectors = vectorizer.fit_transform(symptoms)

while True:
    user_input = input("You: ").lower()

    if user_input == "exit":
        print("Healthcare Chatbot: Goodbye! Take care.")
        break

    # Calculate cosine similarity between user input and symptoms
    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, symptom_vectors)[0]

    # Find the most similar symptom
    most_similar_index = similarity_scores.argmax()
    most_similar_symptom = symptoms[most_similar_index]

    if similarity_scores[most_similar_index] > 0.9:  # Adjust the threshold as needed
        selected_symptom = most_similar_symptom
        disease = get_random_disease(selected_symptom)
        print(f"Healthcare Chatbot: Based on the symptom '{selected_symptom}', a possible disease is '{disease}'.")

        # Retrieve generic medicine and home remedy information
        disease_info = healthcare_data.get("disease_info", {}).get(disease, {})
        generic_medicine = disease_info.get("generic_medicine", [])
        home_remedy = disease_info.get("home_remedy", [])

        if generic_medicine:
            print(f"Generic Medicine(s): {', '.join(generic_medicine)}")
        else:
            print("No generic medicine information available.")

        if home_remedy:
            print(f"Home Remedy: {', '.join(home_remedy)}")
        else:
            print("No home remedy information available.")
    else:
        print("Healthcare Chatbot: I'm not sure how to respond to that.")
#--------------------------------------
@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input'].lower()
        # Your chatbot code to handle user_input here...
        # You can use print statements to send responses to the chat container in HTML.
        print(f"User: {user_input}")
        # Example response:
        print("Healthcare Chatbot: This is a sample response.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
