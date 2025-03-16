import json
import random
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
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
    return random.choice(possible_diseases) if possible_diseases else "Unknown"

# Vectorize symptoms for machine learning
symptoms = list(symptom_to_disease.keys())
vectorizer = TfidfVectorizer()
symptom_vectors = vectorizer.fit_transform(symptoms)

# Define command handler for the /start command
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Hello! I'm your Healthcare Chatbot. How can I help you today?")

# Define message handler for regular user messages
def handle_messages(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text.lower()

    if user_input == "/exit":
        update.message.reply_text("Goodbye! Take care.")
        return

    try:
        # Calculate cosine similarity between user input and symptoms
        user_vector = vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(user_vector, symptom_vectors)[0]

        # Find the most similar symptom
        most_similar_index = similarity_scores.argmax()
        most_similar_symptom = symptoms[most_similar_index]

        if similarity_scores[most_similar_index] > 0.9:  # Adjust the threshold as needed
            selected_symptom = most_similar_symptom
            disease = get_random_disease(selected_symptom)
            response = (
                f"Based on the symptom '{selected_symptom}', a possible disease is '{disease}'.\n"
                f"Generic Medicine(s): {', '.join(disease_info.get('generic_medicine', []) or ['No information'])}\n"
                f"Home Remedy: {', '.join(disease_info.get('home_remedy', []) or ['No information'])}"
            )
        else:
            response = "I'm not sure how to respond to that."

        update.message.reply_text(response)

    except Exception as e:
        print(f"An error occurred: {e}")
        update.message.reply_text("Sorry, I encountered an error. Please try again.")

# Set up the Telegram bot
def main() -> None:
    updater = Updater("6966801227:AAHCoefeo5oolELxAZIjadCbzDJ_26e70i4")
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_messages))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
