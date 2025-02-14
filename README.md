# AI-Powered-Healthcare-Assistant
from transformers import pipeline

# Load the question-answering model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Improved Medical Context with Q&A format
medical_context = """
Q: What are the symptoms of a cold?
A: Runny nose, sneezing, sore throat, mild fever.

Q: What is the treatment for a cold?
A: Rest, drink warm fluids, use nasal drops.

Q: What are the types of burns?
A: First-degree burns (redness, mild pain), second-degree burns (blisters, swelling), third-degree burns (charred skin, no pain).

Q: What is the treatment for burns?
A: First-degree: Cool with running water, apply aloe vera.
Second-degree: Apply antibiotic cream, bandage loosely.
Third-degree: Seek emergency care.

Q: What are the symptoms of fever?
A: High temperature, chills, sweating, body aches.

Q: What is the treatment for fever?
A: Drink plenty of fluids, take paracetamol, rest.
"""

# Function to handle better query understanding
def format_query(user_query):
    user_query = user_query.lower().strip()

    if "treatment" in user_query:
        return f"What is the treatment for {user_query.replace('treatment', '').strip()}?"
    elif len(user_query.split()) == 1:
        return f"What are the symptoms of {user_query}?"
    return user_query

# Main loop for the AI Health Assistant
print("Welcome to AI Health Assistant! Type your query or 'exit' to stop.")

while True:
    user_input = input("Enter your health query: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye! Stay healthy.")
        break

    # Format query for better model understanding
    formatted_question = format_query(user_input)

    # Run question-answering pipeline
    answer = qa_model(question=formatted_question, context=medical_context)

    # Improved threshold check
    if answer['score'] > 0.01:
        print(f"AI: {answer['answer']}")
    else:
        print("AI: I'm sorry, I don't have enough information. Please consult a doctor.")

# Interactive text-based interface
print("Welcome to AI Health Assistant! Type your query or 'exit' to stop.")

while True:
    user_input = input("Enter your health query: ")
    if user_input.lower() == "exit":
        print("Goodbye! Stay healthy. 😊")
        break
    response = get_medical_response(user_input)
    print(f"AI: {response}\n")
