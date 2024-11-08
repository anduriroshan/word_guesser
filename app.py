import streamlit as st
import random
import numpy as np
import requests
import fasttext
import os

# Load the FastText model with caching
@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        model = fasttext.load_model(model_path)
        return model
    else:
        st.error(f"Model file not found at path: {model_path}")
        return None

model_path = r"C:\Users\rosha\Downloads\model.bin"
model = load_model(model_path)

# Fetching random nouns
@st.cache_data
def fetch_random_nouns(count=10, base_url="https://random-word-form.herokuapp.com/random/noun"):
    try:
        api_url = f"{base_url}?count={count}"
        response = requests.get(api_url)
        response.raise_for_status()
        words = response.json()
        return words
    except requests.RequestException as e:
        st.error(f"Error fetching words from API: {e}")
        return ["apple", "banana", "cherry"]  # Default words in case of error

@st.cache_data
def load_words():
    words = fetch_random_nouns()
    return words

def get_word_vectors(word):
    try:
        if model:
            vector = model.get_word_vector(word)
            return vector
        else:
            st.error("Model is not loaded.")
            return None
    except KeyError:
        return None

# Preload hints for the target word
def preload_hints(target_word, k=5):
    if model:
        neighbors = model.get_nearest_neighbors(target_word, k=k)
        return [neighbor[1] for neighbor in neighbors]
    return []

# Calculating similarity using cosine similarity
def calculate_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def main():
    st.title("Word Guessing Game")

    if 'target_word' not in st.session_state:
        st.session_state.target_word = random.choice(load_words())
        st.session_state.target_word_vector = get_word_vectors(st.session_state.target_word)
        if st.session_state.target_word_vector is not None:
            st.session_state.target_word_vector = st.session_state.target_word_vector.squeeze()
        else:
            st.session_state.target_word_vector = np.zeros(300)  # Default vector if word not found
        st.session_state.guess_history = []
        st.session_state.hint_count = 0
        st.session_state.hints = preload_hints(st.session_state.target_word)

    guess = st.text_input("Enter your guess:").lower()

    if st.button("Submit"):
        if guess:
            if guess == st.session_state.target_word:
                st.success("Congratulations! You've guessed the word correctly.")
                st.session_state.target_word = random.choice(load_words())
                st.session_state.target_word_vector = get_word_vectors(st.session_state.target_word)
                if st.session_state.target_word_vector is not None:
                    st.session_state.target_word_vector = st.session_state.target_word_vector.squeeze()
                else:
                    st.session_state.target_word_vector = np.zeros(300)
                st.session_state.guess_history = []
                st.session_state.hint_count = 0
                st.session_state.hints = preload_hints(st.session_state.target_word)
            else:
                guess_vector = get_word_vectors(guess)
                if guess_vector is not None:
                    guess_vector = guess_vector.squeeze()
                    score = calculate_similarity(guess_vector, st.session_state.target_word_vector)
                    st.session_state.guess_history.append((guess, score))
                    st.error(f"Your score: {score:.4f}")
                    st.info("Try again.")
                else:
                    st.error(f"Could not find a vector for the word '{guess}'.")

    if st.button("Hint"):
        if st.session_state.hint_count < 5:
            if st.session_state.hint_count < len(st.session_state.hints):
                hint = st.session_state.hints[st.session_state.hint_count]
                st.session_state.hint_count += 1
                st.info(f"Hint {st.session_state.hint_count}: {hint}")
            else:
                st.error("No hints available.")
        else:
            st.error(f"The correct word is: {st.session_state.target_word}")

    if st.session_state.guess_history:
        st.subheader("Guess History")
        for guess, score in st.session_state.guess_history:
            st.write(f"Guess: {guess}, Score: {score:.4f}")

if __name__ == "__main__":
    main()
