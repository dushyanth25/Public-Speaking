import json
import time
import random
import speech_recognition as sr
import requests
import librosa
import streamlit as st
import numpy as np
from io import BytesIO
import string
import os
from gtts import gTTS  # For text-to-speech conversion
from transformers import AutoModelForCausalLM, AutoTokenizer  # For Hugging Face model
import torch

# Initialize session state for user data
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'points': 0,
        'level': 1,
        'streak': 0,
        'high_score': 0,
        'learned': []
    }

if 'sentences' not in st.session_state:
    st.session_state.sentences = {
        1: [],  # Level 1 will use AI-generated words
        2: [
            "The quick brown fox jumps over the lazy dog.",
            "Programming requires logic and creativity.",
            "Keep your friends close and your enemies closer."
        ],
        3: [
            "Success is the sum of small efforts, repeated daily.",
            "In difficulty lies opportunity.",
            "Great work requires love and dedication."
        ]
    }

# Topics for Level 2
topics = [
    "The importance of education",
    "Climate change and its effects",
    "The future of artificial intelligence",
    "The role of technology in modern life",
    "The benefits of reading books"
]

# Encouragement messages
encouragements = [
    "Nice!", "Great job!", "Awesome!",
    "Fantastic!", "You're crushing it!", 
    "Unstoppable!", "Legendary!"
]

# Load or initialize sentences
def load_sentences():
    try:
        with open('sentences.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return st.session_state.sentences

# Save sentences to file
def save_sentences(sentences):
    with open('sentences.json', 'w') as f:
        json.dump(sentences, f)

# Calculate points
def calculate_points(time_taken, sentence):
    base_points = len(sentence) * 2
    time_bonus = max(100 - int(time_taken), 10)
    streak_bonus = st.session_state.user_data['streak'] * 5
    return base_points + time_bonus + streak_bonus

# Check for level up based on high score
def check_level_up():
    high_score = st.session_state.user_data['high_score']
    if high_score >= 1500 and st.session_state.user_data['level'] < 3:
        st.session_state.user_data['level'] = 3
        st.success(f"🌟 LEVEL UP! You've reached level 3!")
    elif high_score >= 500 and st.session_state.user_data['level'] < 2:
        st.session_state.user_data['level'] = 2
        st.success(f"🌟 LEVEL UP! You've reached level 2!")
    return False

def voice_to_text(duration=5):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write(f"🎤 Speak now for {duration} seconds...")
        recognizer.adjust_for_ambient_noise(source)  # Reduce background noise
        audio = recognizer.listen(source, timeout=duration)

    try:
        text = recognizer.recognize_google(audio)  # Convert speech to text
        st.write(f"📝 You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("❌ Could not understand the audio.")
        return ""
    except sr.RequestError:
        st.error("❌ Speech recognition service unavailable.")
        return ""

def generate_random_words():
    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew"]
    return " ".join(random.sample(words, 3))  # Generate a sentence of 3 random words

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    return text

def play_round():
    level = st.session_state.user_data['level']
    if level == 1:
        sentence = generate_random_words()
    else:
        available = st.session_state.sentences.get(str(level), [])
        if not available:
            st.warning("No sentences available for this level!")
            return
        sentence = random.choice([s for s in available if s not in st.session_state.user_data['learned']] or available)
    
    st.write(f"🎤 SAY THIS (Level {level}):\n**{sentence}**")
    
    if st.button("Start Recording"):
        start = time.time()
        attempt = voice_to_text().strip()
        duration = time.time() - start

        # Preprocess both the target sentence and the user's attempt
        processed_sentence = preprocess_text(sentence)
        processed_attempt = preprocess_text(attempt)

        if processed_attempt == processed_sentence:
            points = calculate_points(duration, sentence)
            st.session_state.user_data['points'] += points
            st.session_state.user_data['streak'] += 1
            st.session_state.user_data['learned'].append(sentence)
            
            st.success(f"✅ CORRECT! Time: {duration:.2f}s")
            st.write(f"➕ Points: {points} (Total: {st.session_state.user_data['points']})")
            st.write(random.choice(encouragements))
            
            if st.session_state.user_data['streak'] % 5 == 0:
                st.write(f"🔥 {st.session_state.user_data['streak']}-STREAK BONUS!")
            
            if st.session_state.user_data['points'] > st.session_state.user_data['high_score']:
                st.session_state.user_data['high_score'] = st.session_state.user_data['points']
                st.write("🎉 NEW HIGH SCORE!")
            
            check_level_up()  # Ensure this is called
        else:
            st.error(f"❌ INCORRECT. The right answer was: {sentence}")
            st.session_state.user_data['streak'] = 0
        
        save_sentences(st.session_state.sentences)

def add_sentence():
    sentence = st.text_input("Enter new sentence:")
    level = st.number_input("Assign difficulty level (1-3):", min_value=1, max_value=3, step=1)
    if st.button("Add Sentence"):
        st.session_state.sentences.setdefault(str(level), []).append(sentence)
        save_sentences(st.session_state.sentences)
        st.success(f"Sentence added to Level {level}!")

def show_leaderboard():
    if os.path.exists('users.json'):
        with open('users.json', 'r') as f:
            users = json.load(f)
        sorted_users = sorted(users.items(), key=lambda x: x[1]['high_score'], reverse=True)
        st.write("### Leaderboard")
        for i, (username, data) in enumerate(sorted_users[:10], 1):
            st.write(f"{i}. {username}: {data['high_score']} points")
    else:
        st.warning("No users found. Leaderboard is empty.")

def text_to_speech(text, gender='female'):
    tts = gTTS(text=text, lang='en')
    if gender == 'male':
        # Adjust parameters for male voice (if possible)
        pass  # gTTS does not support changing voice gender directly
    audio_file = "temp_audio.mp3"
    tts.save(audio_file)
    return audio_file

# Load the Hugging Face model and tokenizer
@st.cache_resource
def load_huggingface_model():
    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        st.write("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_huggingface_model()
if model is None or tokenizer is None:
    st.error("Failed to load the model. Please check your installation.")

# Chat with Hugging Face AI (Audio Input)
def chat_with_ai_audio(user_gender):
    st.write("🎤 Speak to the AI...")
    
    # Capture audio input from the user
    user_input = voice_to_text(duration=5)  # Allow 5 seconds for user input
    
    if user_input:
        st.write(f"📝 You said: {user_input}")
        
        # Generate AI response
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # Determine AI's voice gender (opposite of user's gender)
        ai_gender = 'male' if user_gender.lower() == 'female' else 'female'
        
        # Convert AI response to speech
        audio_file = text_to_speech(ai_response, ai_gender)
        
        # Display AI response and play audio
        st.write(f"🤖 AI: {ai_response}")
        st.audio(audio_file, format='audio/mp3')
    else:
        st.error("❌ Could not understand your audio. Please try again.")

def speak_on_topic():
    if st.session_state.user_data['level'] >= 2:
        topic = random.choice(topics)
        st.write(f"🎤 Speak about the following topic for 30 seconds:\n**{topic}**")
        
        if st.button("Start Speaking"):
            user_speech = voice_to_text(duration=30)
            if user_speech:
                # Validate the speech using AI
                inputs = tokenizer(user_speech, return_tensors="pt")
                outputs = model.generate(**inputs, max_length=100)
                ai_feedback = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                # Provide feedback and update points
                st.write(f"🤖 AI Feedback: {ai_feedback}")
                points = len(user_speech.split()) * 2  # Award points based on the number of words
                st.session_state.user_data['points'] += points
                st.write(f"➕ Points: {points} (Total: {st.session_state.user_data['points']})")
                
                # Update leaderboard
                if st.session_state.user_data['points'] > st.session_state.user_data['high_score']:
                    st.session_state.user_data['high_score'] = st.session_state.user_data['points']
                    st.write("🎉 NEW HIGH SCORE!")
                
                check_level_up()

def login_signup():
    st.write("### Login / Signup")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    gender = st.radio("Select your gender", ('male', 'female'))
    if st.button("Login"):
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                users = json.load(f)
            if username in users and users[username]['password'] == password:
                st.session_state.user_data = users[username]
                st.session_state.user_data['gender'] = gender
                # Set level based on high score
                if st.session_state.user_data['high_score'] >= 1500:
                    st.session_state.user_data['level'] = 3
                elif st.session_state.user_data['high_score'] >= 500:
                    st.session_state.user_data['level'] = 2
                else:
                    st.session_state.user_data['level'] = 1
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password")
        else:
            st.error("No users found. Please sign up.")
    if st.button("Signup"):
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                users = json.load(f)
        else:
            users = {}
        if username in users:
            st.error("Username already exists")
        else:
            users[username] = {
                'password': password,
                'points': 0,
                'level': 1,
                'streak': 0,
                'high_score': 0,
                'learned': [],
                'gender': gender
            }
            with open('users.json', 'w') as f:
                json.dump(users, f)
            st.success("Signed up successfully!")

def main():
    st.title("🎮 Sentence Master Game")
    st.sidebar.header("Menu")
    menu_option = st.sidebar.radio("Choose an option:", ["Play Game", "Add Sentence", "View Stats", "Leaderboard", "Login/Signup", "Chat with AI"])

    if menu_option == "Play Game":
        if st.session_state.user_data['level'] >= 3:
            speak_on_favorite_topic()
        elif st.session_state.user_data['level'] >= 2:
            speak_on_topic()  # Handle Level 2 topic speaking
        else:
            play_round()  # Play the regular game for Level 1
    elif menu_option == "Add Sentence":
        add_sentence()
    elif menu_option == "View Stats":
        st.write("### User Stats")
        st.write(f"🏆 High Score: {st.session_state.user_data['high_score']}")
        st.write(f"⭐ Current Level: {st.session_state.user_data['level']}")
        st.write(f"💰 Points: {st.session_state.user_data['points']}")
        st.write(f"🔥 Streak: {st.session_state.user_data['streak']}")
    elif menu_option == "Leaderboard":
        show_leaderboard()
    elif menu_option == "Login/Signup":
        login_signup()
    elif menu_option == "Chat with AI":
        if 'gender' not in st.session_state.user_data:
            st.warning("Please login/signup first to chat with AI.")
        else:
            if st.button("Start Speaking to AI"):
                chat_with_ai_audio(st.session_state.user_data['gender'])

def speak_on_favorite_topic():
    if st.session_state.user_data['level'] >= 3:
        favorite_topic = st.text_input("Enter your favorite topic:")
        if favorite_topic:
            st.write(f"🎤 Speak about your favorite topic: {favorite_topic}")
            
            if st.button("Start Speaking"):
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    st.write("🎤 Speak now...")
                    recognizer.adjust_for_ambient_noise(source)  # Reduce background noise
                    audio = recognizer.listen(source, timeout=3)  # Stop recording after 3 seconds of silence

                try:
                    user_speech = recognizer.recognize_google(audio)  # Convert speech to text
                    st.write(f"📝 You said: {user_speech}")
                    
                    # Validate the speech using AI
                    inputs = tokenizer(user_speech, return_tensors="pt")
                    outputs = model.generate(**inputs, max_length=100)
                    ai_feedback = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    
                    # Provide feedback and update points
                    st.write(f"🤖 AI Feedback: {ai_feedback}")
                    points = len(user_speech.split()) * 2  # Award points based on the number of words
                    st.session_state.user_data['points'] += points
                    st.write(f"➕ Points: {points} (Total: {st.session_state.user_data['points']})")
                    
                    # Update leaderboard
                    if st.session_state.user_data['points'] > st.session_state.user_data['high_score']:
                        st.session_state.user_data['high_score'] = st.session_state.user_data['points']
                        st.write("🎉 NEW HIGH SCORE!")
                    
                    check_level_up()
                except sr.UnknownValueError:
                    st.error("❌ Could not understand the audio.")
                except sr.RequestError:
                    st.error("❌ Speech recognition service unavailable.")

if __name__ == "__main__":
    main()