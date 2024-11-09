from flask import Flask, request, jsonify
import os
from model import chat_with_model, transcribe_audio, synthesize_speech  # Import functions from your existing code

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question")

    if question:
        response_text = chat_with_model(question)  # Generate a response using the model
        return jsonify({"response": response_text})
    else:
        return jsonify({"error": "No question provided"}), 400

@app.route('/transcribe_audio', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    audio_file.save("temp_audio.wav")
    question_text = transcribe_audio("temp_audio.wav")  # Transcribe the audio to text
    response_text = chat_with_model(question_text)  # Get response based on transcribed text

    # Optionally synthesize the response if you want to return audio as well
    synthesize_speech(response_text)
    with open("response.wav", "rb") as audio_response:
        audio_data = audio_response.read()

    return jsonify({
        "response_text": response_text,
        "response_audio": audio_data.hex()  # Send audio data as a hex string
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
