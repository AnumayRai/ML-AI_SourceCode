from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
from googletrans import Translator

# Google Cloud Speech API
def transcribe_audio(file_path):
    # Set up credentials
    credentials = service_account.Credentials.from_service_account_file('path/to/your/google-cloud-key.json')
    client = speech.SpeechClient(credentials=credentials)

    # Load audio file
    with open(file_path, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)

    # Configure speech recognition
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    # Transcribe audio
    response = client.recognize(config=config, audio=audio)
    transcription = response.results[0].alternatives[0].transcript
    return transcription

# Google Translate API
def translate_text(text, dest_language):
    translator = Translator()
    translation = translator.translate(text, dest=dest_language)
    return translation.text

# Main function
def main():
    file_path = 'path/to/your/audio/file.wav'
    transcription = transcribe_audio(file_path)
    print(f'Transcription: {transcription}')

    dest_language = 'es'  # Spanish
    translation = translate_text(transcription, dest_language)
    print(f'Translation: {translation}')

if __name__ == '__main__':
    main()
