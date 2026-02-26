import speech_recognition as sr

def transcribe_audio_from_mic():
    """
    Transcribes audio captured from the microphone using the SpeechRecognition library.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Say something!")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Processing audio...")
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
        except sr.WaitTimeoutError:
            print("No speech detected within the time limit.")
except sr.UnknownValueError:
            print("Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
if _name_ == "_main_":
    transcribe_audio_from_mic()
