from email.mime import audio
import speech_recognition as sr
import pyttsx3
engine = pyttsx3.init()
r = sr.Recognizer()

def speak_text(command):
    engine.say(command)
    engine.runAndWait()

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

with sr.Microphone() as source1:
    r.adjust_for_ambient_noise(source1,duration=0.2)
    audio1 = r.listen(source1)

    text = r.recognize_google(audio1)
    text = text.lower()

    print("Did you Say "+ text)
    speak_text(text)