import os
import pyaudio
import wave
import threading
import time
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.client import APIClient


# Setup IBM Cloud API
def get_credentials():
    return {
        "url": "https://eu-de.ml.cloud.ibm.com",
        "apikey":"EDvWcSDctufqdnk6D25nqpPzvvIWn8sYGlajStDUlfHD" # Replace with your IBM API Key
    }

model_id = "sdaia/allam-1-13b-instruct"
project_id ="1e873aaa-cd09-423f-8c9f-e11aac6ea139" # Replace with your project ID

# Initialize IBM Model
credentials = get_credentials()
model = Model(
    model_id=model_id,
    params={
        "decoding_method": "greedy",
        "max_new_tokens": 900,
        "repetition_penalty": 1
    },
    credentials=credentials,
    project_id=project_id
)

# Setup Google Cloud APIs
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Asus/Desktop/Mooben/mobeen-439017-d8be3e8adbb5.json"  # Replace with your credentials file

client_speech = speech.SpeechClient()
client_tts = texttospeech.TextToSpeechClient()

# Audio settings
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

def record_audio(duration=8):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    print("Recording...")

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open("output.wav", "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Recording complete.")

def transcribe_audio(filename):
    with open(filename, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ar-SA"  # Change this to your language preference
    )
    response = client_speech.recognize(config=config, audio=audio)
    for result in response.results:
        return result.alternatives[0].transcript

def synthesize_speech(text):
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ar-SA",  # Adjust language as needed
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
    response = client_tts.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

    with open("response.wav", "wb") as out:
        out.write(response.audio_content)

def play_audio(filename):
    p = pyaudio.PyAudio()
    wf = wave.open(filename, 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    p.terminate()

def chat_with_model(question):
    prompt_input = """
__grounding__

<<SYS>>
مرحلة 1:
"أهلاً وسهلاً! ما هو اسمك الجميل؟ 🌟 وكم عمرك؟"

بعد أن يجيب الطفل عن اسمه:

مرحلة 2:
"شكراً يا [اسم الطفل]، اسمك جميل! 🌈 هل تحب اللغة العربية؟ وما هو أكثر شيء تحبه فيها؟ هل تفضل القصص، الشعر، أو القواعد؟"

بعد أن يجيب عن اهتماماته:

مرحلة 3:
"رائع جداً يا [اسم الطفل]! 👏 في أي مدينة تعيش؟"

بعد أن يذكر المدينة:

مرحلة 4:
بما أنك تحب [اهتمام الطفل]، سأحكي لك قصة مفيدة وأخبرني ماذا استفدت منها

بعد أن يعطي الفائدة:

مرحلة 5:
أحسنت! وبما أنك شاطر وتحب [اهتمام الطفل] وتعيش في [المدينة]، أود أن أخبرك عن بعض الأنشطة الممتعة، يمكنك تجربة [اسم النشاط] بتاريخ [تاريخ النشاط]، وأيضاً [اسم نشاط آخر] بتاريخ [تاريخ النشاط الثاني]. أتمنى أن تستمتع بهذه الفعاليات!"

تنبيه :
تأكد من عدم الانتقال إلى السؤال التالي حتى يحصل النظام على إجابة الطفل عن السؤال الحالي.
<</SYS>>

 مرحبا [/INST]  أهلاً وسهلاً! ما هو اسمك الجميل؟ وكم عمرك؟  </s><s> [INST] اسمي شيماء وعمري 10 [/INST]  شكراً يا شيماء، اسمك جميل! هل تحبين اللغة العربية؟ وما هو أكثر شيء تحبينه فيها؟ هل تفضلين القصص، الشعر، أو القواعد؟  </s><s> [INST] نعم، أحب القصص [/INST]  رائع جداً يا شيماء! بما أنك تحبين القصص، سأحكي لك قصة مفيدة. هل تعلمين أن القصص تساعدنا على تعلم القيم والأخلاق بطريقة ممتعة وشيقة؟

في إحدى القصص، كان هناك صديقان يتنافسان في سباق. كان أحدهما سريعاً والآخر صبوراً. في النهاية، فاز الصديق الصبور لأنه لم يستعجل واستخدم استراتيجيته بحكمة.

ماذا تعلمت من هذه القصة يا شيماء؟ هل هناك درس معين استفدته منها؟  </s><s> [INST] الصبر والمثابرة [/INST]  أحسنتِ يا شيماء! تعلمتِ درساً مهماً من القصة وهو الصبر والمثابرة. هذه القيم تساعدنا في تحقيق أهدافنا والنجاح في حياتنا.

فيما يتعلق بالأنشطة المتعلقة بالقصص، يمكنك تجربة ورشة تعليم القصص التفاعلية بتاريخ 1/5/2025 في مدينة الجبيل. ستتعلم هناك كيفية تأليف قصص مبتكرة وجذابة.

أيضاً، يمكنك المشاركة في مسابقة كتابة القصص القصيرة بتاريخ 15/4/2025 في مدينة الرياض. ستتاح لك الفرصة لكتابة قصتك الخاصة والتنافس مع الأطفال الآخرين.

استمتعي بهذه الفعاليات واستمري في تنمية مهاراتك في كتابة القصص!  </s><s> [INST]"""

    formatted_question = f"<s> [INST] {question} [/INST]"
    prompt = f"{prompt_input}{formatted_question}"
    response = model.generate_text(prompt=prompt, guardrails=False)
    return response

def voice_chat():
    while True:
        print("Listening...")
        record_audio()
        question = transcribe_audio("output.wav")
        print(f"User: {question}")

        response_text = chat_with_model(question)
        print(f"AI: {response_text}")

        synthesize_speech(response_text)
        play_audio("response.wav")

if __name__ == "__main__":
    voice_chat()
