import opik
import os
import getpass

from google import genai
from opik import track
from opik.integrations.genai import track_genai

opik.configure()


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] ="your api key"


os.environ["OPIK_PROJECT_NAME"] = "Medical Data Analyzer"

client = genai.Client()
gemini_client = track_genai(client)
document = """
    A 62-year-old male with a history of hypertension, type 2 diabetes, and smoking presented to the ED with acute onset chest pain radiating to the left arm, associated with diaphoresis and shortness of breath. On arrival, BP was 96/62 mmHg, HR 112 bpm, SpO2 89% on room air.

    ECG showed ST elevation in leads II, III, and aVF. Troponin I was elevated at 6.2 ng/mL. He was diagnosed with STEMI (inferior wall). Immediate cardiac catheterization revealed 100% occlusion of the right coronary artery (RCA); a drug-eluting stent was placed successfully.

    Post-PCI, he developed hypotension with signs of right ventricular infarction confirmed by echocardiogram. Fluid resuscitation was initiated, and norepinephrine was started. He was admitted to the CICU. Over 72 hours, he developed new-onset atrial fibrillation and transient acute kidney injury (creatinine rose from 1.1 to 2.3 mg/dL), managed conservatively.

    He stabilized after 6 days, transitioned to oral beta-blockers, statin, dual antiplatelet therapy, and discharged with cardiology follow-up.
    """

prompt = f"Analyze patient's clinical presentation, diagnostic findings, treatment strategy, complications, and outcome using precise medical language content {document}"



response = gemini_client.models.generate_content(
    model="gemini-2.0-flash-001", contents=prompt
)
print(response.text)


@track
def generate_story(prompt):
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash-001", contents=prompt
    )
    return response.text


@track
def generate_topic():
    prompt = "Generate a topic for a story about Opik."
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash-001", contents=prompt
    )
    return response.text


@track
def generate_opik_story():
    topic = generate_topic()
    story = generate_story(topic)
    return story


generate_opik_story()
