import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
import anthropic
import json
import os
import io

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VehicleIQ",
    page_icon="🚗",
    layout="centered"
)

# ─── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ff6b35, #f7c59f, #efefd0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #666680;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.result-card {
    background: linear-gradient(135deg, #13131f, #1a1a2e);
    border: 1px solid #2a2a45;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.vehicle-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #ff6b35;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.confidence-badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    background: #ff6b3520;
    color: #ff6b35;
    border: 1px solid #ff6b3540;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    display: inline-block;
    margin-top: 0.3rem;
}

.description-text {
    font-size: 0.95rem;
    line-height: 1.7;
    color: #b8b8d0;
    margin-top: 1rem;
}

.chat-message {
    padding: 0.8rem 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
}

.user-msg {
    background: #1e1e35;
    border-left: 3px solid #ff6b35;
    color: #e8e8f0;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}

.ai-msg {
    background: #13131f;
    border-left: 3px solid #4a9eff;
    color: #b8b8d0;
}

.divider {
    border: none;
    border-top: 1px solid #2a2a45;
    margin: 1.5rem 0;
}

.stButton > button {
    background: linear-gradient(135deg, #ff6b35, #ff8c5a) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px #ff6b3540 !important;
}

.stTextInput > div > div > input {
    background: #13131f !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #ff6b35 !important;
    box-shadow: 0 0 0 2px #ff6b3520 !important;
}

.footer-note {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #444460;
    text-align: center;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Vehicle Knowledge Base ────────────────────────────────────────────────────
VEHICLE_INFO = {
    "SUV": "An SUV is a versatile passenger vehicle engineered to combine the towing capacity of a truck with the passenger space of a minivan. It features high ground clearance and all-wheel drive for off-road environments. These vehicles are popular among families because they provide superior safety, high visibility, and ample cargo space. Most modern SUVs use gasoline or hybrid engines to balance power with fuel efficiency for daily commuting and long trips.",
    "bus": "A bus is a large-capacity transit vehicle specifically engineered for public transportation and high-volume urban mobility. It features a long-wheelbase chassis with multiple rows of seating and wide entryways to facilitate efficient passenger boarding. Designed for durability and constant operation, modern buses often utilize diesel, compressed natural gas, or electric powertrains to manage the heavy demands of scheduled city routes and commuter flow.",
    "family sedan": "A family sedan is a traditional four-door passenger car optimized for fuel efficiency, safety, and a smooth ride during daily urban or suburban commuting. It typically features a three-box configuration with separate compartments for the engine, passengers, and cargo. These vehicles are favored for their balanced handling and aerodynamic profiles, providing a comfortable environment for small families while maintaining lower operating costs than larger utility vehicles.",
    "fire engine": "A fire engine is a specialized emergency vehicle equipped with high-pressure pumps, large-capacity water tanks, and advanced firefighting apparatus for rapid incident response. It is built on a heavy-duty chassis to support the weight of ladders, hoses, and specialized rescue tools. These vehicles are engineered with powerful engines and emergency signaling systems to navigate traffic quickly, serving as a mobile command center for first responders at the scene of an emergency.",
    "heavy truck": "A heavy truck is a powerful commercial vehicle designed for long-haul logistics and the transport of massive freight loads across interstate highways. Often referred to as a semi-truck or tractor-trailer, it utilizes a high-torque diesel engine and a multi-gear transmission to pull heavy trailers. These vehicles are the backbone of global supply chains, featuring reinforced frames and sleeper cabs to accommodate drivers during multi-day transport missions.",
    "jeep": "A jeep is a rugged, compact 4x4 vehicle built for extreme terrain and off-road exploration. Characterized by its iconic open-body design, heavy-duty suspension system, and short wheelbase, it offers exceptional maneuverability in rocky or muddy environments. Originally developed for military use, modern versions retain a utilitarian aesthetic with removable doors and tops, appealing to outdoor enthusiasts who prioritize mechanical durability over highway luxury.",
    "minibus": "A minibus is a multi-purpose passenger vehicle designed for small group shuttle services or large family transport, prioritizing interior volume and accessibility. It sits between a full-sized van and a bus in terms of capacity, often featuring sliding door access and configurable seating arrangements. These vehicles are commonly used for airport transfers, school transport, and community transit due to their ability to carry up to 15 passengers while remaining easy to park and maneuver.",
    "racing car": "A racing car is a high-performance vehicle precision-engineered for maximum speed, downforce, and aerodynamic efficiency. Featuring a lightweight chassis made of carbon fiber or advanced alloys, it is powered by a high-RPM engine designed for peak output rather than longevity. These vehicles sacrifice comfort and cargo space for safety features like roll cages and specialized fuel cells, allowing drivers to compete at high velocities on closed circuits.",
    "taxi": "A taxi is a standard commercial passenger vehicle equipped with a taximeter and distinct livery, optimized for short-duration urban hire and on-demand transit. These vehicles are typically chosen for their mechanical reliability and rear-seat legroom to ensure passenger comfort in dense city traffic. Modern taxi fleets often incorporate hybrid or electric technology to reduce idle emissions and operational costs during long shifts in metropolitan areas.",
    "truck": "A truck is a light to medium-duty utility vehicle featuring an enclosed passenger cab and an open cargo bed for hauling tools, equipment, or materials. Popular for both professional trades and personal use, it offers a robust body-on-frame construction that provides high payload and towing capacities. Whether used on a construction site or for weekend chores, the pickup truck remains a staple of versatility due to its ability to transition between a passenger vehicle and a heavy-duty work tool."
}

CLASSES = list(VEHICLE_INFO.keys())

# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_url = "https://huggingface.co/SSandeepKumar/Vehicle_Recognition/resolve/main/cnn_model.pth"
    model_path = "/tmp/cnn_model.pth"

    if not os.path.exists(model_path):
        with st.spinner("⚡ Loading AI model for the first time..."):
            response = requests.get(model_url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image: Image.Image):
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        outputs = load_model()(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return CLASSES[predicted.item()], confidence.item()

# ─── Claude API ────────────────────────────────────────────────────────────────
def ask_claude(vehicle_type: str, question: str, facts: str) -> str:
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are VehicleIQ, an expert vehicle analyst. Answer the user's question about the identified vehicle.

Vehicle Type: {vehicle_type}
Vehicle Facts: {facts}
User Question: {question}

Give a helpful, concise, expert answer in 2-3 sentences. Be specific and informative."""
                }
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"Unable to get AI response: {str(e)}"

def generate_description(vehicle_type: str, facts: str) -> str:
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are VehicleIQ, an expert vehicle analyst. Write a sharp, engaging 3-sentence description of the identified vehicle.

Vehicle Type: {vehicle_type}
Facts: {facts}

Be expert, specific, and interesting. No generic filler."""
                }
            ]
        )
        return message.content[0].text
    except Exception as e:
        return facts

# ─── Session State ─────────────────────────────────────────────────────────────
if "vehicle" not in st.session_state:
    st.session_state.vehicle = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "description" not in st.session_state:
    st.session_state.description = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">VehicleIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-Powered Vehicle Recognition & Analysis</div>', unsafe_allow_html=True)

# ─── Sidebar Input ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📸 Input Source")
    input_mode = st.radio("Select Method:", ["Upload Image", "Live Camera"])

    source_file = None
    if input_mode == "Upload Image":
        source_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    else:
        source_file = st.camera_input("Take a photo")

    if st.button("🔄 Reset Session"):
        st.session_state.vehicle = None
        st.session_state.confidence = None
        st.session_state.description = None
        st.session_state.chat_history = []
        st.rerun()

if source_file:
    image = Image.open(source_file)
    st.image(image, use_container_width=True)

    if st.button("🔍 Analyze & Identify"):
        with st.spinner("🔍 Analyzing vehicle..."):
            label, conf = predict(image)
            facts = VEHICLE_INFO.get(label, "A general purpose vehicle.")
            description = generate_description(label, facts)

            st.session_state.vehicle = label
            st.session_state.confidence = conf
            st.session_state.description = description
            st.session_state.chat_history = []

if st.session_state.vehicle:
    st.markdown(f"""
    <div class="result-card">
        <div class="vehicle-label">{st.session_state.vehicle}</div>
        <div class="confidence-badge">Confidence: {st.session_state.confidence*100:.1f}%</div>
        <div class="description-text">{st.session_state.description}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f"#### 💬 Chat with the {st.session_state.vehicle} Expert")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(f"Ask anything about this {st.session_state.vehicle}..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                facts = VEHICLE_INFO.get(st.session_state.vehicle, "")
                answer = ask_claude(st.session_state.vehicle, prompt, facts)
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

st.markdown('<div class="footer-note">Built by SSandeepKumar · Powered by ResNet18 + Claude AI</div>', unsafe_allow_html=True)
