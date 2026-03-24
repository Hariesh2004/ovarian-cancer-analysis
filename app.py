import streamlit as st
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import os

# Set page configuration for a premium look
st.set_page_config(
    page_title="Ovarian Cancer Biopsy Analysis",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism and modern styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #121218 100%);
        color: #f0f0f5;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Highlight text */
    .highlight {
        color: #4CAF50;
        font-weight: bold;
    }

    /* Glassmorphism container */
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
    }
    
    /* Upload box styling overrides */
    div[data-testid="stFileUploadDropzone"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploadDropzone"]:hover {
        border-color: #6366f1;
        background-color: rgba(99, 102, 241, 0.05);
    }
    
    /* Results cards */
    .result-card {
        background: rgba(255, 255, 255, 0.08);
        border-left: 4px solid #6366f1;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🔬 Ovarian Cancer Analysis ")
st.markdown("""
    <div style="color: #a0a0b0; font-size: 1.1rem; margin-bottom: 2rem;">
        Upload a histological slide biopsy image, and our fine-tuned <b>Vision Transformer (ViT)</b> 
        will analyze it to determine the probability of it belonging to one of the four main ovarian cancer classes.
    </div>
""", unsafe_allow_html=True)

# Cache the model loading to prevent reloading on every interaction
@st.cache_resource
def load_model():
    # Using the consolidated model weights
    model_path = "./model_weights"
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processor = ViTImageProcessor.from_pretrained(model_path)
        model = ViTForImageClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return processor, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def format_class_name(class_str):
    """Format strings like 'HGSC' to 'High-Grade Serous Carcinoma' or 'germ_cell' to 'Germ Cell'"""
    # Mapping for Kaggle abbreviations
    abbrev_map = {
        'CC': 'Clear Cell Carcinoma',
        'EC': 'Endometrioid Carcinoma',
        'HGSC': 'High-Grade Serous Carcinoma',
        'LGSC': 'Low-Grade Serous Carcinoma',
        'MC': 'Mucinous Carcinoma'
    }
    if class_str.upper() in abbrev_map:
        return abbrev_map[class_str.upper()]
        
    words = class_str.replace('_', ' ').split()
    return " ".join(word.capitalize() for word in words)

processor, model, device = load_model()

if model is None:
    st.error("❌ Model weights not found. Ensure you have run local training first.")
    st.stop()

# Main Container
# Sidebar - Dataset & Model Information
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/6366f1/microscope.png", width=80)
    st.header("Project Statistics")
    st.markdown("""
    **Model Architecture:** 
    `Vision Transformer (ViT-Base)`
    
    **Dataset:** 
    Kaggle Ovarian Cancer Dataset
    
    **Training Profile:**
    - Images: 34,285
    - Classes: 5 (HGSC, EC, CC, MC, LGSC)
    - Loss: Weighted Cross-Entropy
    - Epochs: 10 (Augmented)
    """)
    st.divider()
    st.info("💡 **Clinical Tip:** The model is optimized to distinguish between High-Grade and Low-Grade Serous Carcinoma based on nuclear morphology.")

# Main Application Tabs
tab1, tab2 = st.tabs(["🔍 Patient Analysis", "📚 Clinical Reference Library"])

with tab1:
    # --- (Moving existing analysis logic here) ---
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.subheader("📤 Upload Biopsy Scan")
    uploaded_file = st.file_uploader("Choose a histological slide", type=["jpg", "jpeg", "png"], key="main_uploader")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1], gap="large")
        
        try:
            image = Image.open(uploaded_file).convert("RGB")
            with col1:
                st.markdown('<div class="glass-container">', unsafe_allow_html=True)
                st.subheader("Captured Specimen")
                t_image = image.copy()
                t_image.thumbnail((400, 400))
                st.image(t_image, use_container_width=True, caption=f"ID: {uploaded_file.name}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="glass-container">', unsafe_allow_html=True)
                st.subheader("Analysis of Clinical Diagnosis")
                
                with st.spinner("Analyzing high-resolution cellular patterns..."):
                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                    
                    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
                    idx_prob_pairs = []
                    for idx, prob in enumerate(probs):
                        cls_name = model.config.id2label[idx]
                        idx_prob_pairs.append((format_class_name(cls_name), prob.item()))
                    
                    idx_prob_pairs.sort(key=lambda x: x[1], reverse=True)
                    predicted_class = idx_prob_pairs[0][0]
                    confidence = idx_prob_pairs[0][1] * 100
                    
                    # Morphology Insights Mapping
                    morphology_insights = {
                        'High-Grade Serous Carcinoma': "Exhibits marked nuclear atypia, pleomorphism, and high mitotic activity. Often presents with TP53 mutations.",
                        'Clear Cell Carcinoma': "Characterized by cells with clear cytoplasm, hobnail patterns, and hyaline bodies. Highly associated with endometriosis.",
                        'Endometrioid Carcinoma': "Displays glandular patterns resembling the uterine endometrium. Often PTEN or ARID1A mutated.",
                        'Low-Grade Serous Carcinoma': "Presents with small, uniform cells and low mitotic rate. Often associated with KRAS or BRAF mutations.",
                        'Mucinous Carcinoma': "Features large cells with abundant intracellular mucin. Often forms complex glandular or papillary structures."
                    }
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h3 style="margin: 0; color: #6366f1;">Primary Diagnosis</h3>
                        <h2 style="margin: 5px 0;">{predicted_class}</h2>
                        <div style="font-size: 1.1rem;">Confidence Score: <b>{confidence:.1f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    risk_color = "#ef4444" if confidence > 70 else "#f59e0b"
                    risk_status = "CRITICAL (High Confidence)" if confidence > 70 else "SUGGESTIVE (Review Needed)"
                    
                    st.markdown(f"""
                    <div class="result-card" style="border-left-color: {risk_color}; background: rgba(239, 68, 68, 0.05);">
                        <h3 style="margin: 0; color: {risk_color};">Clinical Insights</h3>
                        <h4 style="margin: 5px 0;">{risk_status}</h4>
                        <div style="font-size: 0.95rem; color: #d1d5db; line-height: 1.4;">
                            <b>Morphology:</b> {morphology_insights.get(predicted_class, "Standard cellular presentation.")}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### Diagnosis Matrix")
                    for cls_name, prob in idx_prob_pairs:
                        st.write(f"{cls_name} ({prob*100:.1f}%)")
                        st.progress(prob)

                    # Export button - TXT for now
                    report_text = f"OVARIAN CANCER ANALYSIS\nDiagnosis: {predicted_class}\nConfidence: {confidence:.2f}%\n"
                    st.download_button("📄 Export Clinical Report", report_text, file_name="report.txt", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Analysis error: {e}")

with tab2:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.header("🔬 Subtype Reference Library")
    st.write("Compare Analysis results with standard histological characteristics of the five main ovarian carcinoma subtypes.")
    
    sub = st.selectbox("Select Subtype to Study", ["HGSC", "CC", "EC", "LGSC", "MC"])
    
    col_ref1, col_ref2 = st.columns([1, 2])
    
    with col_ref1:
        # Placeholder for standard reference images
        st.image(f"https://placehold.co/400x400/1e1e2f/ffffff?text={sub}+Slide+Example", use_container_width=True)
        
    with col_ref2:
        if sub == "HGSC":
            st.subheader("High-Grade Serous Carcinoma (HGSC)")
            st.markdown("""
            - **Morphology:** Solid, papillary, and glandular patterns. 
            - **Key Identifiers:** Massive nuclear atypia, slit-like spaces.
            - **Clinical:** Most aggressive form, usually diagnosed at Stage III or IV.
            """)
        elif sub == "CC":
            st.subheader("Clear Cell Carcinoma (CC)")
            st.markdown("""
            - **Morphology:** Clear cytoplasm due to glycogen content.
            - **Key Identifiers:** 'Hobnail' cells (nuclei bulging into lumen).
            - **Clinical:** Often resistant to platinum-based chemotherapy.
            """)
        elif sub == "EC":
            st.subheader("Endometrioid Carcinoma (EC)")
            st.markdown("""
            - **Morphology:** Back-to-back glands with smooth internal borders.
            - **Key Identifiers:** Squamous differentiation in ~25% of cases.
            - **Clinical:** Often co-exists with uterine endometriosis.
            """)
        elif sub == "LGSC":
            st.subheader("Low-Grade Serous Carcinoma (LGSC)")
            st.markdown("""
            - **Morphology:** Small nests of cells with mild to moderate atypia.
            - **Key Identifiers:** Absence of high-grade nuclear features seen in HGSC.
            - **Clinical:** Occurs in younger patients; indolent but chemo-resistant.
            """)
        elif sub == "MC":
            st.subheader("Mucinous Carcinoma (MC)")
            st.markdown("""
            - **Morphology:** Large volume of extracellular mucin.
            - **Key Identifiers:** Stratified columnar epithelium with basal nuclei.
            - **Clinical:** Frequently large unilateral masses.
            """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem; margin-top: 3rem;">
        Ovarian Cancer Histopathology AI • Version 2.0 (Integrated Kaggle Dataset)
    </div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    # your model loading code
    return model

model = load_model()