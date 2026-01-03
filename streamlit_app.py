import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np

# Try importing SwinForImageClassification with better error handling
try:
    from transformers import SwinForImageClassification
except ImportError:
    try:
        # Try alternative import path for older versions
        from transformers.models.swin import SwinForImageClassification
    except ImportError:
        st.error("""
        ❌ **Import Error**: `SwinForImageClassification` not found in transformers.
        
        **Solution**: Please update transformers to version 4.11.0 or later:
        ```bash
        pip install --upgrade transformers>=4.11.0
        ```
        """)
        st.stop()

# Page config
st.set_page_config(
    page_title="Eye Disease Classification",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .disease-name {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .confidence-bar {
        height: 30px;
        background-color: #e0e0e0;
        border-radius: 5px;
        margin: 0.5rem 0;
        position: relative;
    }
    </style>
""", unsafe_allow_html=True)

# Model configurations
MODEL_CONFIGS = {
    "Swin Transformer (Small) - Mean Teacher": {
        "file": "best_eye_disease_model.pt",
        "description": "Semi-supervised Mean Teacher with Swin Small",
        "default_model_name": "microsoft/swin-small-patch4-window7-224"
    },
    "Swin Transformer (Tiny) - K-Fold": {
        "file": "swin_tiny_kfold_model.pt",  # You'll need to save this from 481(2).ipynb
        "description": "K-Fold Cross-Validation with Swin Tiny",
        "default_model_name": "microsoft/swin-tiny-patch4-window7-224"
    }
}

# Cache models separately by file name
_model_cache = {}

def load_model(model_file, default_model_name):
    """Load the trained model and metadata - cached by model file"""
    # Check cache first
    cache_key = f"{model_file}_{default_model_name}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_file, map_location=device)
        label2id = checkpoint['label2id']
        id2label = checkpoint['id2label']
        
        # Get model name from checkpoint if available, otherwise use default
        model_name = checkpoint.get('model_name', default_model_name)
        
        # Check transformers version compatibility
        import transformers
        saved_version = checkpoint.get('transformers_version', 'unknown')
        current_version = transformers.__version__
        version_mismatch = (saved_version != 'unknown' and saved_version != current_version)
        
        # Initialize model
        model = SwinForImageClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        ).to(device)
        
        # Load weights with strict=False to handle architecture mismatches
        # This allows partial loading if there are version differences
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['model_state_dict'], 
            strict=False
        )
        
        model.eval()
        
        # Return model and metadata (including warnings info)
        result = (model, label2id, id2label, device, {
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'version_mismatch': version_mismatch,
            'saved_version': saved_version,
            'current_version': current_version
        })
        
        # Cache the result
        _model_cache[cache_key] = result
        return result
        
    except FileNotFoundError:
        return None, None, None, None, {'error': f'Model file {model_file} not found'}
    except Exception as e:
        return None, None, None, None, {'error': str(e)}

@st.cache_data
def get_transform():
    """Get the validation transform (same as used during training)"""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def predict_image(model, image, transform, device, id2label):
    """Run inference on a single image"""
    # Preprocess image
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(pixel_values=img_tensor)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
    
    # Get predictions
    probs_np = probs.cpu().numpy()[0]
    pred_idx = np.argmax(probs_np)
    pred_label = id2label[pred_idx]
    confidence = probs_np[pred_idx]
    
    # Get all class probabilities
    class_probs = {id2label[i]: float(probs_np[i]) for i in range(len(id2label))}
    
    return pred_label, confidence, class_probs

# Main app
st.markdown('<div class="main-header">👁️ Eye Disease Classification</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. Select a model from the dropdown
    2. Upload an eye fundus image
    3. The model will classify it into one of:
       - **Cataract**
       - **Diabetic Retinopathy**
       - **Glaucoma**
       - **Normal**
    4. View predictions and confidence scores
    """)
    
    st.markdown("---")
    st.header("🤖 Model Selection")
    
    # Model selection dropdown
    model_names = list(MODEL_CONFIGS.keys())
    selected_model = st.selectbox(
        "Choose a model:",
        model_names,
        index=0,  # Default to first model
        help="Select which trained model to use for prediction"
    )
    
    # Display model info
    model_config = MODEL_CONFIGS[selected_model]
    st.markdown("**Model Info**")
    st.caption(selected_model)
    st.caption(model_config['description'])
    st.caption(f"File: {model_config['file']}")

# Load selected model (cached)
selected_config = MODEL_CONFIGS[selected_model]
model_file = selected_config['file']
default_model_name = selected_config['default_model_name']

try:
    model, label2id, id2label, device, load_info = load_model(model_file, default_model_name)
    
    # Check for errors
    if model is None:
        if 'error' in load_info:
            if 'not found' in load_info['error'].lower():
                st.error(f"❌ Model file '{model_file}' not found. Please train and save the model first.")
                st.info(f"💡 To save the model from 481(2).ipynb, add this code at the end:")
                st.code(f"""
torch.save({{
    'model_state_dict': final_model.state_dict(),
    'label2id': label2id,
    'id2label': id2label,
    'model_name': 'microsoft/swin-tiny-patch4-window7-224',
    'transformers_version': transformers.__version__,
}}, 'swin_tiny_kfold_model.pt')
""", language='python')
            else:
                st.error(f"❌ Error loading model: {load_info['error']}")
        st.stop()
    
    # Display warnings outside cached function
    if load_info.get('version_mismatch'):
        st.warning(f"⚠️ Transformers version mismatch: Saved with {load_info['saved_version']}, current is {load_info['current_version']}. This may cause loading issues.")
    
    if load_info.get('missing_keys'):
        missing_count = len(load_info['missing_keys'])
        st.warning(f"⚠️ Some model weights were not loaded ({missing_count} keys missing). This may affect performance.")
        with st.expander("🔍 Show missing keys (debug)", expanded=False):
            st.text(str(load_info['missing_keys'][:20]))  # Show first 20
    
    if load_info.get('unexpected_keys'):
        st.info(f"ℹ️ Some checkpoint keys were not used ({len(load_info['unexpected_keys'])} keys).")
        
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an eye fundus image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a fundus image for classification"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.header("🔍 Prediction Results")
    
    if uploaded_file is not None:
        try:
            transform = get_transform()
            pred_label, confidence, class_probs = predict_image(
                model, image, transform, device, id2label
            )
            
            # Display main prediction
            st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="disease-name">Predicted: {pred_label}</div>', unsafe_allow_html=True)
            st.markdown(f'**Confidence: {confidence*100:.2f}%**')
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display all class probabilities
            st.subheader("📊 Confidence Scores")
            
            # Sort by probability
            sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
            
            for disease, prob in sorted_probs:
                # Color code based on probability
                color = "#28a745" if disease == pred_label else "#6c757d"
                
                st.markdown(f"**{disease}**")
                # Progress bar
                st.progress(prob, text=f"{prob*100:.2f}%")
                st.markdown("")
            
            # Additional info
            st.markdown("---")
            st.info("ℹ️ This is a research model. For medical diagnosis, consult a healthcare professional.")
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.exception(e)
    else:
        st.info("👆 Please upload an image to see predictions")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d;'>"
    "Eye Disease Classification Demo | Built with Streamlit & PyTorch"
    "</div>",
    unsafe_allow_html=True
)

