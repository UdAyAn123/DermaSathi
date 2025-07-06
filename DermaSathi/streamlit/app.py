import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image

st.set_page_config(page_title="DermaSaathi", layout="centered")
st.title("🩺 DermaSaathi: AI Skin Cancer Detection (Demo)")

# ----- Transforms -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----- Malignancy Classifier -----
class AttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, W, H = x.size()
        query = self.query_conv(x).view(batch_size, -1, W * H).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, W * H)
        value = self.value_conv(x).view(batch_size, -1, W * H)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, W, H)
        return x + out

class ResNetWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(pretrained=False)
        self.attention = AttentionLayer(self.base_model.fc.in_features)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.attention(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x

# ----- Lesion Diagnosis Model -----
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, C, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1).view(batch_size, C)
        excitation = self.fc1(squeeze)
        excitation = F.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(batch_size, C, 1, 1)
        return x * excitation

class ViTAdapter(nn.Module):
    def __init__(self, num_classes=78):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.se = SqueezeExcitation(self.vit.heads.head.in_features)
        self.vit.heads = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

# ----- Load Models -----
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mal_model = ResNetWithAttention().to(device)
    mal_model.load_state_dict(torch.load("resnet_attention_model.pth", map_location=device))
    mal_model.eval()

    lesion_model = ViTAdapter().to(device)
    lesion_model.load_state_dict(torch.load("vit_diagnosis_model.pth", map_location=device))
    lesion_model.eval()

    return mal_model, lesion_model, device

mal_model, lesion_model, device = load_models()

# ----- Upload UI -----
uploaded = st.file_uploader("📸 Upload a skin lesion image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner("Running predictions..."):
        # Malignancy prediction
        with torch.no_grad():
            mal_out = mal_model(img_tensor)
            mal_prob = F.softmax(mal_out, dim=1)[0]
            mal_label = torch.argmax(mal_prob).item()
            mal_result = ["Benign", "Malignant"][mal_label]
            mal_conf = mal_prob[mal_label].item()

        # Lesion prediction
        with torch.no_grad():
            lesion_out = lesion_model(img_tensor)
            lesion_prob = F.softmax(lesion_out, dim=1)[0]
            lesion_label = torch.argmax(lesion_prob).item()
            lesion_conf = lesion_prob[lesion_label].item()

    st.success("✅ Scan Complete")
    st.subheader("Malignancy Prediction")
    st.write(f"**{mal_result}** ({mal_conf*100:.2f}% confidence)")

    st.subheader("Lesion Type Prediction")
    st.write(f"Class Index: {lesion_label} ({lesion_conf*100:.2f}% confidence)")

    st.caption("⚠️ Note: This is a demo. Real diagnosis should be confirmed by a medical professional.")
