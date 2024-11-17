model_name="microsoft/OmniParser"

from transformers import AutoModel

def load_model(model_name):
    try:
        # 使用from_pretrained加载指定的预训练模型
        model = AutoModel.from_pretrained(model_name)
        print(f"Successfully loaded model: {model_name}")
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None