from transformers import pipeline
import gradio as gr

# Load the model
model = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

# Default example text
example_text = "The Prime Minister announced a new healthcare plan today."

# Inference function
def analyze_news(text_input, file_input, clear=False):
    if clear:
        return "", gr.update(value="")

    results = []

    # File uploaded
    if file_input is not None:
        try:
            content = file_input.read().decode("utf-8").strip().splitlines()
            for idx, line in enumerate(content, 1):
                pred = model(line)[0]
                label = pred["label"]
                score = pred["score"]
                if label == "NEGATIVE":
                    results.append(f"🛑 Article {idx}: **FAKE** ({score:.2%})")
                else:
                    results.append(f"✅ Article {idx}: **REAL** ({score:.2%})")
            return "\n".join(results), gr.update()
        except Exception as e:
            return f"❌ Error reading file: {e}", gr.update()

    # Text input
    elif text_input.strip():
        pred = model(text_input)[0]
        label = pred["label"]
        score = pred["score"]
        if label == "NEGATIVE":
            return f"🛑 Possibly **FAKE** ({score:.2%} confidence)", gr.update()
        else:
            return f"✅ Possibly **REAL** ({score:.2%} confidence)", gr.update()
    
    # No input
    else:
        return "⚠️ Please enter text or upload a file.", gr.update()

# Inputs
text_box = gr.Textbox(label="✍️ Enter news headline or article here:", 
                      placeholder="Type or paste here...", 
                      lines=6, 
                      value=example_text)

file_upload = gr.File(label="📂 Or upload a .txt file with one article per line", 
                      file_types=[".txt"])

output = gr.Textbox(label="🧾 Results", lines=10)

# Interface
with gr.Blocks(title="📰 Fake News Detector") as demo:
    gr.Markdown("## 📰 Fake News Detector")
    gr.Markdown("Classify news as **real or fake** using a transformer model (demo).")

    with gr.Row():
        analyze_btn = gr.Button("🚀 Analyze")
        clear_btn = gr.Button("🔄 Clear")

    with gr.Row():
        text_box.render()
        file_upload.render()

    output.render()

    analyze_btn.click(analyze_news, inputs=[text_box, file_upload, gr.State(False)], outputs=[output, text_box])
    clear_btn.click(analyze_news, inputs=[text_box, file_upload, gr.State(True)], outputs=[output, text_box])

# Run
if __name__ == "__main__":
    demo.launch()