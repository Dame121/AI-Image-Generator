import customtkinter as ctk
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tkinter import messagebox

# Initialize Stable Diffusion pipeline
small_model = "stabilityai/stable-diffusion-2-1"

def initialize_pipeline():
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(small_model, torch_dtype=torch.bfloat16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe = pipe.to("cuda")

def generate_image():
    prompt = prompt_entry.get()
    if not prompt:
        messagebox.showwarning("Input Error", "Please enter a prompt.")
        return

    try:
        results = pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=3.5,
            height=512,
            width=512
        )

        images = results.images
        for i, img in enumerate(images):
            img.save(f"generated_image{i}.png")  # Save each image
        messagebox.showinfo("Success", "Image generation completed! Images have been saved.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Initialize the pipeline
initialize_pipeline()

# Create the GUI
ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

app = ctk.CTk()
app.title("Stable Diffusion Image Generator")
app.geometry("500x300")

# Label for prompt
prompt_label = ctk.CTkLabel(app, text="Enter your prompt:")
prompt_label.pack(pady=10)

# Entry for prompt
prompt_entry = ctk.CTkEntry(app, width=400)
prompt_entry.pack(pady=10)

# Generate button
generate_button = ctk.CTkButton(app, text="Generate Image", command=generate_image)
generate_button.pack(pady=20)

# Run the app
app.mainloop()
