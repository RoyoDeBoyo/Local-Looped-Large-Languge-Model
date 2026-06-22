import tkinter as tk
from tkinter import ttk

def launch_input_gui(input_queue):
    def send_prompt(event=None):
        user_text = entry.get().strip()
        if user_text:
            input_queue.put(user_text)
            entry.delete(0, tk.END)
            
    root = tk.Tk()
    root.title("Agent Control Panel")
    root.geometry("400x100")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding="10")
    frame.pack(fill=tk.BOTH, expand=True)
    label = ttk.Label(frame, text="Inject Prompt to Agent:")
    label.pack(anchor=tk.W, pady=(0, 5))
    entry = ttk.Entry(frame, width=50)
    entry.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 5))
    entry.bind("<Return>", send_prompt)
    send_btn = ttk.Button(frame, text="Send", command=send_prompt)
    send_btn.pack(side=tk.RIGHT)
    root.mainloop()
