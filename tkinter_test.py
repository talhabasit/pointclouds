import tkinter as tk
from tkinter import ttk, messagebox


# Create the main window
root = tk.Tk()
root.title("Checkbox and Buttons")


ttk.Checkbutton(root, text="Checkbox 1").pack(pady=5,padx=100)


# Start the Tkinter event loop
root.mainloop()
