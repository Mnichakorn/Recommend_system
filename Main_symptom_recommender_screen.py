import tkinter as tk
from tkinter import ttk, messagebox
from Main_recommend_system import rec_itemknn

def main():
    root = tk.Tk()
    root.title("Symptom Recommender (Minimal)")
    root.geometry("420x460")

    pad = {"padx": 10, "pady": 6}

    # Gender
    ttk.Label(root, text="Gender").pack(anchor="w", **pad)
    gender_var = tk.StringVar(value="male")
    gframe = ttk.Frame(root); gframe.pack(anchor="w", **pad)
    for g in ("male", "female"):
        ttk.Radiobutton(gframe, text=g.capitalize(), value=g, variable=gender_var)\
            .pack(side="left", padx=6)

    # Age (integer only)
    ttk.Label(root, text="Age").pack(anchor="w", **pad)
    age_var = tk.StringVar()
    def allow_int(P): return P == "" or P.isdigit()
    vcmd = (root.register(allow_int), "%P")
    age_entry = ttk.Entry(root, textvariable=age_var, validate="key", validatecommand=vcmd, width=5)
    age_entry.pack(anchor="w", **pad)

    # Symptoms (one per line)
    ttk.Label(root, text="Symptoms (one per line)").pack(anchor="w", **pad)
    sym_frame = ttk.Frame(root); sym_frame.pack(fill="both", expand=True, **pad)
    sym_text = tk.Text(sym_frame, height=5, wrap="word")
    sym_text.pack(side="left", fill="both", expand=True)
    scroll = ttk.Scrollbar(sym_frame, orient="vertical", command=sym_text.yview)
    scroll.pack(side="right", fill="y")
    sym_text.configure(yscrollcommand=scroll.set)

    # Output
    ttk.Label(root, text="Output").pack(anchor="w", **pad)
    out_list = tk.Listbox(root, height=5)
    out_list.pack(fill="both", expand=True, **pad)

    # Action
    def recommend():
        # read inputs
        gender = gender_var.get()
        age_s = age_var.get().strip()
        if not age_s:
            messagebox.showerror("Missing", "Please enter age."); return
        try:
            age = int(age_s)
        except ValueError:
            messagebox.showerror("Invalid", "Age must be a whole number."); return

        syms = [line.strip() for line in sym_text.get("1.0", "end").splitlines() if line.strip()]
        if not syms:
            messagebox.showerror("Missing", "Please enter at least one symptom (one per line)."); return

        # call your function
        try:
            recs = rec_itemknn(syms, gender, age, k=5, alpha=0.7)
        except Exception as e:
            messagebox.showerror("Error", f"rec_itemknn failed:\n{e}")
            return

        # show results
        out_list.delete(0, tk.END)
        if not recs:
            out_list.insert(tk.END, "No recommendations.")
        else:
            for i, r in enumerate(recs, 1):
                out_list.insert(tk.END, f"{i}. {r}")

    btn_frame = ttk.Frame(root); btn_frame.pack(fill="x", **pad)
    ttk.Button(btn_frame, text="Recommend", command=recommend).pack(side="left", padx=6)
    ttk.Button(btn_frame, text="Quit", command=root.destroy).pack(side="right", padx=6)

    age_entry.focus_set()
    root.mainloop()

if __name__ == "__main__":
    main()