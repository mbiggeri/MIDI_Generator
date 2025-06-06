# GUI.py (versione dinamica e flessibile)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Listbox, MULTIPLE, END
import json
from pathlib import Path
import threading
import logging

# Importa la funzione di generazione dal file refattorizzato
# Assicurati che generate_music.py sia nella stessa cartella o nel python path
try:
    from generate_music import run_generation
except ImportError:
    messagebox.showerror("Errore", "File 'generate_music.py' non trovato. Assicurati che sia nella stessa cartella.")
    exit()

# Setup del logging di base
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MusicGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Generatore di Musica Transformer (Dinamico)")
        self.root.geometry("800x750")

        self.model_path = tk.StringVar()
        self.midi_vocab_path = tk.StringVar()
        self.meta_vocab_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path("./generated_midi_from_gui").resolve()))

        self.metadata_options = {} # Dizionario per contenere le opzioni caricate

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Sezione Selezione File ---
        file_frame = ttk.LabelFrame(main_frame, text="1. Seleziona i File del Modello", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(1, weight=1)

        self.create_file_selector("Modello (.pt):", self.model_path, self.browse_model, 0)
        self.create_file_selector("Vocabolario MIDI (.json):", self.midi_vocab_path, self.browse_midi_vocab, 1)
        self.create_file_selector("Vocabolario Metadati (.json):", self.meta_vocab_path, self.browse_meta_vocab, 2)
        
        ttk.Button(file_frame, text="Carica Vocabolari e Popola Opzioni", command=self.load_and_populate_metadata_options).grid(row=3, column=0, columnspan=2, pady=10)

        # --- Sezione Parametri Generazione ---
        params_frame = ttk.LabelFrame(main_frame, text="2. Imposta i Parametri di Generazione", padding="10")
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        params_frame.columnconfigure(1, weight=1)

        self.control_vars = {}
        self.create_combobox("Tonalità:", "Key", 0, params_frame)
        self.create_combobox("Tempo (BPM):", "Tempo", 1, params_frame)
        self.create_combobox("Dinamica (Avg Vel):", "AvgVel", 2, params_frame)
        self.create_combobox("Range Dinamico:", "VelRange", 3, params_frame)
        self.create_combobox("Metro:", "TimeSig", 4, params_frame)
        
        # --- Sezione Strumenti (Listbox per selezione multipla) ---
        inst_frame = ttk.LabelFrame(main_frame, text="3. Seleziona Strumenti", padding="10")
        inst_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        inst_frame.rowconfigure(0, weight=1)
        inst_frame.columnconfigure(0, weight=1)

        self.instrument_listbox = Listbox(inst_frame, selectmode=MULTIPLE, height=10, exportselection=False)
        self.instrument_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        inst_scrollbar = ttk.Scrollbar(inst_frame, orient=tk.VERTICAL, command=self.instrument_listbox.yview)
        inst_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.instrument_listbox.config(yscrollcommand=inst_scrollbar.set)
        self.instrument_listbox.insert(END, "Caricare un vocabolario metadati...")

        # --- Sezione Controlli Finali ---
        control_frame = ttk.LabelFrame(main_frame, text="4. Finalizza e Genera", padding="10")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        control_frame.columnconfigure(1, weight=1)

        ttk.Label(control_frame, text="Lunghezza (token):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.total_tokens_var = tk.StringVar(value="1024")
        ttk.Entry(control_frame, textvariable=self.total_tokens_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(control_frame, text="Temperatura:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.temperature_var = tk.StringVar(value="0.75")
        ttk.Entry(control_frame, textvariable=self.temperature_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        self.create_file_selector("Cartella di Output:", self.output_dir, self.browse_output_dir, 2, control_frame, is_dir=True)

        self.generate_button = ttk.Button(control_frame, text="Genera Musica", command=self.start_generation_thread)
        self.generate_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.status_label = ttk.Label(main_frame, text="Pronto.", wraplength=780, justify=tk.LEFT)
        self.status_label.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

    def create_file_selector(self, label_text, string_var, command, row, frame, is_dir=False):
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        entry = ttk.Label(frame, textvariable=string_var, wraplength=500, style="Path.TLabel") # Mostra il percorso
        entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(frame, text="Sfoglia...", command=lambda: command(is_dir)).grid(row=row, column=2, sticky=tk.E, padx=5)

    def create_combobox(self, label_text, category_key, row, frame):
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.control_vars[category_key] = tk.StringVar()
        combo = ttk.Combobox(frame, textvariable=self.control_vars[category_key], state="readonly")
        combo.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        combo['values'] = ["Caricare un vocabolario..."]
        combo.set(combo['values'][0])

    def browse_file(self, string_var, is_dir):
        if is_dir:
            path = filedialog.askdirectory(title="Seleziona una cartella")
        else:
            filetypes = [("Tutti i file", "*.*")]
            if "model" in string_var.name:
                filetypes = [("File Modello PyTorch", "*.pt")] + filetypes
            elif "vocab" in string_var.name:
                filetypes = [("File Vocabolario JSON", "*.json")] + filetypes
            path = filedialog.askopenfilename(title="Seleziona un file", filetypes=filetypes)
        
        if path:
            string_var.set(path)
            
    def browse_model(self, is_dir=False): self.browse_file(self.model_path, is_dir)
    def browse_midi_vocab(self, is_dir=False): self.browse_file(self.midi_vocab_path, is_dir)
    def browse_meta_vocab(self, is_dir=False): self.browse_file(self.meta_vocab_path, is_dir)
    def browse_output_dir(self, is_dir=True): self.browse_file(self.output_dir, is_dir)

    def load_and_populate_metadata_options(self):
        meta_vocab_file = self.meta_vocab_path.get()
        if not meta_vocab_file or not Path(meta_vocab_file).exists():
            messagebox.showerror("Errore", "Selezionare un file di vocabolario metadati valido prima di caricarlo.")
            return

        try:
            with open(meta_vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            token_to_id = vocab_data.get('token_to_id', {})
            if not token_to_id:
                raise ValueError("Il file JSON non contiene la chiave 'token_to_id' o è vuota.")
            
            all_tokens = list(token_to_id.keys())
            
            self.metadata_options = {
                "Key": sorted([t for t in all_tokens if t.startswith("Key=")]),
                "TimeSig": sorted([t for t in all_tokens if t.startswith("TimeSig=")]),
                "Tempo": sorted([t for t in all_tokens if t.startswith("Tempo_")]),
                "AvgVel": sorted([t for t in all_tokens if t.startswith("AvgVel_")]),
                "VelRange": sorted([t for t in all_tokens if t.startswith("VelRange_")]),
                "Instrument": sorted([t for t in all_tokens if t.startswith("Instrument=")]),
                "NumInst": sorted([t for t in all_tokens if t.startswith("NumInst_")])
            }
            
            # Popola Combobox
            for cat_key, var in self.control_vars.items():
                combo = var.trace_vinfo()[0][0] # Ottieni il widget Combobox
                values = self.metadata_options.get(cat_key, [])
                if values:
                    combo['values'] = values
                    combo.set(values[0])
                else:
                    combo['values'] = ["Nessuno trovato"]
                    combo.set("Nessuno trovato")
            
            # Popola Listbox strumenti
            self.instrument_listbox.delete(0, END)
            instrument_values = self.metadata_options.get("Instrument", [])
            if instrument_values:
                for item in instrument_values:
                    self.instrument_listbox.insert(END, item)
            else:
                self.instrument_listbox.insert(END, "Nessuno strumento trovato")

            messagebox.showinfo("Successo", "Opzioni dei metadati caricate con successo dal vocabolario.")

        except Exception as e:
            messagebox.showerror("Errore Caricamento Vocabolario", f"Impossibile leggere o analizzare il file:\n{e}")

    def start_generation_thread(self):
        # Disabilita il pulsante per evitare click multipli
        self.generate_button.config(state="disabled")
        self.status_label.config(text="Avvio della generazione in un thread separato...")
        # Crea e avvia il thread
        generation_thread = threading.Thread(target=self.generate_music)
        generation_thread.daemon = True # Permette alla GUI di chiudersi anche se il thread è in esecuzione
        generation_thread.start()

    def update_status(self, message):
        self.root.after(0, self.status_label.config, {'text': message})

    def generate_music(self):
        try:
            # 1. Validazione input
            paths = [self.model_path.get(), self.midi_vocab_path.get(), self.meta_vocab_path.get(), self.output_dir.get()]
            if not all(paths):
                raise ValueError("Tutti i percorsi (modello, vocabolari, output) devono essere specificati.")

            # 2. Costruzione del prompt
            prompt = []
            for cat_key, var in self.control_vars.items():
                val = var.get()
                if val and "Caricare" not in val and "Nessuno" not in val:
                    prompt.append(val)
            
            selected_indices = self.instrument_listbox.curselection()
            selected_instruments = [self.instrument_listbox.get(i) for i in selected_indices]
            
            if not selected_instruments:
                raise ValueError("Selezionare almeno uno strumento.")
                
            # Aggiungi il token per il numero di strumenti
            num_inst_map = {1: "NumInst_Solo", 2: "NumInst_Duet"} # Semplificato
            num_token = num_inst_map.get(len(selected_instruments))
            if not num_token and len(selected_instruments) > 2:
                num_token = "NumInst_LargeEnsemble" # Fallback generico
            
            if num_token and num_token in self.metadata_options.get("NumInst", []):
                 prompt.append(num_token)

            prompt.extend(selected_instruments)

            # 3. Chiamata alla funzione di generazione
            run_generation(
                model_path=self.model_path.get(),
                midi_vocab_path=self.midi_vocab_path.get(),
                meta_vocab_path=self.meta_vocab_path.get(),
                metadata_prompt=prompt,
                output_dir=self.output_dir.get(),
                total_tokens=int(self.total_tokens_var.get()),
                temperature=float(self.temperature_var.get()),
                update_status_callback=self.update_status
            )
        except ValueError as e:
            messagebox.showerror("Errore di Validazione", str(e))
        except Exception as e:
            messagebox.showerror("Errore di Generazione", f"Si è verificato un errore imprevisto:\n{e}")
        finally:
            # Riabilita il pulsante alla fine del thread
            self.generate_button.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicGeneratorApp(root)
    # Aggiungi uno stile per l'etichetta del percorso per una migliore visualizzazione
    style = ttk.Style()
    style.configure("Path.TLabel", foreground="blue", font=('TkDefaultFont', 9))
    root.mainloop()