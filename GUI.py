# GUI.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Listbox, MULTIPLE, END
import json
from pathlib import Path
import threading
import logging
import random

try:
    from generate_music import run_generation
except ImportError:
    messagebox.showerror("Errore", "File 'generate_music.py' non trovato. Assicurati che sia nella stessa cartella.")
    exit()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MusicGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Generatore di Musica Transformer (Dinamico)")
        self.root.geometry("800x780") # Leggermente più alto per la nuova opzione

        self.model_path = tk.StringVar()
        self.midi_vocab_path = tk.StringVar()
        self.meta_vocab_path = tk.StringVar()
        self.meta_freq_vocab_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path("./generated_midi_from_gui").resolve()))

        # NUOVO: Variabile per la checkbox della scelta casuale
        self.random_instruments_var = tk.BooleanVar(value=False)

        self.metadata_options = {}
        self.control_vars = {}
        self.combobox_widgets = {}

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        file_frame = ttk.LabelFrame(main_frame, text="1. Seleziona i File del Modello e Vocabolari", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(1, weight=1)

        self.create_file_selector("Modello (.pt):", self.model_path, self.browse_model, 0, frame=file_frame)
        self.create_file_selector("Vocabolario MIDI (.json):", self.midi_vocab_path, self.browse_midi_vocab, 1, frame=file_frame)
        self.create_file_selector("Vocabolario Metadati (.json):", self.meta_vocab_path, self.browse_meta_vocab, 2, frame=file_frame)
        self.create_file_selector("Vocabolario Frequenze (GUI):", self.meta_freq_vocab_path, self.browse_meta_freq_vocab, 3, frame=file_frame)
        
        ttk.Button(file_frame, text="Carica Vocabolari e Popola Opzioni", command=self.load_and_populate_metadata_options).grid(row=4, column=0, columnspan=2, pady=10)

        params_frame = ttk.LabelFrame(main_frame, text="2. Imposta i Parametri di Generazione", padding="10")
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        params_frame.columnconfigure(1, weight=1)

        self.create_combobox("Tonalità:", "Key", 0, params_frame)
        self.create_combobox("Tempo (BPM):", "Tempo", 1, params_frame)
        self.create_combobox("Dinamica (Avg Vel):", "AvgVel", 2, params_frame)
        self.create_combobox("Range Dinamico:", "VelRange", 3, params_frame)
        self.create_combobox("Metro:", "TimeSig", 4, params_frame)
        
        ttk.Button(params_frame, text="Seleziona Metadati Casuali", command=self.randomize_metadata).grid(row=5, column=0, columnspan=2, pady=10)

        inst_frame = ttk.LabelFrame(main_frame, text="3. Seleziona Strumenti", padding="10")
        inst_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        inst_frame.rowconfigure(1, weight=1) # Modifica per fare spazio alla checkbox
        inst_frame.columnconfigure(0, weight=1)
        
        # NUOVO: Checkbox per la selezione casuale
        random_inst_check = ttk.Checkbutton(
            inst_frame, 
            text="Scegli strumenti casualmente", 
            variable=self.random_instruments_var,
            command=self.toggle_instrument_list_state
        )
        random_inst_check.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        self.instrument_listbox = Listbox(inst_frame, selectmode=MULTIPLE, height=10, exportselection=False)
        self.instrument_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        inst_scrollbar = ttk.Scrollbar(inst_frame, orient=tk.VERTICAL, command=self.instrument_listbox.yview)
        inst_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.instrument_listbox.config(yscrollcommand=inst_scrollbar.set)
        self.instrument_listbox.insert(END, "Caricare i vocabolari...")
        
        control_frame = ttk.LabelFrame(main_frame, text="4. Finalizza e Genera", padding="10")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        control_frame.columnconfigure(1, weight=1)

        ttk.Label(control_frame, text="Lunghezza (token):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.total_tokens_var = tk.StringVar(value="1024")
        ttk.Entry(control_frame, textvariable=self.total_tokens_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(control_frame, text="Temperatura:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.temperature_var = tk.StringVar(value="0.75")
        ttk.Entry(control_frame, textvariable=self.temperature_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        self.create_file_selector("Cartella di Output:", self.output_dir, self.browse_output_dir, 2, frame=control_frame, is_dir=True)

        self.generate_button = ttk.Button(control_frame, text="Genera Musica", command=self.start_generation_thread)
        self.generate_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.status_label = ttk.Label(main_frame, text="Pronto.", wraplength=780, justify=tk.LEFT)
        self.status_label.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

    # ... (create_file_selector e create_combobox rimangono invariati) ...
    def create_file_selector(self, label_text, string_var, command, row, frame, is_dir=False):
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        entry = ttk.Label(frame, textvariable=string_var, wraplength=500, style="Path.TLabel")
        entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(frame, text="Sfoglia...", command=lambda: command(is_dir)).grid(row=row, column=2, sticky=tk.E, padx=5)

    def create_combobox(self, label_text, category_key, row, frame):
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.control_vars[category_key] = tk.StringVar()
        combo = ttk.Combobox(frame, textvariable=self.control_vars[category_key], state="readonly")
        combo.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        combo['values'] = ["Caricare un vocabolario..."]
        combo.set(combo['values'][0])
        self.combobox_widgets[category_key] = combo
    
    # NUOVA: Funzione per abilitare/disabilitare la lista strumenti
    def toggle_instrument_list_state(self):
        if self.random_instruments_var.get():
            self.instrument_listbox.config(state="disabled")
            self.instrument_listbox.selection_clear(0, END)
        else:
            self.instrument_listbox.config(state="normal")

    # ... (le funzioni di browse rimangono invariate) ...
    def browse_file(self, string_var, file_category, is_dir=False):
        if is_dir:
            path = filedialog.askdirectory(title="Seleziona una cartella")
        else:
            filetypes = [("Tutti i file", "*.*")]
            if file_category == "model":
                filetypes.insert(0, ("File Modello PyTorch", "*.pt"))
            elif file_category == "vocab":
                filetypes.insert(0, ("File Vocabolario JSON", "*.json"))
            path = filedialog.askopenfilename(title="Seleziona un file", filetypes=filetypes)
        if path:
            string_var.set(path)
            
    def browse_model(self, is_dir=False):
        self.browse_file(self.model_path, "model", is_dir)

    def browse_midi_vocab(self, is_dir=False):
        self.browse_file(self.midi_vocab_path, "vocab", is_dir)

    def browse_meta_vocab(self, is_dir=False):
        self.browse_file(self.meta_vocab_path, "vocab", is_dir)

    def browse_meta_freq_vocab(self, is_dir=False):
        self.browse_file(self.meta_freq_vocab_path, "vocab", is_dir)

    def browse_output_dir(self, is_dir=True):
        self.browse_file(self.output_dir, "directory", is_dir)

    def load_and_populate_metadata_options(self):
        meta_vocab_file = self.meta_vocab_path.get()
        meta_freq_file = self.meta_freq_vocab_path.get()

        if not meta_vocab_file or not Path(meta_vocab_file).exists():
            messagebox.showerror("Errore", "Selezionare un file di vocabolario metadati valido prima di caricarlo.")
            return

        freq_counts = {}
        if not meta_freq_file or not Path(meta_freq_file).exists():
            messagebox.showwarning("Attenzione", "File vocabolario frequenze non trovato. Le opzioni saranno ordinate alfabeticamente.")
        else:
            try:
                with open(meta_freq_file, 'r', encoding='utf-8') as f:
                    freq_data = json.load(f)
                freq_counts = freq_data.get('metadata_token_counts', {})
                if not isinstance(freq_counts, dict):
                    raise TypeError("Il file delle frequenze non contiene un dizionario 'metadata_token_counts' valido.")
            except Exception as e:
                messagebox.showerror("Errore Caricamento Frequenze", f"Impossibile leggere il file delle frequenze:\n{e}")
                return

        try:
            with open(meta_vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            token_to_id = vocab_data.get('token_to_id', {})
            if not token_to_id:
                raise ValueError("Il file JSON non contiene la chiave 'token_to_id' o è vuota.")
            
            all_tokens = list(token_to_id.keys())
            
            def sort_key(token):
                return freq_counts.get(token, -1)

            self.metadata_options = {
                "Key": sorted([t for t in all_tokens if t.startswith("Key=")], key=sort_key, reverse=True),
                "TimeSig": sorted([t for t in all_tokens if t.startswith("TimeSig=")], key=sort_key, reverse=True),
                "Tempo": sorted([t for t in all_tokens if t.startswith("Tempo_")], key=sort_key, reverse=True),
                "AvgVel": sorted([t for t in all_tokens if t.startswith("AvgVel_")], key=sort_key, reverse=True),
                "VelRange": sorted([t for t in all_tokens if t.startswith("VelRange_")], key=sort_key, reverse=True),
                "Instrument": sorted([t for t in all_tokens if t.startswith("Instrument=")], key=sort_key, reverse=True),
                "NumInst": sorted([t for t in all_tokens if t.startswith("NumInst_")], key=sort_key, reverse=True)
            }
            
            for cat_key, combo in self.combobox_widgets.items():
                values = self.metadata_options.get(cat_key, [])
                if values:
                    combo['values'] = values
                    combo.set(values[0])
                else:
                    combo['values'] = ["Nessuno trovato"]
                    combo.set("Nessuno trovato")
            
            self.instrument_listbox.delete(0, END)
            instrument_values = self.metadata_options.get("Instrument", [])
            if instrument_values:
                for item in instrument_values:
                    self.instrument_listbox.insert(END, item)
            else:
                self.instrument_listbox.insert(END, "Nessuno strumento trovato")

            messagebox.showinfo("Successo", "Opzioni dei metadati caricate e ordinate con successo.")

        except Exception as e:
            messagebox.showerror("Errore Caricamento Vocabolario", f"Impossibile leggere o analizzare il file:\n{e}")
            
    # ... (randomize_metadata rimane invariato) ...
    def randomize_metadata(self):
        if not self.metadata_options:
            messagebox.showerror("Errore", "Per favore, carica prima un vocabolario di metadati.")
            return

        for cat_key, var in self.control_vars.items():
            options = self.metadata_options.get(cat_key, [])
            if options:
                random_choice = random.choice(options)
                var.set(random_choice)

        self.instrument_listbox.selection_clear(0, END)
        all_instruments = self.metadata_options.get("Instrument", [])
        if all_instruments:
            max_instruments_to_select = min(5, len(all_instruments))
            num_to_select = random.randint(1, max_instruments_to_select)
            selected_instruments = random.sample(all_instruments, num_to_select)
            listbox_items = list(self.instrument_listbox.get(0, END))
            for instrument_name in selected_instruments:
                try:
                    idx = listbox_items.index(instrument_name)
                    self.instrument_listbox.selection_set(idx)
                except ValueError:
                    logging.warning(f"Strumento casuale '{instrument_name}' non trovato nella listbox.")
        
        messagebox.showinfo("Fatto!", "Metadati e strumenti sono stati selezionati casualmente.")

    # ... (start_generation_thread e update_status rimangono invariati) ...
    def start_generation_thread(self):
        self.generate_button.config(state="disabled")
        self.status_label.config(text="Avvio della generazione in un thread separato...")
        generation_thread = threading.Thread(target=self.generate_music)
        generation_thread.daemon = True
        generation_thread.start()

    def update_status(self, message):
        self.root.after(0, self.status_label.config, {'text': message})

    # MODIFICATA: Logica di generazione per gestire la selezione casuale
    def generate_music(self):
        try:
            paths = [self.model_path.get(), self.midi_vocab_path.get(), self.meta_vocab_path.get(), self.output_dir.get()]
            if not all(paths):
                raise ValueError("Tutti i percorsi (modello, vocabolari, output) devono essere specificati.")

            prompt = []
            for cat_key, var in self.control_vars.items():
                val = var.get()
                if val and "Caricare" not in val and "Nessuno" not in val:
                    prompt.append(val)
            
            selected_instruments = []
            all_instrument_options = self.metadata_options.get("Instrument", [])

            if self.random_instruments_var.get():
                # Se la checkbox è attiva, scegliamo noi casualmente
                if not all_instrument_options:
                    raise ValueError("Nessuno strumento disponibile nel vocabolario per la selezione casuale.")
                # Scegli da 1 a 3 strumenti
                num_to_select = random.randint(1, min(3, len(all_instrument_options)))
                selected_instruments = random.sample(all_instrument_options, num_to_select)
                self.update_status(f"Strumenti scelti casualmente: {', '.join(selected_instruments)}")
            else:
                # Altrimenti, usiamo la selezione manuale dell'utente
                selected_indices = self.instrument_listbox.curselection()
                selected_instruments = [self.instrument_listbox.get(i) for i in selected_indices]
                if not selected_instruments:
                    raise ValueError("Selezionare almeno uno strumento o spuntare 'Scegli strumenti casualmente'.")
            
            # Da qui la logica è la stessa, ma usa la lista 'selected_instruments'
            # che ora è garantito non sia vuota.
            num_inst_len = len(selected_instruments)
            num_token = ""
            if num_inst_len == 1: num_token = "NumInst_Solo"
            elif num_inst_len == 2: num_token = "NumInst_Duet"
            elif 2 < num_inst_len <= 4: num_token = "NumInst_SmallChamber"
            elif 4 < num_inst_len <= 8: num_token = "NumInst_MediumEnsemble"
            elif num_inst_len > 8: num_token = "NumInst_LargeEnsemble"
            
            if num_token and num_token in self.metadata_options.get("NumInst", []):
                 prompt.append(num_token)

            prompt.extend(selected_instruments)

            self.update_status(f"Generazione con prompt: {prompt}")

            final_message = run_generation(
                model_path=self.model_path.get(),
                midi_vocab_path=self.midi_vocab_path.get(),
                meta_vocab_path=self.meta_vocab_path.get(),
                metadata_prompt=prompt,
                output_dir=self.output_dir.get(),
                total_tokens=int(self.total_tokens_var.get()),
                temperature=float(self.temperature_var.get()),
                update_status_callback=self.update_status
            )
            messagebox.showinfo("Generazione Completata", final_message)

        except ValueError as e:
            self.update_status(f"Errore di validazione: {e}")
            messagebox.showerror("Errore di Validazione", str(e))
        except Exception as e:
            self.update_status(f"Errore imprevisto: {e}")
            logging.error("Errore durante la generazione", exc_info=True)
            messagebox.showerror("Errore di Generazione", f"Si è verificato un errore imprevisto:\n{e}")
        finally:
            self.root.after(0, self.generate_button.config, {'state': 'normal'})


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicGeneratorApp(root)
    style = ttk.Style()
    style.configure("Path.TLabel", foreground="blue", font=('TkDefaultFont', 9))
    root.mainloop()