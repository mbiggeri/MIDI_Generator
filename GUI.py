# GUI.py (Versione aggiornata con layout scorrevole per massima compatibilità)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Listbox, MULTIPLE, END
import json
from pathlib import Path
import threading
import logging
import random

try:
    from generate_music import run_generation, get_model_info
except ImportError:
    messagebox.showerror("Errore", "File 'generate_music.py' non trovato. Assicurati che sia nella stessa cartella.")
    exit()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MusicGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Generatore di Musica Transformer")
        
        # Geometria iniziale ragionevole, ma la finestra sarà ridimensionabile
        self.root.geometry("900x800")
        self.root.minsize(800, 600) # Dimensione minima molto più flessibile

        # Variabili per i percorsi dei file
        self.model_path = tk.StringVar()
        self.midi_vocab_path = tk.StringVar()
        self.meta_vocab_path = tk.StringVar()
        self.meta_freq_vocab_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path("./generated_midi_from_gui").resolve()))
        self.primer_midi_path = tk.StringVar()
        self.profiles_path = tk.StringVar()

        # Variabili per lo stato della GUI
        self.random_instruments_var = tk.BooleanVar(value=False)
        self.generation_mode = tk.StringVar(value="Manuale")
        self.selected_profile = tk.StringVar()
        
        # Contenitori per i dati caricati
        self.metadata_options = {}
        self.profiles = []
        
        # Dizionari per gestire i widget
        self.control_vars = {}
        self.combobox_widgets = {}

        self.create_widgets()
        self.toggle_ui_mode()

    def create_widgets(self):
        # --- NUOVA STRUTTURA CON CANVAS E SCROLLBAR ---
        # 1. Creare un Canvas e una Scrollbar nella finestra principale (root)
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)

        # 2. Creare un Frame all'INTERNO del Canvas. Questo sarà il nostro contenitore principale.
        # Tutti i widget andranno dentro a 'content_frame'
        content_frame = ttk.Frame(main_canvas, padding="10")
        
        # 3. Aggiungere il frame al canvas
        main_canvas.create_window((0, 0), window=content_frame, anchor="nw")

        # 4. Funzione per aggiornare la scroll region del canvas quando la dimensione del frame cambia
        content_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        # --- FINE DELLA NUOVA STRUTTURA ---

        # Da qui in poi, tutti i widget sono aggiunti a 'content_frame' invece che a 'main_frame'
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)

        # --- Sezione 1: File ---
        file_frame = ttk.LabelFrame(content_frame, text="1. File di Configurazione", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        file_frame.columnconfigure(1, weight=1)

        self.create_file_selector("Modello (.pt):", self.model_path, self.browse_model, 0, frame=file_frame, has_analyze_button=True)
        self.create_file_selector("Vocabolario MIDI (.json):", self.midi_vocab_path, self.browse_midi_vocab, 1, frame=file_frame)
        self.create_file_selector("Vocabolario Metadati (.json):", self.meta_vocab_path, self.browse_meta_vocab, 2, frame=file_frame)
        self.create_file_selector("Vocabolario Frequenze (GUI):", self.meta_freq_vocab_path, self.browse_meta_freq_vocab, 3, frame=file_frame)
        self.create_file_selector("Primer MIDI (opzionale):", self.primer_midi_path, self.browse_primer_midi, 4, frame=file_frame)
        self.create_file_selector("File Profili Consigliati (.json):", self.profiles_path, self.browse_profiles, 5, frame=file_frame)
        ttk.Button(file_frame, text="Carica Tutti i File di Configurazione", command=self.load_all_data).grid(row=6, column=0, columnspan=3, pady=10)

        # --- Sezione Info Modello ---
        info_frame = ttk.LabelFrame(content_frame, text="Informazioni Modello", padding="10")
        info_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        info_frame.columnconfigure(0, weight=1)
        self.model_info_label = ttk.Label(info_frame, text="Nessun modello analizzato.", justify=tk.LEFT, font=('TkDefaultFont', 9))
        self.model_info_label.grid(row=0, column=0, sticky="ew")

        # --- Sezione Modalità di Generazione ---
        mode_frame = ttk.LabelFrame(content_frame, text="2. Modalità di Generazione", padding="10")
        mode_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Radiobutton(mode_frame, text="Manuale", variable=self.generation_mode, value="Manuale", command=self.toggle_ui_mode).pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Radiobutton(mode_frame, text="Guidata (Profili Consigliati)", variable=self.generation_mode, value="Guidata", command=self.toggle_ui_mode).pack(side=tk.LEFT, padx=10, pady=5)

        # --- Sezione Modalità Guidata ---
        self.profile_frame = ttk.LabelFrame(content_frame, text="3a. Modalità Guidata", padding="10")
        self.profile_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        self.profile_frame.columnconfigure(1, weight=1)

        ttk.Label(self.profile_frame, text="Seleziona Profilo:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.profile_combobox = ttk.Combobox(self.profile_frame, textvariable=self.selected_profile, state="readonly", width=80)
        self.profile_combobox.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.profile_combobox['values'] = ["Caricare un file profili..."]
        self.profile_combobox.set(self.profile_combobox['values'][0])
        self.profile_combobox.bind("<<ComboboxSelected>>", self.on_profile_select)

        # --- Sezione Modalità Manuale ---
        manual_mode_frame = ttk.Frame(content_frame)
        manual_mode_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        manual_mode_frame.columnconfigure(0, weight=1)
        manual_mode_frame.columnconfigure(1, weight=1)
        manual_mode_frame.rowconfigure(0, weight=1)

        self.manual_params_frame = ttk.LabelFrame(manual_mode_frame, text="3b. Modalità Manuale: Parametri", padding="10")
        self.manual_params_frame.grid(row=0, column=0, sticky="nsew", pady=5, padx=(0,5))
        self.manual_params_frame.columnconfigure(1, weight=1)

        self.create_combobox("Tonalità:", "Key", 0, self.manual_params_frame)
        self.create_combobox("Tempo (BPM):", "Tempo", 1, self.manual_params_frame)
        self.create_combobox("Dinamica (Avg Vel):", "AvgVel", 2, self.manual_params_frame)
        self.create_combobox("Range Dinamico:", "VelRange", 3, self.manual_params_frame)
        self.create_combobox("Metro:", "TimeSig", 4, self.manual_params_frame)
        ttk.Button(self.manual_params_frame, text="Seleziona Metadati Casuali", command=self.randomize_metadata).grid(row=5, column=0, columnspan=2, pady=10)

        self.inst_frame = ttk.LabelFrame(manual_mode_frame, text="3b. Modalità Manuale: Strumenti", padding="10")
        self.inst_frame.grid(row=0, column=1, sticky="nsew", pady=5, padx=(5,0))
        self.inst_frame.rowconfigure(1, weight=1)
        self.inst_frame.columnconfigure(0, weight=1)
        
        random_inst_check = ttk.Checkbutton(self.inst_frame, text="Scegli strumenti casualmente", variable=self.random_instruments_var, command=self.toggle_instrument_list_state)
        random_inst_check.grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.instrument_listbox = Listbox(self.inst_frame, selectmode=MULTIPLE, height=10, exportselection=False)
        self.instrument_listbox.grid(row=1, column=0, sticky="nsew")
        inst_scrollbar = ttk.Scrollbar(self.inst_frame, orient=tk.VERTICAL, command=self.instrument_listbox.yview)
        inst_scrollbar.grid(row=1, column=1, sticky="ns")
        self.instrument_listbox.config(yscrollcommand=inst_scrollbar.set)
        self.instrument_listbox.insert(END, "Caricare i vocabolari...")

        # --- Sezione 4: Controlli Finali ---
        control_frame = ttk.LabelFrame(content_frame, text="4. Finalizza e Genera", padding="10")
        control_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)
        control_frame.columnconfigure(1, weight=1)
        
        ttk.Label(control_frame, text="Lunghezza base (token):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.total_tokens_var = tk.StringVar(value="1024")
        ttk.Entry(control_frame, textvariable=self.total_tokens_var, width=10).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(control_frame, text="(token base x n° strumenti)").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        
        ttk.Label(control_frame, text="Temperatura:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.temperature_var = tk.StringVar(value="0.75")
        ttk.Entry(control_frame, textvariable=self.temperature_var, width=10).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(control_frame, text="Top-K:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.top_k_var = tk.StringVar(value="40")
        ttk.Entry(control_frame, textvariable=self.top_k_var, width=10).grid(row=2, column=1, sticky="w", padx=5)

        ttk.Label(control_frame, text="Max Rest Penalty (0=off):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.max_rest_penalty_var = tk.StringVar(value="0.0")
        ttk.Entry(control_frame, textvariable=self.max_rest_penalty_var, width=10).grid(row=3, column=1, sticky="w", padx=5)
        self.create_file_selector("Cartella di Output:", self.output_dir, self.browse_output_dir, 4, frame=control_frame, is_dir=True)

        self.generate_button = ttk.Button(content_frame, text="Genera Musica", command=self.start_generation_thread)
        self.generate_button.grid(row=6, column=0, columnspan=2, pady=10)
        
        self.progress_bar = ttk.Progressbar(content_frame, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(5, 0), padx=20)

        self.status_label = ttk.Label(content_frame, text="Pronto. Selezionare i file di configurazione e caricarli.", wraplength=830, justify=tk.LEFT)
        self.status_label.grid(row=8, column=0, columnspan=2, sticky="ew", pady=5)
    
    # ... (tutte le altre funzioni da create_file_selector in poi rimangono identiche) ...
    def create_file_selector(self, label_text, string_var, command, row, frame, is_dir=False, has_analyze_button=False):
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        entry = ttk.Label(frame, textvariable=string_var, wraplength=550, style="Path.TLabel")
        entry.grid(row=row, column=1, sticky="ew", padx=5)
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=row, column=2, sticky="e")
        ttk.Button(button_frame, text="Sfoglia...", command=lambda: command(is_dir)).pack(side=tk.LEFT, padx=(0, 5))
        if has_analyze_button:
            ttk.Button(button_frame, text="Analizza", command=self.analyze_model).pack(side=tk.LEFT)
    
    def create_combobox(self, label_text, category_key, row, frame):
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.control_vars[category_key] = tk.StringVar()
        combo = ttk.Combobox(frame, textvariable=self.control_vars[category_key], state="readonly")
        combo.grid(row=row, column=1, sticky="ew", padx=5)
        combo['values'] = ["Caricare un vocabolario..."]
        combo.set(combo['values'][0])
        self.combobox_widgets[category_key] = combo
    
    def browse_file(self, string_var, file_category, is_dir=False):
        if is_dir:
            path = filedialog.askdirectory(title="Seleziona una cartella")
        else:
            filetypes = [("Tutti i file", "*.*")]
            if file_category == "model":
                filetypes.insert(0, ("File Modello PyTorch", "*.pt"))
            elif file_category == "vocab":
                filetypes.insert(0, ("File JSON", "*.json"))
            elif file_category == "primer":
                filetypes.insert(0, ("File MIDI", "*.mid;*.midi"))
            path = filedialog.askopenfilename(title="Seleziona un file", filetypes=filetypes)
        if path:
            string_var.set(path)
            
    def browse_model(self, is_dir=False): self.browse_file(self.model_path, "model", is_dir)
    def browse_midi_vocab(self, is_dir=False): self.browse_file(self.midi_vocab_path, "vocab", is_dir)
    def browse_meta_vocab(self, is_dir=False): self.browse_file(self.meta_vocab_path, "vocab", is_dir)
    def browse_meta_freq_vocab(self, is_dir=False): self.browse_file(self.meta_freq_vocab_path, "vocab", is_dir)
    def browse_output_dir(self, is_dir=True): self.browse_file(self.output_dir, "directory", is_dir)
    def browse_primer_midi(self, is_dir=False): self.browse_file(self.primer_midi_path, "primer", is_dir)
    def browse_profiles(self, is_dir=False): self.browse_file(self.profiles_path, "vocab", is_dir)

    def load_all_data(self):
        self.load_and_populate_metadata_options() 
        self.load_profiles()

    def load_profiles(self):
        profiles_file = self.profiles_path.get()
        if not profiles_file or not Path(profiles_file).exists():
            messagebox.showwarning("Attenzione Profili", "File dei profili non selezionato o non trovato. La modalità guidata non sarà disponibile.")
            self.profile_combobox['values'] = ["File profili non caricato"]
            self.profile_combobox.set(self.profile_combobox['values'][0])
            self.profiles = []
            self.toggle_ui_mode()
            return
        try:
            with open(profiles_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.profiles = data.get("profiles", [])
            
            if not self.profiles:
                messagebox.showerror("Errore Profili", "Nessun profilo trovato nel file.")
                return

            profile_names = [p['profile_name'] for p in self.profiles]
            self.profile_combobox['values'] = profile_names
            if profile_names:
                self.profile_combobox.set(profile_names[0])
                self.on_profile_select()
            
            self.update_status(f"Caricati {len(self.profiles)} profili con successo.")
        except Exception as e:
            messagebox.showerror("Errore Caricamento Profili", f"Impossibile leggere il file:\n{e}")
            self.profiles = []
        finally:
            self.toggle_ui_mode()

    def toggle_ui_mode(self):
        mode = self.generation_mode.get()
        is_manual = mode == "Manuale"
        
        manual_state = tk.NORMAL if is_manual else tk.DISABLED
        listbox_state = tk.NORMAL if is_manual else tk.DISABLED
        
        for combo in self.combobox_widgets.values():
            combo.config(state="readonly" if is_manual else "disabled")
        
        if hasattr(self, 'instrument_listbox'):
            self.instrument_listbox.config(state=listbox_state)

        for child in self.manual_params_frame.winfo_children():
            if isinstance(child, (ttk.Button, ttk.Checkbutton)):
                 child.config(state=manual_state)
        for child in self.inst_frame.winfo_children():
            if isinstance(child, (ttk.Button, ttk.Checkbutton)):
                 child.config(state=manual_state)

        guided_state = "readonly" if not is_manual and self.profiles else "disabled"
        self.profile_combobox.config(state=guided_state)

    def on_profile_select(self, event=None):
        profile_name = self.selected_profile.get()
        profile_data = next((p for p in self.profiles if p['profile_name'] == profile_name), None)
        if not profile_data: return

        key_map = {
            "recommended_key": "Key", "recommended_timesig": "TimeSig",
            "recommended_tempo": "Tempo", "recommended_avg_vel": "AvgVel",
            "recommended_vel_range": "VelRange"
        }
        for p_key, v_key in key_map.items():
            value = profile_data.get(p_key)
            if value and v_key in self.control_vars and self.combobox_widgets[v_key]['values']:
                if value in self.combobox_widgets[v_key]['values']:
                    self.control_vars[v_key].set(value)
                else:
                    self.control_vars[v_key].set(self.combobox_widgets[v_key]['values'][0])
        
        if hasattr(self, 'instrument_listbox'):
            self.instrument_listbox.selection_clear(0, END)
            listbox_items = list(self.instrument_listbox.get(0, END))
            for inst_token in profile_data.get("instruments", []):
                try:
                    idx = listbox_items.index(inst_token)
                    self.instrument_listbox.selection_set(idx)
                except (ValueError, tk.TclError): pass

    def generate_music(self):
        try:
            self.update_progress(0)
            paths = [self.model_path.get(), self.midi_vocab_path.get(), self.meta_vocab_path.get(), self.output_dir.get()]
            if not all(paths): raise ValueError("Tutti i percorsi (modello, vocabolari, output) devono essere specificati.")

            prompt, num_inst_len = [], 0
            mode = self.generation_mode.get()

            if mode == "Guidata":
                self.update_status("Modalità Guidata: costruzione prompt dal profilo selezionato...")
                profile_name = self.selected_profile.get()
                if not profile_name or "Caricare" in profile_name or "non caricato" in profile_name:
                    raise ValueError("Selezionare un profilo valido dalla lista.")
                
                selected_profile_data = next((p for p in self.profiles if p['profile_name'] == profile_name), None)
                if not selected_profile_data: raise ValueError(f"Dati per il profilo '{profile_name}' non trovati.")

                prompt.extend(selected_profile_data.get("instruments", []))
                recommended_tokens = [
                    selected_profile_data.get("recommended_key"), selected_profile_data.get("recommended_timesig"),
                    selected_profile_data.get("recommended_tempo"), selected_profile_data.get("recommended_avg_vel"),
                    selected_profile_data.get("recommended_vel_range"), selected_profile_data.get("recommended_num_inst")]
                prompt.extend([token for token in recommended_tokens if token])
                num_inst_len = len(selected_profile_data.get("instruments", []))
            else:
                self.update_status("Modalità Manuale: costruzione prompt dai controlli...")
                prompt = [var.get() for var in self.control_vars.values() if var.get() and "Caricare" not in var.get() and "Nessuno" not in var.get()]
                
                selected_instruments = []
                if self.random_instruments_var.get():
                    all_instrument_options = self.metadata_options.get("Instrument", [])
                    if not all_instrument_options: raise ValueError("Nessuno strumento disponibile per la selezione casuale.")
                    num_to_select = random.randint(1, min(3, len(all_instrument_options)))
                    selected_instruments = random.sample(all_instrument_options, num_to_select)
                else:
                    selected_instruments = [self.instrument_listbox.get(i) for i in self.instrument_listbox.curselection()]
                
                if not selected_instruments: raise ValueError("Selezionare almeno uno strumento o spuntare 'Scegli casualmente'.")
                
                num_inst_len = len(selected_instruments)
                num_map = {1: "NumInst_Solo", 2: "NumInst_Duet"}
                num_token = num_map.get(num_inst_len)
                if not num_token:
                    if 2 < num_inst_len <= 4: num_token = "NumInst_SmallChamber"
                    elif 4 < num_inst_len <= 8: num_token = "NumInst_MediumEnsemble"
                    else: num_token = "NumInst_LargeEnsemble"
                if num_token and num_token in self.metadata_options.get("NumInst", []): prompt.append(num_token)
                prompt.extend(selected_instruments)

            if num_inst_len == 0: raise ValueError("Il prompt non contiene strumenti. Selezionarne almeno uno.")

            base_tokens = int(self.total_tokens_var.get())
            final_tokens = base_tokens * num_inst_len
            self.update_status(f"Budget token: {base_tokens} x {num_inst_len} strumenti = {final_tokens} token totali.")

            final_message = run_generation(
                model_path=self.model_path.get(), midi_vocab_path=self.midi_vocab_path.get(),
                meta_vocab_path=self.meta_vocab_path.get(), metadata_prompt=prompt,
                output_dir=self.output_dir.get(), total_tokens=final_tokens,
                temperature=float(self.temperature_var.get()), top_k=int(self.top_k_var.get()),
                max_rest_penalty=float(self.max_rest_penalty_var.get()), primer_midi_path=self.primer_midi_path.get(),
                update_status_callback=self.update_status, progress_callback=self.update_progress)
            
            messagebox.showinfo("Generazione Completata", final_message)
        except (ValueError, RuntimeError) as e:
            self.update_status(f"Errore: {e}")
            messagebox.showerror("Errore di Configurazione", str(e))
        except Exception as e:
            self.update_status(f"Errore imprevisto: {e}")
            logging.error("Errore durante la generazione", exc_info=True)
            messagebox.showerror("Errore Imprevisto", f"Si è verificato un errore:\n{e}")
        finally:
            self.update_progress(0)
            self.root.after(0, self.generate_button.config, {'state': 'normal'})
    
    def update_progress(self, value):
        self.root.after(0, self.progress_bar.config, {'value': value})

    def load_and_populate_metadata_options(self):
        meta_vocab_file = self.meta_vocab_path.get()
        meta_freq_file = self.meta_freq_vocab_path.get()
        if not meta_vocab_file or not Path(meta_vocab_file).exists():
            messagebox.showerror("Errore", "Selezionare un file di vocabolario metadati valido.")
            return
        freq_counts = {}
        if meta_freq_file and Path(meta_freq_file).exists():
            try:
                with open(meta_freq_file, 'r', encoding='utf-8') as f: freq_counts = json.load(f).get('metadata_token_counts', {})
            except Exception as e: messagebox.showerror("Errore Frequenze", f"Impossibile leggere file frequenze:\n{e}")
        else:
            messagebox.showwarning("Attenzione", "File vocabolario frequenze non trovato.")
        try:
            with open(meta_vocab_file, 'r', encoding='utf-8') as f: token_to_id = json.load(f).get('token_to_id', {})
            all_tokens = list(token_to_id.keys())
            def sort_key(token): return freq_counts.get(token, -1)
            self.metadata_options = {
                "Key": sorted([t for t in all_tokens if t.startswith("Key=")], key=sort_key, reverse=True),
                "TimeSig": sorted([t for t in all_tokens if t.startswith("TimeSig=")], key=sort_key, reverse=True),
                "Tempo": sorted([t for t in all_tokens if t.startswith("Tempo_")], key=sort_key, reverse=True),
                "AvgVel": sorted([t for t in all_tokens if t.startswith("AvgVel_")], key=sort_key, reverse=True),
                "VelRange": sorted([t for t in all_tokens if t.startswith("VelRange_")], key=sort_key, reverse=True),
                "Instrument": sorted([t for t in all_tokens if t.startswith("Instrument=")], key=sort_key, reverse=True),
                "NumInst": sorted([t for t in all_tokens if t.startswith("NumInst_")], key=sort_key, reverse=True)}
            for cat_key, combo in self.combobox_widgets.items():
                values = self.metadata_options.get(cat_key, ["Nessuno trovato"])
                combo['values'] = values if values else ["Nessuno trovato"]
                if values: combo.set(values[0])
            self.instrument_listbox.delete(0, END)
            instrument_values = self.metadata_options.get("Instrument", [])
            if instrument_values: [self.instrument_listbox.insert(END, item) for item in instrument_values]
            else: self.instrument_listbox.insert(END, "Nessuno strumento trovato")
            self.update_status("Opzioni dei metadati caricate con successo.")
        except Exception as e: messagebox.showerror("Errore Vocabolario", f"Impossibile leggere file:\n{e}")

    def toggle_instrument_list_state(self):
        if self.random_instruments_var.get():
            self.instrument_listbox.config(state=tk.DISABLED)
            self.instrument_listbox.selection_clear(0, END)
        else:
            self.instrument_listbox.config(state=tk.NORMAL)

    def randomize_metadata(self):
        if self.generation_mode.get() == "Guidata":
            messagebox.showinfo("Info", "La selezione casuale è disponibile solo in Modalità Manuale.")
            return
        if not self.metadata_options:
            messagebox.showerror("Errore", "Per favore, carica prima un vocabolario di metadati.")
            return
        for cat_key, var in self.control_vars.items():
            options = self.metadata_options.get(cat_key, [])
            if options: var.set(random.choice(options))
        self.instrument_listbox.selection_clear(0, END)
        all_instruments = self.metadata_options.get("Instrument", [])
        if all_instruments:
            num_to_select = random.randint(1, min(5, len(all_instruments)))
            selected_instruments = random.sample(all_instruments, num_to_select)
            listbox_items = list(self.instrument_listbox.get(0, END))
            for name in selected_instruments:
                try: self.instrument_listbox.selection_set(listbox_items.index(name))
                except (ValueError, tk.TclError): pass
        self.update_status("Metadati e strumenti selezionati casualmente.")

    def start_generation_thread(self):
        self.generate_button.config(state=tk.DISABLED)
        self.update_progress(0)
        self.status_label.config(text="Avvio della generazione in un thread separato...")
        threading.Thread(target=self.generate_music, daemon=True).start()

    def update_status(self, message):
        self.root.after(0, self.status_label.config, {'text': message})

    def analyze_model(self):
        model_path = self.model_path.get()
        if not model_path:
            messagebox.showerror("Errore", "Per favore, prima seleziona un file modello (.pt).")
            return
        self.status_label.config(text="Analisi del modello in corso... Attendere.")
        self.root.update_idletasks()
        info_dict = get_model_info(model_path)
        if "error" in info_dict:
            messagebox.showerror("Errore Analisi Modello", info_dict["error"])
            self.model_info_label.config(text="Analisi fallita.")
            self.status_label.config(text="Errore.")
        else:
            info_text = "\n".join([f"{key}: {value}" for key, value in info_dict.items()])
            self.model_info_label.config(text=info_text)
            self.status_label.config(text="Informazioni modello caricate.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MusicGeneratorApp(root)
    style = ttk.Style()
    # Stili specifici per macOS per migliorare la leggibilità
    if root.tk.call('tk', 'windowingsystem') == 'aqua':
        style.configure("TLabel", font=("Helvetica", 12))
        style.configure("TButton", font=("Helvetica", 12))
        style.configure("TCombobox", font=("Helvetica", 12))
        style.configure("TRadiobutton", font=("Helvetica", 12))
        style.configure("TCheckbutton", font=("Helvetica", 12))
        style.configure("TEntry", font=("Helvetica", 12))
        style.configure("TLabelFrame.Label", font=("Helvetica", 13, "bold"))

    style.configure("Path.TLabel", foreground="blue")
    root.mainloop()
