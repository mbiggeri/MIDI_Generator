import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
import torch.nn as nn
import torch.nn.functional as F
import miditok
from pathlib import Path
import json
import math
import logging
import time
import sys
from symusic import Score # Importa Score direttamente da symusic
import random
import threading

# --- Configurazione / Costanti Essenziali (dallo script originale) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODIFICA QUESTI PERCORSI SE NECESSARIO ===
PATH_MODELLO_CHECKPOINT = Path(r"C:\Users\Michael\Desktop\ModelliMusicGenerator\ModelloPianoTrasposto\transformer_periodic_epoch81_valloss1.5116_20250529-215034.pt")
PATH_VOCAB_MIDI = Path(r"C:\Users\Michael\Desktop\PianoGenerator\ModelloPianoTrasposto\midi_vocab.json")
PATH_VOCAB_METADATA = Path(r"C:\Users\Michael\Desktop\PianoGenerator\ModelloPianoTrasposto\metadata_vocab.json")
GENERATION_OUTPUT_DIR = Path("./generated_midi_inference_gui")
# ================================

# Token Speciali (dallo script originale)
MIDI_TOKENIZER_STRATEGY = miditok.REMI
MIDI_PAD_TOKEN_NAME = "PAD_None"
MIDI_SOS_TOKEN_NAME = "SOS_None"
MIDI_EOS_TOKEN_NAME = "EOS_None"
MIDI_UNK_TOKEN_NAME = "UNK_None"
META_PAD_TOKEN_NAME = "<pad_meta>"
META_UNK_TOKEN_NAME = "<unk_meta>"
META_SOS_TOKEN_NAME = "<sos_meta>"
META_EOS_TOKEN_NAME = "<eos_meta>"

MAX_SEQ_LEN_MIDI = 1024 # Usato come fallback se non nel checkpoint
MAX_SEQ_LEN_META = 128
PRIMER_TOKEN_COUNT = 50
MIN_TOKENS_PER_CHUNK = 100
DEFAULT_TOTAL_MIDI_TARGET_LENGTH = 2048 
DEFAULT_TEMPERATURE = 0.75 

# --- Logger per la GUI ---
class GuiLogger(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.configure(state='disabled')
        self.text_widget.see(tk.END)

# --- Funzioni di Generazione e Classi Modello (adattate dallo script originale) ---
def load_midi_tokenizer_for_inference(vocab_path):
    logging.info(f"Caricamento tokenizer MIDI da {vocab_path}")
    if not Path(vocab_path).exists():
        logging.error(f"File vocabolario MIDI non trovato: {vocab_path}")
        raise FileNotFoundError(f"File vocabolario MIDI non trovato: {vocab_path}")
    try:
        tokenizer = MIDI_TOKENIZER_STRATEGY(params=str(vocab_path))
        logging.info(f"Tokenizer MIDI caricato con successo. Strategia: {MIDI_TOKENIZER_STRATEGY.__name__}")
        for token_name in [MIDI_PAD_TOKEN_NAME, MIDI_SOS_TOKEN_NAME, MIDI_EOS_TOKEN_NAME, MIDI_UNK_TOKEN_NAME]:
            _ = tokenizer[token_name]
        return tokenizer
    except Exception as e:
        logging.error(f"Errore nel caricare il tokenizer MIDI da {vocab_path}: {e}", exc_info=True)
        raise

def load_metadata_vocab_for_inference(vocab_path):
    logging.info(f"Caricamento vocabolario Metadati da {vocab_path}")
    if not Path(vocab_path).exists():
        logging.error(f"File vocabolario Metadati non trovato: {vocab_path}")
        raise FileNotFoundError(f"File vocabolario Metadati non trovato: {vocab_path}")
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        token_to_id = vocab_data['token_to_id']
        for token_name in [META_PAD_TOKEN_NAME, META_UNK_TOKEN_NAME, META_SOS_TOKEN_NAME, META_EOS_TOKEN_NAME]:
            if token_name not in token_to_id:
                logging.error(f"ERRORE CRITICO: Token speciale Metadati '{token_name}' non trovato.")
                raise ValueError(f"Token speciale Metadati '{token_name}' non trovato.")
        return token_to_id, vocab_data.get('id_to_token', {v: k for k, v in token_to_id.items()})
    except Exception as e:
        logging.error(f"Errore nel caricare il vocabolario Metadati da {vocab_path}: {e}", exc_info=True)
        raise

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_size, max_pe_len,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_pe_len)
        self.transformer = nn.Transformer(
            d_model=emb_size, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, memory_key_padding_mask=None, tgt_mask=None):
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))
        if tgt_mask is None:
             tgt_len = tgt.size(1)
             tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool), diagonal=1)
        if memory_key_padding_mask is None:
             memory_key_padding_mask = src_padding_mask
        outs = self.transformer(src_emb, tgt_emb,
                                tgt_mask=tgt_mask,
                                src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))
        return self.transformer.decoder(tgt_emb, memory,
                                         tgt_mask=tgt_mask,
                                         tgt_key_padding_mask=tgt_padding_mask,
                                         memory_key_padding_mask=memory_key_padding_mask)

def generate_sequence(model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                      max_new_tokens, min_new_tokens, temperature=DEFAULT_TEMPERATURE, 
                      top_k=None, device=DEVICE, primer_token_ids=None, model_max_pe_len=5000):
    model.eval()
    try:
        sos_meta_id = metadata_vocab_map[META_SOS_TOKEN_NAME]
        eos_meta_id = metadata_vocab_map[META_EOS_TOKEN_NAME]
        unk_meta_id = metadata_vocab_map[META_UNK_TOKEN_NAME]
        meta_pad_id = metadata_vocab_map[META_PAD_TOKEN_NAME]
        sos_midi_id = midi_tokenizer[MIDI_SOS_TOKEN_NAME]
        eos_midi_id = midi_tokenizer[MIDI_EOS_TOKEN_NAME]
    except KeyError as e:
        logging.error(f"Errore critico: Token speciale '{e}' non trovato nei vocabolari.")
        raise ValueError(f"Token speciale '{e}' non trovato.")

    meta_token_ids = [metadata_vocab_map.get(token, unk_meta_id) for token in metadata_prompt]
    src_seq = torch.tensor([[sos_meta_id] + meta_token_ids[:MAX_SEQ_LEN_META-2] + [eos_meta_id]], dtype=torch.long, device=device)
    src_padding_mask = (src_seq == meta_pad_id)

    with torch.no_grad():
        memory = model.encode(src_seq, src_padding_mask)
        memory_key_padding_mask = src_padding_mask
        
        initial_ids = [sos_midi_id] + (primer_token_ids if primer_token_ids else [])
        tgt_tokens = torch.tensor([initial_ids], dtype=torch.long, device=device)
        generated_ids_this_chunk = []

        for i in range(max_new_tokens):
            current_total_len = tgt_tokens.size(1)
            if current_total_len >= model_max_pe_len:
                logging.warning(f"Raggiunta capacità massima del modello ({model_max_pe_len}). Interruzione chunk.")
                break

            tgt_mask_step = torch.triu(torch.ones(current_total_len, current_total_len, device=device, dtype=torch.bool), diagonal=1)
            tgt_padding_mask_step = torch.zeros_like(tgt_tokens, dtype=torch.bool, device=device)

            decoder_output = model.decode(tgt=tgt_tokens, memory=memory, tgt_mask=tgt_mask_step,
                                          tgt_padding_mask=tgt_padding_mask_step,
                                          memory_key_padding_mask=memory_key_padding_mask)
            logits = model.generator(decoder_output[:, -1:, :]) 
            last_logits = logits[:, -1, :]


            if temperature > 0:
                last_logits = last_logits / temperature

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)), dim=-1)
                last_logits[last_logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(last_logits, dim=-1)
            next_token_id_tensor = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token_id_tensor.item()

            if next_token_id == eos_midi_id and len(generated_ids_this_chunk) < min_new_tokens:
                if probs.size(-1) > 1:
                    top_2_probs, top_2_indices = torch.topk(probs, 2, dim=-1)
                    if top_2_indices[0, 0].item() == eos_midi_id and top_2_indices.size(1) > 1:
                        next_token_id_tensor = top_2_indices[0, 1].unsqueeze(0).unsqueeze(0)
                        next_token_id = next_token_id_tensor.item()
            
            generated_ids_this_chunk.append(next_token_id)
            tgt_tokens = torch.cat((tgt_tokens, next_token_id_tensor), dim=1)

            if next_token_id == eos_midi_id and len(generated_ids_this_chunk) >= min_new_tokens:
                logging.info(f"Token EOS generato dopo {len(generated_ids_this_chunk)} nuovi token.")
                break
    return generated_ids_this_chunk

def generate_multi_chunk_midi(model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                              total_target_tokens, model_chunk_capacity, generation_config, device=DEVICE):
    all_generated_tokens = []
    current_primer_ids = []
    sos_midi_id = midi_tokenizer[MIDI_SOS_TOKEN_NAME]
    eos_midi_id = midi_tokenizer[MIDI_EOS_TOKEN_NAME]

    max_new_tokens_per_chunk = min(2048, model_chunk_capacity - PRIMER_TOKEN_COUNT - 5)
    if max_new_tokens_per_chunk <= MIN_TOKENS_PER_CHUNK:
        logging.error(f"max_new_tokens_per_chunk ({max_new_tokens_per_chunk}) troppo piccolo.")
        return []

    while len(all_generated_tokens) < total_target_tokens:
        remaining_tokens_to_generate = total_target_tokens - len(all_generated_tokens)
        current_pass_max_new = min(max_new_tokens_per_chunk, remaining_tokens_to_generate)
        
        if len(current_primer_ids) + current_pass_max_new + 1 > model_chunk_capacity:
            current_pass_max_new = model_chunk_capacity - len(current_primer_ids) - 1
            if current_pass_max_new < MIN_TOKENS_PER_CHUNK / 2 :
                 logging.warning("Spazio insufficiente nel chunk. Interruzione.")
                 break
        if current_pass_max_new < MIN_TOKENS_PER_CHUNK / 2 and len(all_generated_tokens) > 0 :
            logging.info("Chunk finale sarebbe troppo corto. Interruzione.")
            break

        logging.info(f"Generazione chunk. Totali: {len(all_generated_tokens)}/{total_target_tokens}. Primer: {len(current_primer_ids)}. Nuovi: {current_pass_max_new}")
        newly_generated_ids = generate_sequence(
            model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
            max_new_tokens=current_pass_max_new,
            min_new_tokens=min(MIN_TOKENS_PER_CHUNK, current_pass_max_new),
            temperature=generation_config.get("temperature", DEFAULT_TEMPERATURE),
            top_k=generation_config.get("top_k", 40), device=device,
            primer_token_ids=current_primer_ids, model_max_pe_len=model_chunk_capacity
        )
        if not newly_generated_ids:
            logging.warning("Chunk vuoto restituito. Interruzione.")
            break

        chunk_ended_with_eos = eos_midi_id in newly_generated_ids
        tokens_to_add = newly_generated_ids
        if chunk_ended_with_eos:
            eos_index = newly_generated_ids.index(eos_midi_id)
            tokens_to_add = newly_generated_ids[:eos_index + 1]
        
        all_generated_tokens.extend(tokens_to_add)
        logging.info(f"Chunk generato di {len(tokens_to_add)} tokens. Totale: {len(all_generated_tokens)}.")

        if chunk_ended_with_eos and len(all_generated_tokens) >= total_target_tokens * 0.8:
            logging.info("EOS generato e lunghezza vicina al target. Fine.")
            break
        elif chunk_ended_with_eos:
            logging.info("EOS generato, ma lunghezza non raggiunta. Continuo.")
            primer_candidate = tokens_to_add[:-1] if tokens_to_add[-1] == eos_midi_id else tokens_to_add
            current_primer_ids = primer_candidate[-PRIMER_TOKEN_COUNT:] if len(primer_candidate) > PRIMER_TOKEN_COUNT else []
            if not current_primer_ids : logging.warning("Chunk con EOS troppo corto per primer. Prossimo chunk senza primer.")
        else:
            current_primer_ids = tokens_to_add[-PRIMER_TOKEN_COUNT:] if len(tokens_to_add) >= PRIMER_TOKEN_COUNT else tokens_to_add
        
        if not newly_generated_ids: break

    if all_generated_tokens and all_generated_tokens[-1] != eos_midi_id:
        if len(all_generated_tokens) >= total_target_tokens:
            all_generated_tokens = all_generated_tokens[:total_target_tokens-1] + [eos_midi_id]
        else:
            all_generated_tokens.append(eos_midi_id)
    elif all_generated_tokens and len(all_generated_tokens) > total_target_tokens:
         all_generated_tokens = all_generated_tokens[:total_target_tokens-1] + [eos_midi_id]
    return all_generated_tokens

# --- Classe Applicazione GUI ---
class MusicGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Generatore Musicale MIDI")
        self.root.geometry("800x750") # Aumentata leggermente l'altezza

        self.model = None
        self.midi_tokenizer = None
        self.metadata_vocab_map = None
        self.id_to_metadata_token = None
        self.model_params = None
        self.effective_model_chunk_capacity = MAX_SEQ_LEN_MIDI

        self.cat_tokens_map = {} 
        self.gui_selections_vars = {} 

        # Variabili per Temperatura e Lunghezza Target
        self.temperature_var = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        self.target_length_var = tk.IntVar(value=DEFAULT_TOTAL_MIDI_TARGET_LENGTH)


        # Setup logging
        self.log_text = scrolledtext.ScrolledText(root, state='disabled', height=10, wrap=tk.WORD)
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        gui_handler = GuiLogger(self.log_text)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        gui_handler.setFormatter(formatter)
        logging.getLogger().addHandler(gui_handler)
        logging.getLogger().setLevel(logging.INFO)

        # Frame per i controlli
        controls_frame = ttk.Frame(root, padding="10")
        controls_frame.pack(fill=tk.X, expand=False)

        self.category_order = ["Key", "Instrument", "TimeSig", "Tempo", "AvgVel", "VelRange", "NumInst"]
        self.category_prefixes = {
            "Key": "Key=", "Instrument": "Instrument=", "TimeSig": "TimeSig=",
            "Tempo": "Tempo_", "AvgVel": "AvgVel_", "VelRange": "VelRange_",
            "NumInst": "NumInst_"
        }
        self.display_names = { 
            "Key": "Tonalità (Key)", "Instrument": "Strumento", "TimeSig": "Tempo (TimeSig)",
            "Tempo": "Velocità (Tempo BPM)", "AvgVel": "Velocity Media", "VelRange": "Range Velocity",
            "NumInst": "Numero Strumenti"
        }

        self.load_vocabs() 

        # Crea dropdown per ogni categoria
        current_row = 0
        for cat_key in self.category_order:
            display_name = self.display_names.get(cat_key, cat_key)
            ttk.Label(controls_frame, text=f"{display_name}:").grid(row=current_row, column=0, padx=5, pady=5, sticky=tk.W)
            
            self.gui_selections_vars[cat_key] = tk.StringVar()
            options = ["Random"] + sorted(self.cat_tokens_map.get(cat_key, []))
            
            combobox = ttk.Combobox(controls_frame, textvariable=self.gui_selections_vars[cat_key], values=options, width=40)
            if options:
                combobox.set("Random") 
            combobox.grid(row=current_row, column=1, padx=5, pady=5, sticky=tk.EW)
            current_row += 1

        # Campo per la Temperatura
        ttk.Label(controls_frame, text="Temperatura (es. 0.75):").grid(row=current_row, column=0, padx=5, pady=5, sticky=tk.W)
        temp_entry = ttk.Entry(controls_frame, textvariable=self.temperature_var, width=10)
        temp_entry.grid(row=current_row, column=1, padx=5, pady=5, sticky=tk.W) # sticky W per allineare a sinistra
        current_row += 1

        # Campo per la Lunghezza Target MIDI
        ttk.Label(controls_frame, text="Lunghezza Target MIDI (tokens):").grid(row=current_row, column=0, padx=5, pady=5, sticky=tk.W)
        length_entry = ttk.Entry(controls_frame, textvariable=self.target_length_var, width=10)
        length_entry.grid(row=current_row, column=1, padx=5, pady=5, sticky=tk.W) # sticky W
        current_row +=1

        controls_frame.columnconfigure(1, weight=1) 

        # Pulsante di generazione
        self.generate_button = ttk.Button(root, text="Genera 3 MIDI", command=self.start_generation_thread)
        self.generate_button.pack(pady=10)

        GENERATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"L'output MIDI verrà salvato in: {GENERATION_OUTPUT_DIR.resolve()}")
        logging.info("Pronto per la generazione. Seleziona i parametri e clicca 'Genera 3 MIDI'.")
        logging.info(f"Utilizzo del dispositivo: {DEVICE}")


    def load_vocabs(self):
        try:
            self.midi_tokenizer = load_midi_tokenizer_for_inference(PATH_VOCAB_MIDI)
            self.metadata_vocab_map, self.id_to_metadata_token = load_metadata_vocab_for_inference(PATH_VOCAB_METADATA)
            
            all_meta_tokens = list(self.metadata_vocab_map.keys())
            for cat_key, prefix in self.category_prefixes.items():
                self.cat_tokens_map[cat_key] = [t for t in all_meta_tokens if t.startswith(prefix)]
            logging.info("Vocabolari caricati con successo.")
        except Exception as e:
            logging.error(f"Errore fatale durante il caricamento dei vocabolari: {e}", exc_info=True)
            messagebox.showerror("Errore Vocabolari", f"Impossibile caricare i vocabolari necessari: {e}")
            self.root.quit()


    def load_model(self):
        if self.model is not None:
            return True
        try:
            logging.info(f"Caricamento checkpoint modello da: {PATH_MODELLO_CHECKPOINT}")
            if not PATH_MODELLO_CHECKPOINT.exists():
                logging.error(f"File checkpoint modello non trovato: {PATH_MODELLO_CHECKPOINT}")
                messagebox.showerror("Errore Modello", f"File checkpoint non trovato: {PATH_MODELLO_CHECKPOINT}")
                return False
            
            checkpoint = torch.load(PATH_MODELLO_CHECKPOINT, map_location=DEVICE)
            self.model_params = checkpoint.get('model_params')
            if not self.model_params:
                logging.error("ERRORE: 'model_params' non trovato nel checkpoint.")
                messagebox.showerror("Errore Modello", "'model_params' non trovato nel checkpoint.")
                return False

            self.effective_model_chunk_capacity = self.model_params.get('max_pe_len', MAX_SEQ_LEN_MIDI)
            logging.info(f"Capacità massima del modello per singolo chunk: {self.effective_model_chunk_capacity}")

            self.model = Seq2SeqTransformer(**self.model_params).to(DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logging.info("Modello caricato e impostato in modalità valutazione.")
            return True
        except Exception as e:
            logging.error(f"Errore durante il caricamento del modello: {e}", exc_info=True)
            messagebox.showerror("Errore Modello", f"Errore durante il caricamento del modello: {e}")
            self.model = None 
            return False

    def build_prompt_from_gui_selections(self):
        current_prompt_tokens = []
        gui_selections = {cat: var.get() for cat, var in self.gui_selections_vars.items()}

        for category_key in self.category_order:
            selected_value = gui_selections.get(category_key)
            chosen_token_for_category = None

            if selected_value and selected_value != "Random":
                if selected_value in self.metadata_vocab_map: 
                    chosen_token_for_category = selected_value
                else:
                    logging.warning(f"Token selezionato '{selected_value}' per '{category_key}' non valido/trovato. Scelta casuale.")
            
            if not chosen_token_for_category: 
                available_tokens_for_category = self.cat_tokens_map.get(category_key)
                if available_tokens_for_category:
                    chosen_token_for_category = random.choice(available_tokens_for_category)
                    logging.info(f"Per '{category_key}', scelto casualmente: '{chosen_token_for_category}'")
                else:
                    logging.warning(f"Nessun token disponibile per la categoria '{category_key}'. Categoria omessa.")
            
            if chosen_token_for_category:
                current_prompt_tokens.append(chosen_token_for_category)
        return current_prompt_tokens

    def generation_task(self):
        self.generate_button.config(state=tk.DISABLED)
        
        if not self.load_model(): 
            self.generate_button.config(state=tk.NORMAL)
            return

        # Recupera Temperatura e Lunghezza Target dalla GUI
        try:
            current_temperature = self.temperature_var.get()
            if not (0.01 <= current_temperature <= 2.0): # Range di esempio per la temperatura
                logging.warning(f"Temperatura '{current_temperature}' fuori range valido (0.01-2.0). Uso default: {DEFAULT_TEMPERATURE}")
                current_temperature = DEFAULT_TEMPERATURE
        except tk.TclError:
            logging.warning(f"Input Temperatura non valido. Uso default: {DEFAULT_TEMPERATURE}")
            current_temperature = DEFAULT_TEMPERATURE
        
        try:
            current_target_length = self.target_length_var.get()
            if not (MIN_TOKENS_PER_CHUNK * 2 <= current_target_length <= self.effective_model_chunk_capacity * 5): # Range di esempio
                logging.warning(f"Lunghezza Target '{current_target_length}' fuori range valido. Uso default: {DEFAULT_TOTAL_MIDI_TARGET_LENGTH}")
                current_target_length = DEFAULT_TOTAL_MIDI_TARGET_LENGTH
        except tk.TclError:
            logging.warning(f"Input Lunghezza Target non valido. Uso default: {DEFAULT_TOTAL_MIDI_TARGET_LENGTH}")
            current_target_length = DEFAULT_TOTAL_MIDI_TARGET_LENGTH


        generation_params = {
            "temperature": current_temperature,
            "top_k": 40 # Potrebbe essere reso configurabile
        }
        
        num_midi_to_generate = 3
        for i in range(num_midi_to_generate):
            logging.info(f"--- Inizio generazione MIDI {i+1}/{num_midi_to_generate} ---")
            
            current_metadata_prompt = self.build_prompt_from_gui_selections()
            if not current_metadata_prompt:
                logging.error("Prompt metadati vuoto. Impossibile generare.")
                continue
            
            logging.info(f"Prompt metadati per MIDI {i+1}: {current_metadata_prompt}")
            logging.info(f"Temperatura: {current_temperature}, Lunghezza Target: {current_target_length} tokens.")

            try:
                generated_token_ids = generate_multi_chunk_midi(
                    self.model, self.midi_tokenizer, self.metadata_vocab_map, current_metadata_prompt,
                    total_target_tokens=current_target_length, # Usa la lunghezza dalla GUI
                    model_chunk_capacity=self.effective_model_chunk_capacity,
                    generation_config=generation_params, # Passa la temperatura dalla GUI
                    device=DEVICE
                )

                if generated_token_ids:
                    logging.info(f"Generati {len(generated_token_ids)} token MIDI totali per MIDI {i+1}.")
                    generated_midi_object = self.midi_tokenizer.decode(generated_token_ids)
                    
                    if generated_midi_object:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        prompt_name_part_list = []
                        for token_str in current_metadata_prompt:
                            name = token_str.split('=')[-1].replace('_', '')[:10] 
                            prompt_name_part_list.append(name)
                        prompt_name_part = "_".join(prompt_name_part_list)
                        prompt_name_part = ''.join(c for c in prompt_name_part if c.isalnum() or c == '_')[:70]

                        output_filename_path = GENERATION_OUTPUT_DIR / f"gui_T{current_temperature:.2f}_L{current_target_length}_{prompt_name_part}_{timestamp}_{i+1}.mid"
                        
                        generated_midi_object.dump_midi(str(output_filename_path))
                        logging.info(f"File MIDI {i+1} salvato in: {output_filename_path.resolve()}")
                    else:
                        logging.warning(f"midi_tokenizer.decode ha restituito None per MIDI {i+1} (prompt: {current_metadata_prompt}).")
                else:
                    logging.warning(f"Generazione per MIDI {i+1} (prompt: {current_metadata_prompt}) fallita o prodotto sequenza vuota.")
            
            except Exception as e:
                logging.error(f"Errore durante la generazione o il salvataggio per MIDI {i+1} (prompt: {current_metadata_prompt}): {e}", exc_info=True)
            
            if i < num_midi_to_generate - 1:
                time.sleep(0.5) 
        
        logging.info("--- Tutte le generazioni MIDI sono terminate ---")
        self.generate_button.config(state=tk.NORMAL)


    def start_generation_thread(self):
        generation_thread = threading.Thread(target=self.generation_task, daemon=True)
        generation_thread.start()


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicGeneratorApp(root)
    root.mainloop()
