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

# --- Configurazione / Costanti Essenziali per la Generazione ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODIFICA QUESTI PERCORSI ===
# Percorso del checkpoint del modello addestrato (.pt)
PATH_MODELLO_CHECKPOINT = Path(r"C:\Users\Michael\Desktop\ModelliMusicGenerator\ModelloPianoTrasposto\transformer_periodic_epoch81_valloss1.5116_20250529-215034.pt")
# Percorso del file di vocabolario MIDI (.json) usato per addestrare il modello sopra
PATH_VOCAB_MIDI = Path(r"C:\Users\Michael\Desktop\SheetsMusicGenerator\ModelloPianoTrasposto\midi_vocab.json")
# Percorso del file di vocabolario dei metadati (.json) usato per addestrare il modello sopra
PATH_VOCAB_METADATA = Path(r"C:\Users\Michael\Desktop\SheetsMusicGenerator\ModelloPianoTrasposto\metadata_vocab.json")
# Directory dove salvare i file MIDI generati
GENERATION_OUTPUT_DIR = Path("./generated_midi_inference")
# ================================

# === IMPOSTAZIONI PER CHIAVE/STRUMENTO FORZATI ===
# Se il tuo modello è stato addestrato con una chiave specifica (es. per "piano_only_transposed")
# imposta qui i token corrispondenti. Altrimenti, per una selezione completamente casuale, imposta FORCED_KEY_INFO = None.
#
# !!! IMPORTANTE !!!
# VERIFICA CHE QUESTI TOKEN ESISTANO ESATTAMENTE COSÌ NEL FILE `metadata_vocab.json`
# CHE STAI USANDO CON IL MODELLO SPECIFICATO IN `PATH_MODELLO_CHECKPOINT`.
# SE `Key=Target_Cmaj_Amin` NON ESISTE NEL TUO VOCAB, IL MODELLO POTREBBE NON ESSERE STATO
# ADDESTRATO COME PREVISTO O STAI USANDO IL VOCABOLARIO SBAGLIATO.
FORCED_KEY_INFO = {
    "key_token": "Key=C_major",        # Esempio: Sostituisci se il tuo token è diverso.
    "instrument_token": "Instrument=Acoustic_Grand_Piano", # Esempio: Sostituisci con lo strumento desiderato.
    "avg_token": "AvgVel_Medium",
    "tempo_token": "Tempo_Moderate",
    "time_signature_token": "TimeSig=4/4",
    "range_token": "VelRange_Wide",
    # "num_inst_token": "NumInst_Solo" # Aggiungi se vuoi forzare anche NumInst
}
# Se non vuoi forzare nulla e vuoi che tutti i metadati (inclusa la chiave) siano casuali:
# FORCED_KEY_INFO = None
# -------------------------------------------------

# Strategia di Tokenizzazione (DEVE CORRISPONDERE A QUELLA USATA PER L'ADDDESTRAMENTO DEL MODELLO CARICATO)
# Ad esempio, se il modello è stato addestrato con REMI:
MIDI_TOKENIZER_STRATEGY = miditok.REMI
# Se era CPWord:
# MIDI_TOKENIZER_STRATEGY = miditok.CPWord # Attenzione: CPWord ha mostrato problemi di multi-vocabolario

# Token Speciali MIDI (devono corrispondere a quelli usati per l'addestramento)
MIDI_PAD_TOKEN_NAME = "PAD_None"
MIDI_SOS_TOKEN_NAME = "SOS_None"
MIDI_EOS_TOKEN_NAME = "EOS_None"
MIDI_UNK_TOKEN_NAME = "UNK_None"

# Token Speciali per Metadati (devono corrispondere)
META_PAD_TOKEN_NAME = "<pad_meta>"
META_UNK_TOKEN_NAME = "<unk_meta>"
META_SOS_TOKEN_NAME = "<sos_meta>"
META_EOS_TOKEN_NAME = "<eos_meta>"

# Lunghezze massime sequenza (idealmente prese dal checkpoint o corrispondenti al training)
# Queste sono usate in generate_sequence e PositionalEncoding.
# Se il checkpoint non le salva esplicitamente per PositionalEncoding, assicurati che siano abbastanza grandi.
MAX_SEQ_LEN_MIDI = 1024
MAX_SEQ_LEN_META = 128

PRIMER_TOKEN_COUNT = 50  # Quanti token usare dalla fine del chunk precedente come primer
                          # Deve essere minore di effective_model_max_len
MIN_TOKENS_PER_CHUNK = 100 # Lunghezza minima per un chunk prima di accettare EOS
                           # o per continuare la generazione
                           
TOTAL_MIDI_TARGET_LENGTH = 2048 # lunghezza totale desiderata in token
                                # Aumenta questo valore per brani più lunghi.
                                # Se è più grande di effective_model_chunk_capacity, verranno generati più chunk.

TEMPERATURE = 0.75 # Temperatura per la generazione, regola la casualità
# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Crea directory di output se non esiste
GENERATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Definizioni Helper, Classi Modello e Funzioni (copiate da training.py) ---

def tokenize_metadata(metadata_dict):
    tokens = []
    if 'style' in metadata_dict and metadata_dict['style']:
        tokens.append(f"Style={metadata_dict['style'].replace(' ', '_')}")
    if 'key' in metadata_dict and metadata_dict['key']:
        tokens.append(f"Key={metadata_dict['key'].replace(' ', '_')}")
    if 'time_signature' in metadata_dict and metadata_dict['time_signature']:
        tokens.append(f"TimeSig={metadata_dict['time_signature']}")
    if 'title' in metadata_dict and metadata_dict['title']:
        clean_title = metadata_dict['title'].replace(' ', '_').lower()
        clean_title = ''.join(c for c in clean_title if c.isalnum() or c == '_')
        tokens.append(f"Title={clean_title[:30]}")
    return tokens

def load_midi_tokenizer_for_inference(vocab_path):
    logging.info(f"Caricamento tokenizer MIDI da {vocab_path}")
    if not Path(vocab_path).exists():
        logging.error(f"File vocabolario MIDI non trovato: {vocab_path}")
        sys.exit(1)
    try:
        # La configurazione del tokenizer (use_programs, ecc.) è definita nel file params.
        # Non è necessario passare una TokenizerConfig qui se il file params è completo.
        tokenizer = MIDI_TOKENIZER_STRATEGY(params=str(vocab_path))
        logging.info(f"Tokenizer MIDI caricato con successo. Strategia: {MIDI_TOKENIZER_STRATEGY.__name__}")

        # Verifica ID token speciali essenziali
        for token_name in [MIDI_PAD_TOKEN_NAME, MIDI_SOS_TOKEN_NAME, MIDI_EOS_TOKEN_NAME, MIDI_UNK_TOKEN_NAME]:
            try:
                _ = tokenizer[token_name] # Tenta di accedere
            except KeyError:
                logging.error(f"ERRORE CRITICO: Token speciale MIDI '{token_name}' non trovato nel vocabolario caricato da {vocab_path}.")
                sys.exit(1)
        return tokenizer
    except Exception as e:
        logging.error(f"Errore nel caricare il tokenizer MIDI da {vocab_path}: {e}", exc_info=True)
        sys.exit(1)

def load_metadata_vocab_for_inference(vocab_path):
    logging.info(f"Caricamento vocabolario Metadati da {vocab_path}")
    if not Path(vocab_path).exists():
        logging.error(f"File vocabolario Metadati non trovato: {vocab_path}")
        sys.exit(1)
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        token_to_id = vocab_data['token_to_id']
        
        # Verifica token speciali essenziali per metadati
        for token_name in [META_PAD_TOKEN_NAME, META_UNK_TOKEN_NAME, META_SOS_TOKEN_NAME, META_EOS_TOKEN_NAME]:
            if token_name not in token_to_id:
                logging.error(f"ERRORE CRITICO: Token speciale Metadati '{token_name}' non trovato nel vocabolario caricato da {vocab_path}.")
                sys.exit(1)
        return token_to_id
    except Exception as e:
        logging.error(f"Errore nel caricare il vocabolario Metadati da {vocab_path}: {e}", exc_info=True)
        sys.exit(1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000): # max_len dovrebbe essere abbastanza grande
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
                      max_new_tokens, 
                      min_new_tokens, 
                      temperature=TEMPERATURE, top_k=None, device=DEVICE,
                      primer_token_ids=None, 
                      model_max_pe_len=5000): 
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
        sys.exit(1)

    meta_token_ids = [metadata_vocab_map.get(token, unk_meta_id) for token in metadata_prompt]
    src_seq = torch.tensor([[sos_meta_id] + meta_token_ids[:MAX_SEQ_LEN_META-2] + [eos_meta_id]], dtype=torch.long, device=device)
    src_padding_mask = (src_seq == meta_pad_id)

    with torch.no_grad():
        memory = model.encode(src_seq, src_padding_mask)
        memory_key_padding_mask = src_padding_mask
        
        if primer_token_ids and len(primer_token_ids) > 0:
            initial_ids = [sos_midi_id] + primer_token_ids
        else:
            initial_ids = [sos_midi_id]
        
        tgt_tokens = torch.tensor([initial_ids], dtype=torch.long, device=device)
        
        generated_ids_this_chunk = []
        generated_eos_in_chunk = False

        for i in range(max_new_tokens):
            current_total_len = tgt_tokens.size(1)
            if current_total_len >= model_max_pe_len:
                logging.warning(f"Raggiunta capacità massima del modello ({model_max_pe_len}) per il chunk corrente. Interruzione anticipata del chunk.")
                break

            tgt_mask_step = torch.triu(torch.ones(current_total_len, current_total_len, device=device, dtype=torch.bool), diagonal=1)
            tgt_padding_mask_step = torch.zeros_like(tgt_tokens, dtype=torch.bool, device=device) 

            decoder_output = model.decode(tgt=tgt_tokens, memory=memory, tgt_mask=tgt_mask_step,
                                          tgt_padding_mask=tgt_padding_mask_step,
                                          memory_key_padding_mask=memory_key_padding_mask)
            logits = model.generator(decoder_output)
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
                    if top_2_indices[0, 0].item() == eos_midi_id:
                        if top_2_indices.size(1) > 1:
                            next_token_id_tensor = top_2_indices[0, 1].unsqueeze(0).unsqueeze(0)
                            next_token_id = next_token_id_tensor.item()
                        else: 
                            logging.warning("Forzato a generare EOS nonostante min_new_tokens non raggiunta (solo EOS possibile).")
                            generated_eos_in_chunk = True
            
            generated_ids_this_chunk.append(next_token_id)
            tgt_tokens = torch.cat((tgt_tokens, next_token_id_tensor), dim=1)

            if next_token_id == eos_midi_id and len(generated_ids_this_chunk) >= min_new_tokens:
                logging.info(f"Token EOS generato nel chunk dopo {len(generated_ids_this_chunk)} nuovi token (min_new_tokens raggiunta).")
                generated_eos_in_chunk = True
                break 
            
            if next_token_id != eos_midi_id:
                generated_eos_in_chunk = False 
    
    return generated_ids_this_chunk

def generate_multi_chunk_midi(
    model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
    total_target_tokens, 
    model_chunk_capacity, 
    generation_config, 
    device=DEVICE
):
    all_generated_tokens = []
    current_primer_ids = []
    sos_midi_id = midi_tokenizer[MIDI_SOS_TOKEN_NAME]
    eos_midi_id = midi_tokenizer[MIDI_EOS_TOKEN_NAME]

    max_new_tokens_per_chunk = min(2048, model_chunk_capacity - PRIMER_TOKEN_COUNT - 5) 
    if max_new_tokens_per_chunk <= MIN_TOKENS_PER_CHUNK:
        logging.error(f"max_new_tokens_per_chunk ({max_new_tokens_per_chunk}) troppo piccolo. Controlla model_chunk_capacity e PRIMER_TOKEN_COUNT.")
        return []

    while len(all_generated_tokens) < total_target_tokens:
        remaining_tokens_to_generate = total_target_tokens - len(all_generated_tokens)
        current_pass_max_new = min(max_new_tokens_per_chunk, remaining_tokens_to_generate)
        
        if len(current_primer_ids) + current_pass_max_new + 1 > model_chunk_capacity: 
            current_pass_max_new = model_chunk_capacity - len(current_primer_ids) - 1
            if current_pass_max_new < MIN_TOKENS_PER_CHUNK / 2 : 
                 logging.warning("Spazio insufficiente nel chunk per generare una quantità significativa di nuovi token. Interruzione.")
                 break

        if current_pass_max_new < MIN_TOKENS_PER_CHUNK / 2 and len(all_generated_tokens) > 0 : 
            logging.info("Chunk finale sarebbe troppo corto, interruzione generazione.")
            break

        logging.info(f"Generazione nuovo chunk. Tokens totali finora: {len(all_generated_tokens)}/{total_target_tokens}. "
                     f"Primer attuale: {len(current_primer_ids)} tokens. Target nuovi token per questo chunk: {current_pass_max_new}")

        newly_generated_ids = generate_sequence(
            model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
            max_new_tokens=current_pass_max_new,
            min_new_tokens=min(MIN_TOKENS_PER_CHUNK, current_pass_max_new), 
            temperature=generation_config.get("temperature", 0.7),
            top_k=generation_config.get("top_k", 40),
            device=device,
            primer_token_ids=current_primer_ids,
            model_max_pe_len=model_chunk_capacity
        )

        if not newly_generated_ids:
            logging.warning("generate_sequence ha restituito un chunk vuoto. Interruzione.")
            break

        chunk_ended_with_eos = False
        if eos_midi_id in newly_generated_ids:
            chunk_ended_with_eos = True
            eos_index = newly_generated_ids.index(eos_midi_id)
            tokens_to_add = newly_generated_ids[:eos_index + 1] 
        else:
            tokens_to_add = newly_generated_ids
        
        all_generated_tokens.extend(tokens_to_add)
        logging.info(f"Chunk generato di {len(tokens_to_add)} tokens. Totale attuale: {len(all_generated_tokens)}.")

        if chunk_ended_with_eos and len(all_generated_tokens) >= total_target_tokens * 0.8: 
            logging.info("EOS generato e lunghezza vicina al target. Generazione terminata.")
            break
        elif chunk_ended_with_eos: 
            logging.info("EOS generato nel chunk, ma la lunghezza totale non è ancora raggiunta. Si tenta di continuare.")
            if len(tokens_to_add) > PRIMER_TOKEN_COUNT : 
                 primer_candidate = tokens_to_add[:-1] if tokens_to_add[-1] == eos_midi_id else tokens_to_add
                 current_primer_ids = primer_candidate[-PRIMER_TOKEN_COUNT:]
            else: 
                 current_primer_ids = []
            if not current_primer_ids and len(all_generated_tokens) < total_target_tokens : 
                 logging.warning("Chunk terminato con EOS troppo corto per un primer, ma non abbiamo finito. Prossimo chunk senza primer.")
        else: 
            if len(tokens_to_add) >= PRIMER_TOKEN_COUNT:
                current_primer_ids = tokens_to_add[-PRIMER_TOKEN_COUNT:]
            else: 
                current_primer_ids = tokens_to_add 
        
        if not newly_generated_ids: 
            break

    if all_generated_tokens:
        if all_generated_tokens[-1] != eos_midi_id:
            if len(all_generated_tokens) >= total_target_tokens:
                all_generated_tokens = all_generated_tokens[:total_target_tokens-1] + [eos_midi_id]
            else:
                all_generated_tokens.append(eos_midi_id)
        else: 
            if len(all_generated_tokens) > total_target_tokens:
                 all_generated_tokens = all_generated_tokens[:total_target_tokens-1] + [eos_midi_id]
    return all_generated_tokens

def create_random_prompts(token_to_id_map, num_prompts=3, forced_key_info=None):
    prompts = []
    all_tokens = list(token_to_id_map.keys())

    # Raccoglie i token disponibili per ogni categoria semantica basandosi sui prefissi
    # Questo assicura che se una categoria non ha token nel vocabolario, non causerà errori.
    cat_tokens = {
        "Key": [t for t in all_tokens if t.startswith("Key=")],
        "TimeSig": [t for t in all_tokens if t.startswith("TimeSig=")],
        "Instrument": [t for t in all_tokens if t.startswith("Instrument=")],
        "Tempo": [t for t in all_tokens if t.startswith("Tempo_")],
        "AvgVel": [t for t in all_tokens if t.startswith("AvgVel_")],
        "VelRange": [t for t in all_tokens if t.startswith("VelRange_")],
        "NumInst": [t for t in all_tokens if t.startswith("NumInst_")] 
    }

    # Mappa per collegare le chiavi in forced_key_info ai nomi delle categorie definite sopra.
    # Questo aiuta a generalizzare la logica di forzatura.
    forced_map = {
        "key_token": "Key",
        "instrument_token": "Instrument",
        "avg_token": "AvgVel",
        "tempo_token": "Tempo",
        "time_signature_token": "TimeSig",
        "range_token": "VelRange",
        "num_inst_token": "NumInst" # Aggiunta per coerenza, se l'utente vuole forzare NumInst
    }

    # Ordine desiderato per i token di metadati nel prompt.
    # Questo può influenzare come il modello interpreta il prompt.
    order_of_categories = ["Key", "TimeSig", "NumInst", "Instrument", "Tempo", "AvgVel", "VelRange"]

    for i in range(num_prompts):
        current_prompt_tokens = [] # Lista per i token del prompt corrente
        
        # Dizionario per tenere traccia dei token selezionati (forzati o casuali) per questo prompt
        selected_tokens_for_this_prompt = {}

        # 1. Applica i token forzati, se specificati e validi
        if forced_key_info:
            for config_key, category_name in forced_map.items():
                forced_token_value = forced_key_info.get(config_key)
                if forced_token_value: # Se un valore è specificato in FORCED_KEY_INFO
                    if forced_token_value in token_to_id_map:
                        selected_tokens_for_this_prompt[category_name] = forced_token_value
                        logging.info(f"Prompt {i+1}: Categoria '{category_name}' forzata a '{forced_token_value}'.")
                    else:
                        logging.warning(f"Prompt {i+1}: Token forzato '{forced_token_value}' per categoria '{category_name}' NON TROVATO nel vocabolario. Sarà scelto casualmente se disponibile.")
        
        # 2. Seleziona i token per ogni categoria (forzati o scelti casualmente)
        #    seguendo l'ordine specificato.
        for category_name in order_of_categories:
            chosen_token_for_category = selected_tokens_for_this_prompt.get(category_name)

            if not chosen_token_for_category: # Se non è stato forzato o la forzatura è fallita
                available_tokens_for_category = cat_tokens.get(category_name)
                if available_tokens_for_category: # Se ci sono token disponibili per questa categoria
                    chosen_token_for_category = random.choice(available_tokens_for_category)
                    
                    # Log specifico se la chiave è scelta casualmente per modelli trasposti
                    if category_name == "Key" and (not forced_key_info or not forced_key_info.get("key_token")):
                         logging.info(f"Prompt {i+1} - AVVISO CHIAVE: Selezionata chiave casuale '{chosen_token_for_category}'. "
                                      f"Se usi un modello 'piano_only_transposed' che si aspetta una chiave specifica, "
                                      f"questa chiave casuale potrebbe non dare i risultati sperati.")
                else:
                    logging.warning(f"Prompt {i+1}: Nessun token disponibile per la categoria '{category_name}'. Categoria omessa dal prompt.")
            
            if chosen_token_for_category: # Se un token è stato forzato con successo o scelto casualmente
                current_prompt_tokens.append(chosen_token_for_category)
        
        if current_prompt_tokens:
            prompts.append(current_prompt_tokens)
            logging.info(f"Prompt {i+1} costruito: {current_prompt_tokens}")
        else:
            logging.warning(f"Prompt {i+1}: Non è stato possibile costruire un prompt valido.")
            
    return prompts


# --- Esecuzione Principale per la Sola Generazione ---
if __name__ == "__main__":
    logging.info("--- Script di Generazione Avviato ---")

    # 1. Carica Tokenizer e Vocabolari
    midi_tokenizer = load_midi_tokenizer_for_inference(PATH_VOCAB_MIDI)
    metadata_vocab_map = load_metadata_vocab_for_inference(PATH_VOCAB_METADATA)

    # 2. Carica il Checkpoint del Modello
    if not PATH_MODELLO_CHECKPOINT.exists():
        logging.error(f"File checkpoint modello non trovato: {PATH_MODELLO_CHECKPOINT}")
        sys.exit(1)
    checkpoint = torch.load(PATH_MODELLO_CHECKPOINT, map_location=DEVICE)
    model_params = checkpoint.get('model_params')
    if not model_params:
        logging.error("ERRORE: 'model_params' non trovato nel checkpoint.")
        sys.exit(1)
    effective_model_chunk_capacity = model_params.get('max_pe_len', MAX_SEQ_LEN_MIDI)
    if effective_model_chunk_capacity == MAX_SEQ_LEN_MIDI and not model_params.get('max_pe_len'):
         logging.warning(f"max_pe_len non trovato nel checkpoint! Uso MAX_SEQ_LEN_MIDI ({MAX_SEQ_LEN_MIDI}), potrebbe causare errori.")
    logging.info(f"Capacità massima del modello per singolo chunk: {effective_model_chunk_capacity}")

    # 3. Istanzia e Carica il Modello
    model = Seq2SeqTransformer(**model_params).to(DEVICE)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        logging.error(f"Errore caricamento model_state_dict: {e}")
        # Considera di aggiungere qui un sys.exit(1) se l'errore è critico
    model.eval()
    logging.info("Modello caricato e impostato in modalità valutazione.")

    # 4. Prepara Prompt e Genera
    numero_di_prompt_da_generare = 3
    logging.info(f"Creazione di {numero_di_prompt_da_generare} prompt di metadati...")
    
    example_metadata_prompt_list = create_random_prompts(
        metadata_vocab_map,
        num_prompts=numero_di_prompt_da_generare,
        forced_key_info=FORCED_KEY_INFO 
    )

    if not example_metadata_prompt_list:
        logging.error("Nessun prompt è stato generato. Controlla i log e il vocabolario dei metadati. Uscita.")
        sys.exit(1)
    
    # Rimuovi il log ridondante qui, dato che create_random_prompts ora logga ogni prompt costruito
    # logging.info(f"Prompt generati: {example_metadata_prompt_list}") 

    generation_params = {
        "temperature": TEMPERATURE,
        "top_k": 40 # Puoi anche aggiungere top_k a FORCED_KEY_INFO o passarlo diversamente
    }

    for idx, example_metadata_prompt in enumerate(example_metadata_prompt_list):
        logging.info(f"--- Generazione {idx+1}/{len(example_metadata_prompt_list)} ---")
        logging.info(f"Prompt metadati per questa generazione: {example_metadata_prompt}")
        logging.info(f"Target lunghezza totale MIDI: {TOTAL_MIDI_TARGET_LENGTH} tokens.")

        try:
            generated_token_ids = generate_multi_chunk_midi(
                model, midi_tokenizer, metadata_vocab_map, example_metadata_prompt,
                total_target_tokens=TOTAL_MIDI_TARGET_LENGTH,
                model_chunk_capacity=effective_model_chunk_capacity,
                generation_config=generation_params,
                device=DEVICE
            )

            if generated_token_ids:
                logging.info(f"Generati {len(generated_token_ids)} token MIDI totali.")
                generated_midi_object = midi_tokenizer.decode(generated_token_ids)
                
                if generated_midi_object:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    prompt_name_part = "_".join(example_metadata_prompt).replace("=", "").replace("/", "")
                    prompt_name_part = ''.join(c for c in prompt_name_part if c.isalnum() or c == '_')[:70]
                    
                    # Modificato il nome del file per riflettere meglio se la chiave è forzata o meno
                    filename_prefix = "generated"
                    if FORCED_KEY_INFO and FORCED_KEY_INFO.get("key_token"):
                        filename_prefix = f"generated_forcedkey_{FORCED_KEY_INFO['key_token'].replace('Key=','')}"
                    elif FORCED_KEY_INFO: # Forzato ma non la chiave specificamente
                        filename_prefix = "generated_forced_other_meta"
                    else: # Completamente casuale
                        filename_prefix = "generated_random_meta"

                    output_filename_path = GENERATION_OUTPUT_DIR / f"{filename_prefix}_{prompt_name_part}_{timestamp}_{idx}.mid"
                    GENERATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    
                    generated_midi_object.dump_midi(str(output_filename_path))
                    logging.info(f"File MIDI salvato in: {output_filename_path}")
                else:
                    logging.warning(f"midi_tokenizer.decode ha restituito None per prompt '{example_metadata_prompt}'.")
            else:
                logging.warning(f"Generazione per prompt '{example_metadata_prompt}' fallita o prodotto sequenza vuota.")
        
        except Exception as e:
            logging.error(f"Errore durante la generazione o il salvataggio per prompt '{example_metadata_prompt}': {e}", exc_info=True)
        
        if idx < len(example_metadata_prompt_list) - 1:
            time.sleep(1) # Breve pausa tra una generazione e l'altra
            
    logging.info("--- Script di Generazione Terminato ---")