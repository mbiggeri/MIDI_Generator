# generate_music.py (versione libreria)

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
from symusic import Score
import warnings

# Soppressione di avvisi comuni di miditok
warnings.filterwarnings("ignore", category=UserWarning, module='miditok.midi_tokenizer_base')

# --- DEFINIZIONI DEL MODELLO (invariate) ---
# Le classi PositionalEncoding e Seq2SeqTransformer rimangono identiche a prima.
# ... (le classi del modello sono qui, o importate da un file separato) ...
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

# --- FUNZIONI DI GENERAZIONE (modificate per essere più robuste) ---

# MODIFICATO: La firma della funzione ora accetta `rest_ids`
def generate_sequence(model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                      max_new_tokens, min_new_tokens, temperature, top_k, device,
                      primer_token_ids, model_max_pe_len, max_rest_penalty, rest_ids):
    model.eval()
    
    # Costanti token speciali (come prima)
    META_SOS_TOKEN_NAME = "<sos_meta>"
    META_EOS_TOKEN_NAME = "<eos_meta>"
    META_UNK_TOKEN_NAME = "<unk_meta>"
    META_PAD_TOKEN_NAME = "<pad_meta>"
    MIDI_SOS_TOKEN_NAME = "SOS_None"
    MIDI_EOS_TOKEN_NAME = "EOS_None"

    try:
        sos_meta_id = metadata_vocab_map[META_SOS_TOKEN_NAME]
        eos_meta_id = metadata_vocab_map[META_EOS_TOKEN_NAME]
        unk_meta_id = metadata_vocab_map[META_UNK_TOKEN_NAME]
        meta_pad_id = metadata_vocab_map[META_PAD_TOKEN_NAME]
        sos_midi_id = midi_tokenizer[MIDI_SOS_TOKEN_NAME]
        eos_midi_id = midi_tokenizer[MIDI_EOS_TOKEN_NAME]
    except KeyError as e:
        logging.error(f"Token speciale '{e}' non trovato nei vocabolari.")
        raise ValueError(f"Token speciale '{e}' mancante.")

    # --- INIZIO OTTIMIZZAZIONE ---

    with torch.no_grad():
        # 1. Codifica dei metadati (SRC): eseguita UNA SOLA VOLTA
        meta_token_ids = [metadata_vocab_map.get(token, unk_meta_id) for token in metadata_prompt]
        src_seq = torch.tensor([[sos_meta_id] + meta_token_ids + [eos_meta_id]], dtype=torch.long, device=device)
        src_padding_mask = (src_seq == meta_pad_id)
        
        memory = model.encode(src_seq, src_padding_mask)

        # 2. Inizializzazione della sequenza di output (TGT)
        initial_ids = [sos_midi_id] + (primer_token_ids if primer_token_ids else [])
        tgt_tokens = torch.tensor([initial_ids], dtype=torch.long, device=device)
        
        # 3. Loop di generazione autoregressivo
        for i in range(max_new_tokens):
            current_total_len = tgt_tokens.size(1)
            if current_total_len >= model_max_pe_len:
                logging.warning(f"Raggiunta capacità massima del modello ({model_max_pe_len}). Interruzione.")
                break

            tgt_mask = torch.triu(torch.ones(current_total_len, current_total_len, device=device, dtype=torch.bool), diagonal=1)
            
            decoder_output = model.decode(tgt=tgt_tokens, memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
            
            logits = model.generator(decoder_output[:, -1, :])
            
            # --- LOGICA DI PENALIZZAZIONE DINAMICA ---
            if max_rest_penalty > 0 and rest_ids is not None and rest_ids.numel() > 0:
                progress = i / max_new_tokens
                current_penalty = max_rest_penalty * progress
                logits[:, rest_ids] -= current_penalty
            
            # 4. Sampling (Top-K, Temperatura) - invariato
            if temperature > 0:
                scaled_logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(scaled_logits, dim=-1)
                next_token_id_tensor = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id_tensor = torch.argmax(logits, dim=-1, keepdim=True)
            next_token_id = next_token_id_tensor.item()

            # 5. Gestione EOS e aggiunta del nuovo token - invariato
            if next_token_id == eos_midi_id and tgt_tokens.size(1) - len(initial_ids) < min_new_tokens:
                _, top_2_indices = torch.topk(probs, 2, dim=-1)
                if top_2_indices[0, 0].item() == eos_midi_id and top_2_indices.size(1) > 1:
                    next_token_id_tensor = top_2_indices[0, 1].unsqueeze(0).unsqueeze(0)
                else:
                    break 
            
            if next_token_id == eos_midi_id and tgt_tokens.size(1) - len(initial_ids) >= min_new_tokens:
                break
            
            tgt_tokens = torch.cat((tgt_tokens, next_token_id_tensor), dim=1)

    return tgt_tokens[0, len(initial_ids):].tolist()


def generate_multi_chunk_midi(model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                              total_target_tokens, model_chunk_capacity, generation_config, device,
                              initial_primer_ids=None, rest_ids=None): # MODIFICATO
    all_generated_tokens = []
    current_primer_ids = initial_primer_ids.copy() if initial_primer_ids else [] 

    PRIMER_TOKEN_COUNT = 50
    MIN_TOKENS_PER_CHUNK = 100
    max_new_tokens_per_chunk = min(2048, model_chunk_capacity - PRIMER_TOKEN_COUNT - 5)
    eos_midi_id = midi_tokenizer["EOS_None"]

    while len(all_generated_tokens) < total_target_tokens:
        remaining_tokens_to_generate = total_target_tokens - len(all_generated_tokens)
        current_pass_max_new = min(max_new_tokens_per_chunk, remaining_tokens_to_generate)
        if len(current_primer_ids) + current_pass_max_new + 1 > model_chunk_capacity: break
        logging.info(f"Generazione chunk. Totale: {len(all_generated_tokens)}/{total_target_tokens}.")
        
        # MODIFICATO: Passaggio dei nuovi parametri (`max_rest_penalty` e `rest_ids`)
        newly_generated_ids = generate_sequence(
            model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
            max_new_tokens=current_pass_max_new,
            min_new_tokens=min(MIN_TOKENS_PER_CHUNK, current_pass_max_new),
            temperature=generation_config.get("temperature", 0.75),
            top_k=generation_config.get("top_k", 40),
            max_rest_penalty=generation_config.get("max_rest_penalty", 0.0), # Ottiene il valore da config
            device=device,
            primer_token_ids=current_primer_ids,
            model_max_pe_len=model_chunk_capacity,
            rest_ids=rest_ids) # Passa i rest_ids
        if not newly_generated_ids: break
        
        chunk_ended_with_eos = eos_midi_id in newly_generated_ids
        tokens_to_add = newly_generated_ids[:newly_generated_ids.index(eos_midi_id) + 1] if chunk_ended_with_eos else newly_generated_ids
        all_generated_tokens.extend(tokens_to_add)

        if chunk_ended_with_eos:
            if len(all_generated_tokens) >= total_target_tokens * 0.8: break
            primer_candidate = tokens_to_add[:-1]
            current_primer_ids = primer_candidate[-PRIMER_TOKEN_COUNT:] if len(primer_candidate) > PRIMER_TOKEN_COUNT else []
        else:
            current_primer_ids = tokens_to_add[-PRIMER_TOKEN_COUNT:]
    if all_generated_tokens and all_generated_tokens[-1] != eos_midi_id:
        all_generated_tokens.append(eos_midi_id)
    return all_generated_tokens

# --- FUNZIONE DI ANALISI MODELLO ---
def get_model_info(model_path: str) -> dict:
    """
    Carica un checkpoint del modello e ne estrae le informazioni sull'architettura.

    Args:
        model_path: Il percorso del file del modello (.pt).

    Returns:
        Un dizionario contenente i parametri del modello o un dizionario di errore.
    """
    if not model_path or not Path(model_path).exists():
        return {"error": "Selezionare un percorso valido per il modello."}

    try:
        # Carica il checkpoint su CPU per evitare problemi di memoria GPU solo per l'analisi
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        model_params = checkpoint.get('model_params')
        if not model_params:
            return {"error": "'model_params' non trovato nel checkpoint del modello."}

        # Istanzia temporaneamente il modello per calcolare i parametri
        model = Seq2SeqTransformer(**model_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Prepara un dizionario con le informazioni formattate
        info = {
            "Parametri Addestrabili": f"{trainable_params:,}".replace(",", "."),
            "Dimensione Embedding (d_model)": model_params.get('emb_size', 'N/D'),
            "Attention Heads": model_params.get('nhead', 'N/D'),
            "Livelli Encoder": model_params.get('num_encoder_layers', 'N/D'),
            "Livelli Decoder": model_params.get('num_decoder_layers', 'N/D'),
            "Dimensione Feedforward": model_params.get('dim_feedforward', 'N/D'),
            "Max Lunghezza Sequenza": model_params.get('max_pe_len', 'N/D'),
        }
        return info

    except Exception as e:
        logging.error(f"Errore durante l'analisi del modello: {e}", exc_info=True)
        return {"error": f"Impossibile leggere il file del modello.\nErrore: {e}"}

# --- FUNZIONE PRINCIPALE DI GENERAZIONE (da chiamare dalla GUI) ---

def run_generation(model_path, midi_vocab_path, meta_vocab_path, 
                   metadata_prompt, output_dir, total_tokens, temperature, top_k, max_rest_penalty,
                   primer_midi_path=None, update_status_callback=None):
    try:
        def log_and_update(message):
            logging.info(message)
            if update_status_callback: update_status_callback(message)
        log_and_update("Inizio generazione...")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_and_update(f"Device rilevato: {DEVICE}")

        log_and_update("Caricamento vocabolari...")
        midi_tokenizer = miditok.REMI(params=str(midi_vocab_path))
        with open(meta_vocab_path, 'r', encoding='utf-8') as f:
            metadata_vocab_map = json.load(f)['token_to_id']

        # NUOVO: Trova dinamicamente i token di "pausa" (Rest) per la penalizzazione
        rest_ids = None
        if max_rest_penalty > 0:
            # miditok > 2.1.8 usa `vocab.token_to_id`, versioni precedenti potrebbero usare `vocab.token_to_event`
            vocab_map = midi_tokenizer.vocab.token_to_id if hasattr(midi_tokenizer.vocab, 'token_to_id') else midi_tokenizer.vocab.token_to_event
            rest_token_ids_list = [i for t, i in vocab_map.items() if t.startswith("Rest_")]
            if rest_token_ids_list:
                rest_ids = torch.tensor(rest_token_ids_list, device=DEVICE, dtype=torch.long)
                log_and_update(f"Trovati {len(rest_token_ids_list)} token di pausa per la penalizzazione.")
            else:
                log_and_update("ATTENZIONE: Nessun token di pausa trovato nel vocabolario. La penalizzazione non verrà applicata.")

        primer_token_ids = []
        if primer_midi_path and Path(primer_midi_path).exists():
            log_and_update(f"Tokenizzazione del primer MIDI: {primer_midi_path}")
            try:
                primer_tokens_per_track = midi_tokenizer.encode(primer_midi_path)
                if primer_tokens_per_track:
                    primer_token_ids = primer_tokens_per_track[0].ids
                    log_and_update(f"Primer di {len(primer_token_ids)} token caricato con successo.")
                else:
                    log_and_update("ATTENZIONE: Il file MIDI di primer è vuoto o non ha prodotto token. Verrà ignorato.")
            except Exception as e:
                log_and_update(f"ATTENZIONE: Impossibile caricare il primer MIDI. Errore: {e}. Continuo senza.")
        
        log_and_update("Caricamento modello...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model_params = checkpoint.get('model_params')
        if not model_params: raise ValueError("'model_params' non trovato nel checkpoint.")
        
        model = Seq2SeqTransformer(**model_params).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        log_and_update("Modello caricato con successo.")

        # MODIFICATO: Passaggio dei nuovi parametri in generation_config
        generation_config = {
            "temperature": temperature, 
            "top_k": top_k, 
            "max_rest_penalty": max_rest_penalty # Nuovo
        }
        model_chunk_capacity = model_params.get('max_pe_len', 1024)
        log_and_update(f"Prompt metadati: {metadata_prompt}")

        # MODIFICATO: Passaggio di `rest_ids` a `generate_multi_chunk_midi`
        generated_token_ids = generate_multi_chunk_midi(
            model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
            total_target_tokens=total_tokens,
            model_chunk_capacity=model_chunk_capacity,
            generation_config=generation_config,
            device=DEVICE,
            initial_primer_ids=primer_token_ids,
            rest_ids=rest_ids # Nuovo
        )

        if not generated_token_ids: raise RuntimeError("La generazione non ha prodotto token.")
        log_and_update(f"Generati {len(generated_token_ids)} token MIDI.")
        
        log_and_update("Decodifica dei token in MIDI...")
        generated_midi_object = midi_tokenizer.decode(generated_token_ids)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        prompt_name_part = "_".join(metadata_prompt).replace("=", "").replace("/", "")[:50]
        output_filename = f"generated_{prompt_name_part}_{timestamp}.mid"
        output_path = Path(output_dir) / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generated_midi_object.dump_midi(str(output_path))
        
        final_message = f"Successo! File MIDI salvato in:\n{output_path}"
        log_and_update(final_message)
        return final_message
    except Exception as e:
        error_message = f"ERRORE: {e}"
        logging.error("Errore durante la generazione.", exc_info=True)
        if update_status_callback: update_status_callback(error_message)
        return error_message