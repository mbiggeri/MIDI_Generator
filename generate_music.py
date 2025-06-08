# generate_music.py (versione libreria con callback di progresso e KV CACHING)
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for Mac M1 compatibility

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
from typing import Optional, List, Dict

# Soppressione di avvisi comuni di miditok
warnings.filterwarnings("ignore", category=UserWarning, module='miditok.midi_tokenizer_base')

# --- DEFINIZIONI DEL MODELLO ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # --- MODIFICA: La forma ora è (1, max_len, d_model) per essere compatibile con batch_first=True
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # --- MODIFICA: Logica adattata per batch_first=True
        # x ha forma (batch, seq_len, emb_size)
        x = x + self.pe[:, :x.size(1)]
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
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True # batch_first=True è cruciale
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
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: Optional[torch.Tensor] = None,
               cache: Optional[List[Dict[str, torch.Tensor]]] = None) -> tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Esegue un passo di decodifica. Se viene fornito un cache, si assume che `tgt` contenga solo
        l'ultimo token, e si usano i valori K e V cachati per l'attenzione.
        """
        is_primer_pass = cache is None

        if is_primer_pass:
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))
        else:
            past_len = cache[0]['k'].size(1)
            emb = self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size)
            tgt_emb = emb + self.positional_encoding.pe[:, past_len:past_len + 1]

        new_cache = []
        output = tgt_emb

        for i, layer in enumerate(self.transformer.decoder.layers):
            past_layer_cache = cache[i] if cache is not None else None

            query = output
            key = output
            value = output

            if past_layer_cache is not None:
                past_key = past_layer_cache['k']
                past_value = past_layer_cache['v']
                key = torch.cat([past_key, key], dim=1)
                value = torch.cat([past_value, value], dim=1)
            
            new_cache.append({'k': key, 'v': value})
            
            attn_mask = None
            if is_primer_pass:
                tgt_len = key.size(1)
                attn_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool), diagonal=1)

            attn_output, _ = layer.self_attn(query, key, value, attn_mask=attn_mask, need_weights=False)
            
            output = layer.norm1(output + layer.dropout1(attn_output))

            cross_attn_output, _ = layer.multihead_attn(output, memory, memory,
                                                         key_padding_mask=memory_key_padding_mask,
                                                         need_weights=False)
            output = layer.norm2(output + layer.dropout2(cross_attn_output))

            ff_output = layer.linear2(layer.dropout(F.relu(layer.linear1(output))))
            output = layer.norm3(output + layer.dropout3(ff_output))

        # --- INIZIO FIX FINALE ---
        # Applichiamo il layer lineare finale per convertire l'embedding in logits.
        # Questa era la riga mancante che causava l'errore di dimensione.
        logits = self.generator(output)

        # Se abbiamo processato un primer, restituiamo solo i logits dell'ultimo token
        if is_primer_pass:
            logits = logits[:, -1:, :]
        # --- FINE FIX FINALE ---

        return logits, new_cache
    
# --- FUNZIONI DI GENERAZIONE ---
# --- MODIFICA: La funzione `generate_sequence` è riscritta per usare il caching KV ---
def generate_sequence(model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                      max_new_tokens, min_new_tokens, temperature, top_k, device,
                      primer_token_ids, model_max_pe_len, max_rest_penalty, rest_ids,
                      progress_callback=None):
    model.eval()
    META_SOS_TOKEN_NAME, META_EOS_TOKEN_NAME, META_UNK_TOKEN_NAME, META_PAD_TOKEN_NAME = "<sos_meta>", "<eos_meta>", "<unk_meta>", "<pad_meta>"
    MIDI_SOS_TOKEN_NAME, MIDI_EOS_TOKEN_NAME = "SOS_None", "EOS_None"
    try:
        sos_meta_id, eos_meta_id, unk_meta_id, meta_pad_id = metadata_vocab_map[META_SOS_TOKEN_NAME], metadata_vocab_map[META_EOS_TOKEN_NAME], metadata_vocab_map[META_UNK_TOKEN_NAME], metadata_vocab_map[META_PAD_TOKEN_NAME]
        sos_midi_id, eos_midi_id = midi_tokenizer[MIDI_SOS_TOKEN_NAME], midi_tokenizer[MIDI_EOS_TOKEN_NAME]
    except KeyError as e:
        logging.error(f"Token speciale '{e}' non trovato nei vocabolari.")
        raise ValueError(f"Token speciale '{e}' mancante.")

    with torch.no_grad():
        # 1. Codifica del prompt (invariato)
        meta_token_ids = [metadata_vocab_map.get(token, unk_meta_id) for token in metadata_prompt]
        src_seq = torch.tensor([[sos_meta_id] + meta_token_ids + [eos_meta_id]], dtype=torch.long, device=device)
        src_padding_mask = (src_seq == meta_pad_id)
        memory = model.encode(src_seq, src_padding_mask)

        # 2. Preparazione della generazione
        initial_ids = [sos_midi_id] + (primer_token_ids if primer_token_ids else [])
        
        # --- NUOVA LOGICA CON CACHE ---
        # Il primo input al decodificatore contiene tutti i token iniziali
        current_tokens = torch.tensor([initial_ids], dtype=torch.long, device=device)
        generated_ids = []
        cache = None

        # Processiamo l'input iniziale (primer) tutto in una volta per popolare il cache
        if len(initial_ids) > 0:
            # Decodifica l'intero primer per ottenere l'ultimo logit e il primo cache
            logits, cache = model.decode(current_tokens, memory, memory_key_padding_mask=src_padding_mask)
            # Usiamo solo l'output dell'ultimo token del primer
            logits = logits[:, -1, :]
        else:
            # Se non c'è primer, iniziamo con il token SOS
            current_tokens = torch.tensor([[sos_midi_id]], dtype=torch.long, device=device)
            logits, cache = model.decode(current_tokens, memory, memory_key_padding_mask=src_padding_mask)


        for i in range(max_new_tokens):
            if len(initial_ids) + len(generated_ids) >= model_max_pe_len:
                logging.warning(f"Raggiunta capacità massima del modello ({model_max_pe_len}). Interruzione.")
                break

            # 3. Campionamento del prossimo token (logica invariata, ma applicata a `logits`)
            if max_rest_penalty > 0 and rest_ids is not None and rest_ids.numel() > 0:
                progress = i / max_new_tokens
                current_penalty = max_rest_penalty * progress
                logits[:, rest_ids] -= current_penalty
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
            generated_ids.append(next_token_id)
            
            # Logica di terminazione (invariata)
            if next_token_id == eos_midi_id and len(generated_ids) < min_new_tokens:
                 _, top_2_indices = torch.topk(probs, 2, dim=-1)
                 if top_2_indices[0, 0].item() == eos_midi_id and top_2_indices.size(1) > 1:
                     next_token_id_tensor = top_2_indices[0, 1].unsqueeze(0).unsqueeze(0)
                     generated_ids[-1] = next_token_id_tensor.item() # Correggiamo l'ultimo token
                 else: break
            if next_token_id == eos_midi_id and len(generated_ids) >= min_new_tokens:
                break
            
            # --- AGGIORNAMENTO DEL PROGRESSO (invariato) ---
            if progress_callback:
                progress_callback(i + 1)

            # 4. Chiamata al decodificatore con il solo ultimo token e il cache
            # Questo è il cuore dell'ottimizzazione KV Caching
            logits, cache = model.decode(next_token_id_tensor, memory,
                                          memory_key_padding_mask=src_padding_mask, cache=cache)
            logits = logits.squeeze(1)

    return generated_ids

# --- Il resto del file (generate_multi_chunk_midi, get_model_info, run_generation) rimane invariato ---
# ... (copia qui il resto del tuo file da generate_multi_chunk_midi in poi)
# --- MODIFICATO: Aggiunto `progress_callback` ---
def generate_multi_chunk_midi(model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
                              total_target_tokens, model_chunk_capacity, generation_config, device,
                              initial_primer_ids=None, rest_ids=None, progress_callback=None):
    all_generated_tokens = []
    current_primer_ids = initial_primer_ids.copy() if initial_primer_ids else [] 
    PRIMER_TOKEN_COUNT = 50
    MIN_TOKENS_PER_CHUNK = 100
    max_new_tokens_per_chunk = min(2048, model_chunk_capacity - PRIMER_TOKEN_COUNT - 5)
    eos_midi_id = midi_tokenizer["EOS_None"]

    while len(all_generated_tokens) < total_target_tokens:
        tokens_generated_so_far = len(all_generated_tokens)
        if progress_callback:
            progress_percentage = (tokens_generated_so_far / total_target_tokens) * 100 if total_target_tokens > 0 else 0
            progress_callback(progress_percentage)
        
        chunk_progress_callback = None
        if progress_callback:
            def chunk_progress_callback_wrapper(newly_generated_in_chunk):
                total_generated = tokens_generated_so_far + newly_generated_in_chunk
                overall_percentage = (total_generated / total_target_tokens) * 100
                progress_callback(overall_percentage)
            chunk_progress_callback = chunk_progress_callback_wrapper

        remaining_tokens_to_generate = total_target_tokens - len(all_generated_tokens)
        current_pass_max_new = min(max_new_tokens_per_chunk, remaining_tokens_to_generate)
        if len(current_primer_ids) + current_pass_max_new + 1 > model_chunk_capacity: break
        
        newly_generated_ids = generate_sequence(
            model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
            max_new_tokens=current_pass_max_new,
            min_new_tokens=min(MIN_TOKENS_PER_CHUNK, current_pass_max_new),
            temperature=generation_config.get("temperature", 0.75),
            top_k=generation_config.get("top_k", 40),
            max_rest_penalty=generation_config.get("max_rest_penalty", 0.0),
            device=device,
            primer_token_ids=current_primer_ids,
            model_max_pe_len=model_chunk_capacity,
            rest_ids=rest_ids,
            progress_callback=chunk_progress_callback
        )
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

# --- FUNZIONE DI ANALISI MODELLO (invariata) ---
def get_model_info(model_path: str) -> dict:
    if not model_path or not Path(model_path).exists():
        return {"error": "Selezionare un percorso valido per il modello."}
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_params = checkpoint.get('model_params')
        if not model_params:
            return {"error": "'model_params' non trovato nel checkpoint del modello."}
        model = Seq2SeqTransformer(**model_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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

# --- FUNZIONE PRINCIPALE DI GENERAZIONE ---
def run_generation(model_path, midi_vocab_path, meta_vocab_path, 
                   metadata_prompt, output_dir, total_tokens, temperature, top_k, max_rest_penalty,
                   primer_midi_path=None, update_status_callback=None, progress_callback=None):
    try:
        def log_and_update(message):
            logging.info(message)
            if update_status_callback: update_status_callback(message)
        
        log_and_update("Inizio generazione...")
        # --- MODIFICA per il fallback su Mac M1 ---
        # Se usi un Mac con Apple Silicon, questo abiliterà il fallback su CPU per le operazioni non supportate
        # Rimuovilo o commentalo se non sei su Mac o se causa problemi.
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        if torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
        elif torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device("cpu")
            
        log_and_update(f"Device rilevato: {DEVICE}")

        log_and_update("Caricamento vocabolari...")
        midi_tokenizer = miditok.REMI(params=str(midi_vocab_path))
        with open(meta_vocab_path, 'r', encoding='utf-8') as f:
            metadata_vocab_map = json.load(f)['token_to_id']

        rest_ids = None
        if max_rest_penalty > 0:
            vocab_map = midi_tokenizer.vocab
            rest_token_ids_list = [i for t, i in vocab_map.items() if t.startswith("Rest_")]
            if rest_token_ids_list:
                rest_ids = torch.tensor(rest_token_ids_list, device=DEVICE, dtype=torch.long)
                log_and_update(f"Trovati {len(rest_token_ids_list)} token di pausa per la penalizzazione.")
            else:
                log_and_update("ATTENZIONE: Nessun token di pausa trovato. Penalizzazione non applicata.")

        primer_token_ids = []
        if primer_midi_path and Path(primer_midi_path).exists():
            log_and_update(f"Tokenizzazione del primer MIDI: {primer_midi_path}")
            try:
                primer_tokens_per_track = midi_tokenizer.encode(primer_midi_path)
                if primer_tokens_per_track:
                    primer_token_ids = primer_tokens_per_track[0].ids
                    log_and_update(f"Primer di {len(primer_token_ids)} token caricato.")
                else:
                    log_and_update("ATTENZIONE: Primer MIDI vuoto o invalido. Verrà ignorato.")
            except Exception as e:
                log_and_update(f"ATTENZIONE: Impossibile caricare il primer MIDI. Errore: {e}. Continuo senza.")
        
        log_and_update("Caricamento modello...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model_params = checkpoint.get('model_params')
        if not model_params: raise ValueError("'model_params' non trovato nel checkpoint.")
        
        model = Seq2SeqTransformer(**model_params).to(DEVICE)
        
        # Questo codice corregge il mismatch di forma del positional encoding per i vecchi checkpoint.
        state_dict = checkpoint['model_state_dict']
        pe_key = 'positional_encoding.pe'

        # Controlliamo se la chiave esiste e se la forma è quella vecchia
        if pe_key in state_dict and state_dict[pe_key].shape[0] != 1 and state_dict[pe_key].shape[1] == 1:
            log_and_update(f"Rilevato formato 'vecchio' per il tensore '{pe_key}'. Adattamento della forma in corso...")
            # Trasponiamo le prime due dimensioni per passare da [seq, 1, emb] a [1, seq, emb]
            state_dict[pe_key] = state_dict[pe_key].transpose(0, 1)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        log_and_update("Modello caricato con successo.")

        generation_config = {"temperature": temperature, "top_k": top_k, "max_rest_penalty": max_rest_penalty}
        model_chunk_capacity = model_params.get('max_pe_len', 2048) # Default aumentato a 2048
        log_and_update(f"Prompt metadati: {metadata_prompt}")

        generated_token_ids = generate_multi_chunk_midi(
            model, midi_tokenizer, metadata_vocab_map, metadata_prompt,
            total_target_tokens=total_tokens,
            model_chunk_capacity=model_chunk_capacity,
            generation_config=generation_config,
            device=DEVICE,
            initial_primer_ids=primer_token_ids,
            rest_ids=rest_ids,
            progress_callback=progress_callback
        )

        if not generated_token_ids: raise RuntimeError("La generazione non ha prodotto token.")
        
        log_and_update(f"Generati {len(generated_token_ids)} token MIDI. Decodifica in corso...")
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