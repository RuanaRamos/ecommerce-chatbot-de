import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

# Laden do modelo e tokenizador
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Datenbank simulada
bestell_daten = {
    "bestellnummer": ["12345", "67890", "11121", "22232"],
    "status": ["Versandt", "In Bearbeitung", "Geliefert", "Storniert"]
}
df_bestellstatus = pd.DataFrame(bestell_daten)

def pruefe_bestellstatus(bestellnummer):
    """Überprüft den Status einer Bestellung."""
    try:
        ergebnis = df_bestellstatus[df_bestellstatus['bestellnummer'] == str(bestellnummer)]
        if not ergebnis.empty:
            status = ergebnis['status'].iloc[0]
            return f'Der Status Ihrer Bestellung {bestellnummer} ist: {status}.'
        return 'Bestellnummer nicht gefunden. Bitte prüfen Sie Ihre Eingabe.'
    except Exception:
        return 'Ein Fehler ist aufgetreten.'

status_schluesselwoerter = ['status', 'bestellung', 'wo ist mein paket', 'lieferung', 'tracking']

def generiere_antwort(benutzer_eingabe, chat_historie_ids):
    """Generiert eine Antwort com DialoGPT ou identifica intenção de status."""
    if any(wort in benutzer_eingabe.lower() for wort in status_schluesselwoerter):
        return 'Könnten Sie bitte Ihre Bestellnummer eingeben?', chat_historie_ids
    
    neu_benutzer_input_ids = tokenizer.encode(benutzer_eingabe + tokenizer.eos_token, return_tensors='pt')
    
    if chat_historie_ids is not None:
        bot_input_ids = torch.cat([chat_historie_ids, neu_benutzer_input_ids], dim=-1)
    else:
        bot_input_ids = neu_benutzer_input_ids

    chat_historie_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )
    
    antwort = tokenizer.decode(chat_historie_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return antwort, chat_historie_ids

with gr.Blocks() as app:
    gr.Markdown("# E-Commerce KI-Support Bot 🤖")
    
    # Removido o parâmetro 'type' para garantir compatibilidade
    chatbot = gr.Chatbot(label="Support-Chat")
    msg = gr.Textbox(placeholder='Schreiben Sie hier...', label="Ihre Nachricht")

    chat_status = gr.State(None) 
    wartet_auf_nummer = gr.State(False)

    def verarbeite_eingabe(benutzer_eingabe, historie, chat_status, ist_am_warten):
        # Garante que a história comece como uma lista vazia
        if historie is None: historie = []
        
        if ist_am_warten:
            antwort = pruefe_bestellstatus(benutzer_eingabe)
            ist_am_warten = False
        else:
            antwort, chat_status = generiere_antwort(benutzer_eingabe, chat_status)
            if antwort == 'Könnten Sie please enter your order number?' or 'Könnten Sie bitte Ihre Bestellnummer eingeben?' in antwort:
                ist_am_warten = True

        # Formato de TUPLA (Pergunta, Resposta) compatível com versões antigas
        historie.append((benutzer_eingabe, antwort))
        
        return historie, chat_status, ist_am_warten, ""

    msg.submit(
        verarbeite_eingabe,
        [msg, chatbot, chat_status, wartet_auf_nummer],
        [chatbot, chat_status, wartet_auf_nummer, msg]
    )

if __name__ == "__main__":
    app.launch()
