import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

# Modelo
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Banco fake
bestell_daten = {
    "bestellnummer": ["12345", "67890", "11121", "22232"],
    "status": ["Versandt", "In Bearbeitung", "Geliefert", "Storniert"]
}
df_bestellstatus = pd.DataFrame(bestell_daten)

def pruefe_bestellstatus(bestellnummer):
    try:
        ergebnis = df_bestellstatus[df_bestellstatus['bestellnummer'] == str(bestellnummer)]
        if not ergebnis.empty:
            status = ergebnis['status'].iloc[0]
            return f'Der Status Ihrer Bestellung {bestellnummer} ist: {status}.'
        return 'Bestellnummer nicht gefunden.'
    except:
        return 'Fehler beim Abrufen.'

status_keywords = ['status', 'bestellung', 'paket', 'lieferung', 'tracking']

def generiere_antwort(input_text, chat_history_ids):
    if any(w in input_text.lower() for w in status_keywords):
        return 'Könnten Sie bitte Ihre Bestellnummer eingeben?', chat_history_ids

    new_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_ids], dim=-1)
    else:
        bot_input_ids = new_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    return response, chat_history_ids


with gr.Blocks() as app:
    gr.Markdown("# E-Commerce KI-Support Bot 🤖")

    chatbot = gr.Chatbot(label="Support-Chat")  # SEM type

    msg = gr.Textbox(placeholder="Schreiben Sie hier...")

    chat_state = gr.State(None)
    waiting_for_number = gr.State(False)

    def handle_input(user_input, history, chat_state, waiting):

        if history is None:
            history = []

        if waiting:
            answer = pruefe_bestellstatus(user_input)
            waiting = False
        else:
            answer, chat_state = generiere_antwort(user_input, chat_state)
            if "Bestellnummer" in answer:
                waiting = True

        # 👉 FORMATO CORRETO (messages)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})

        return history, chat_state, waiting, ""

    msg.submit(
        handle_input,
        [msg, chatbot, chat_state, waiting_for_number],
        [chatbot, chat_state, waiting_for_number, msg]
    )

if __name__ == "__main__":
    app.launch()
