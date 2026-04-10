---
title: E-commerce KI-Support Bot
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
python_version: 3.11
---

# 🤖 E-Commerce KI-Support Bot

Dieses Projekt ist ein intelligenter Chatbot, der für den Kundensupport im E-Commerce entwickelt wurde. Er kombiniert natürliche Sprachverarbeitung (NLP) mit einer logischen Abfrage von Bestelldaten.

### 🔗 Links úteis
* **Live Demo (Hugging Face):** [Klicken Sie hier, um den Chatbot zu testen](https://huggingface.co/spaces/Ruanaramos/ecommerce-chatbot-de)
* **Status do Build:** ![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)

---
## 🚀 Funktionalitäten
* **Automatisierter Support:** Beantwortet allgemeine Fragen auf Deutsch.
* **Bestellstatus-Abfrage:** Erkennt Anfragen zu Bestellungen und liefert Echtzeit-Status aus einer Datenbank (simuliert mit Pandas).
* **Kontextbewusstsein:** Behält den Gesprächsverlauf bei, um ein flüssiges Chat-Erlebnis zu bieten.

## 🛠️ Verwendete Technologien
* **Modell:** [Microsoft DialoGPT-medium](https://huggingface.co/microsoft/DialoGPT-medium) (Transformers).
* **Interface:** [Gradio 6+](https://gradio.app/) für eine moderne Web-UI.
* **Hosting:** [Hugging Face Spaces](https://huggingface.co/spaces) mit automatischer Synchronisation.
* **Infrastruktur:** **GitHub Actions (CI/CD)** sorgt dafür, dass jede Code-Änderung sofort auf dem Server aktualisiert wird.

## 📁 Projektstruktur
* `app.py`: Die Hauptlogik des Bots und die Gradio-Schnittstelle.
* `requirements.txt`: Liste der notwendigen Python-Bibliotheken.
* `.github/workflows/sync.yml`: Automatisierungsskript für das Deployment.

---
*Entwickelt als Demonstration für KI-gestützten Kundenservice.*
