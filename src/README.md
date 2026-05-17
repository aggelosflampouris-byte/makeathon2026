# 🧾 Invoice RAG Pipeline — Makeathoon 2026

Ένα full-stack σύστημα αυτόματης επεξεργασίας τιμολογίων με OCR, Vector Store και Conversational AI.

---

## 🔐 Ασφάλεια & Διαχείριση API Keys

> **Αυστηρή απαίτηση**: Κανένα API key, token ή credential δεν επιτρέπεται να βρίσκεται μέσα στον κώδικα ή στο repository.

Όλα τα ευαίσθητα δεδομένα διαχειρίζονται αποκλειστικά μέσω αρχείων περιβάλλοντος (`.env`).

---

## 🚀 Γρήγορη Εκκίνηση

### 1. Κλωνοποίηση & Εγκατάσταση

```bash
git clone <repository-url>
cd Makeathoon2026
pip install -r requirements.txt
```

### 2. Δημιουργία `.env` από το template

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

### 3. Συμπλήρωση του `.env`

Ανοίξτε το αρχείο `.env` και συμπληρώστε τα API keys σας:

```env
GEMINI_API_KEYS=your_gemini_api_key_1,your_gemini_api_key_2
```

### 4. Εκκίνηση

```bash
python main.py
```

Ο server ξεκινά στο `http://127.0.0.1:8000`.

---

## 🔑 Απόκτηση Gemini API Key

| Βήμα | Ενέργεια |
|------|----------|
| 1 | Μεταβείτε στο **[Google AI Studio](https://aistudio.google.com/app/apikey)** |
| 2 | Συνδεθείτε με τον Google λογαριασμό σας |
| 3 | Κάντε κλικ στο **"Create API key"** |
| 4 | Αντιγράψτε το κλειδί |
| 5 | Επικολλήστε το στο `.env` σας στην μεταβλητή `GEMINI_API_KEYS` |

> **💡 Tip:** Μπορείτε να προσθέσετε πολλαπλά keys διαχωρισμένα με κόμμα. Το σύστημα τα εναλλάσσει αυτόματα για να αποφύγει τα rate limits.

---

## 📁 Δομή Αρχείων Ασφαλείας

```
Makeathoon2026/
├── .env              ← ✅ Τοπικό αρχείο με τα ΠΡΑΓΜΑΤΙΚΑ keys (gitignored)
├── .env.example      ← ✅ Ασφαλές template για το repository (χωρίς keys)
├── .gitignore        ← ✅ Εξαιρεί το .env, chroma_db, __pycache__ κ.ά.
└── ...
```

### Τι προστατεύεται από το `.gitignore`

| Αρχείο/Φάκελος | Λόγος |
|---|---|
| `.env` | Περιέχει API keys |
| `chroma_db/` | Μεγάλα δυαδικά δεδομένα vector store |
| `__pycache__/` | Compiled Python bytecode |
| `batch_*/` | Uploaded invoices (ευαίσθητα οικονομικά δεδομένα) |
| `*.log` | Logs που μπορεί να περιέχουν sensitive data |

---

## ⚙️ Μεταβλητές Περιβάλλοντος

| Μεταβλητή | Υποχρεωτική | Περιγραφή |
|---|---|---|
| `GEMINI_API_KEYS` | ✅ Ναι | Comma-separated Gemini API keys |
| `TESSERACT_CMD` | ❌ Όχι | Path στο `tesseract.exe` (auto-detect αν λείπει) |

---

## 🏗️ Αρχιτεκτονική Pipeline

```
Invoice (PDF/Image)
        │
        ▼
  [Step 1] OCR Engine (Tesseract + Gemini Vision)
        │   └─ qwentest.py → OCRResult
        │
        ▼
  [Step 2] Vector Store Ingestion
        │   └─ vector_store.py → ChromaDB (chroma_db/)
        │
        ▼
  [Step 3] RAG Query Engine
        │   └─ rag_engine.py → Gemini Text API
        │
        ▼
  [Step 4] Web Interface
            └─ FastAPI (main.py) → front.html
```

---

## 🛡️ Compliance Summary

| Κανόνας | Υλοποίηση |
|---|---|
| Χωρίς hardcoded keys | ✅ Όλα τα keys διαβάζονται από `os.getenv()` |
| `.env` gitignored | ✅ Αναγράφεται στο `.gitignore` |
| Safe template στο repo | ✅ `.env.example` χωρίς πραγματικές τιμές |
| Fail-fast αν λείπουν keys | ✅ `EnvironmentError` κατά την εκκίνηση |
| Rotation πολλαπλών keys | ✅ Αυτόματο key × model retry matrix |

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

Βεβαιωθείτε ότι **Tesseract OCR** είναι εγκατεστημένο:
- Windows: [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt install tesseract-ocr tesseract-ocr-ell`
