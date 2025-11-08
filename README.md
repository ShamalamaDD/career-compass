---

# 🦧 Groovy Gibbon — Full-Stack Setup Guide

Hey team 👋

This repo now contains both the **React (Vite) frontend** and the **FastAPI backend** for our Groovy Gibbon prototype.
Follow this guide carefully to set up both sides locally, even if you’re new to Python or Node.

---

## ⚛️ Frontend — Fix & Setup (after rename)

We recently renamed the frontend folder from **`careercompass-ai` → `frontend-gibbon`**.
Some of you might still have the old version locally, which can cause errors when running `npm install` or `npm run dev`.

Here’s how to clean that up and get the frontend working again.

---

### 🧱 Step 1: Remove the old folder

If you still have `careercompass-ai`, delete it before pulling the latest code.

**Windows**

```bash
cd C:\Users\<yourname>\Documents\GitHub\groovy-gibbon
rmdir /s /q careercompass-ai
```

**macOS / Linux**

```bash
cd ~/Documents/GitHub/groovy-gibbon
rm -rf careercompass-ai
```

---

### ⚙️ Step 2: Pull or clone the latest project

```bash
git pull
```

If it’s your first time:

```bash
git clone https://github.com/<your-org-or-user>/groovy-gibbon.git
cd groovy-gibbon/frontend-gibbon
```

---

### 🧩 Step 3: Open in VS Code

```bash
cd frontend-gibbon
code .
```

---

### 🧾 Step 4: Verify key files

Make sure:

* `package.json` starts with `"name": "frontend-gibbon"`
* `vite.config.ts` has no references to the old project name
* `README.md` mentions `frontend-gibbon`

---

### 🚀 Step 5: Clean and reinstall dependencies

**Windows**

```bash
cd frontend-gibbon
rmdir /s /q node_modules dist .vite
npm install
```

**macOS / Linux**

```bash
cd frontend-gibbon
rm -rf node_modules dist .vite
npm install
```

Then start the dev server:

```bash
npm run dev
```

Visit 👉 [http://localhost:5173](http://localhost:5173)
If the Groovy Gibbon site loads, you’re set 🎉

---

## 🧠 Backend — FastAPI Setup Guide

Now let’s get the backend running in `backend-gibbon`.

---

### 🗂 Project Structure (Explained)

Here’s how the full repo is laid out, with an explanation for every key folder and file:

```
groovy-gibbon/
│
├── backend-gibbon/                # 🧠 FastAPI backend
│   ├── app/                       # Main backend application package
│   │   ├── __init__.py            # Makes 'app' a Python package
│   │   └── main.py                # Entry point: defines API routes, logic, and middleware
│   │
│   ├── gg/                        # Virtual environment (local only; ignored by Git)
│   │   ├── bin/                   # Activation scripts and executables
│   │   ├── lib/                   # Installed Python libraries
│   │   └── pyvenv.cfg             # Virtual environment config
│   │
│   ├── requirements.txt           # Backend dependencies list for reproducibility
│   └── .gitignore                 # Ensures 'gg/' and other local files aren’t committed
│
├── frontend-gibbon/               # ⚛️ React + Vite frontend
│   ├── src/                       # All app code (components, pages, assets)
│   ├── public/                    # Static files served directly (logos, icons)
│   ├── node_modules/              # Installed JS dependencies (auto-generated)
│   ├── package.json               # Project metadata, scripts, dependencies
│   ├── vite.config.ts             # Vite build + dev server config
│   ├── tailwind.config.js         # Tailwind CSS configuration
│   └── tsconfig.json              # TypeScript compiler settings
│
└── README.md                      # You’re reading it — full project guide
```

This structure is designed to cleanly separate the **frontend app** (Vite + React + Tailwind)
from the **backend service** (FastAPI + spaCy NLP).

---

### ⚙️ Step 1: Create a virtual environment

**macOS / Linux**

```bash
cd backend-gibbon
python3 -m venv gg
source gg/bin/activate
```

**Windows**

```powershell
cd backend-gibbon
python -m venv gg
gg\Scripts\activate
```

---

### 📦 Step 2: Install dependencies

If `requirements.txt` exists:

```bash
pip install -r requirements.txt
```

Otherwise install manually:

```bash
pip install fastapi uvicorn spacy PyPDF2 matplotlib seaborn python-multipart
pip freeze > requirements.txt
```

---

### 🧠 Step 3: Download spaCy language model

```bash
python -m spacy download en_core_web_md
```

---

### 🚀 Step 4: Run the backend

```bash
uvicorn app.main:app --reload
```

Then open 👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

If you see the **Groovy Gibbon – CV Ranking API** docs, it’s running fine.

---

### 🔄 Step 5: Reset the environment (if it breaks)

**macOS / Linux**

```bash
rm -rf gg
python3 -m venv gg
source gg/bin/activate
pip install -r requirements.txt
```

**Windows**

```powershell
rd /s /q gg
python -m venv gg
gg\Scripts\activate
pip install -r requirements.txt
```

---

### 🔗 Frontend ↔ Backend Integration

CORS is already configured to accept requests from:

```
http://localhost:5173
```

so the React app can freely call backend endpoints during local development.

---

### 🧩 Current Backend Status

✅ FastAPI app launches successfully
✅ `/api/rank` endpoint exists (placeholder implementation)
🚧 Semantic ranking logic still to be added with Sentence Transformers

---

### 💬 Recommended Commit Message

```
init: setup FastAPI backend with placeholder /api/rank endpoint and CORS configuration
```

---

### ✅ Final Snapshot

```
groovy-gibbon/
├── backend-gibbon/
│   ├── app/
│   │   └── main.py
│   ├── gg/
│   └── requirements.txt
│
├── frontend-gibbon/
│   ├── src/
│   ├── public/
│   └── package.json
│
└── README.md
```

---

Now everyone can set up, run, and understand both sides of Groovy Gibbon without confusion — one repo, clean separation, and fully documented setup.
