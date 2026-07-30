# AI Training Web Interface

A lightweight React + FastAPI web interface for launching AI training scripts from a browser instead of the command line.

The frontend collects training parameters such as epochs, steps, batch size, learning rate, and other options, then sends them as JSON to a FastAPI backend. The backend converts those parameters into command-line arguments and launches `train.py`.

# IMPORTANT
The train.py included in this repo is mine, I've made it with Google's Flan T5 Base model in mind (present in DEFAULT_MODEL_ID = "google/flan-t5-base" in train.py).
As is it just a placeholder, you can include your train.py if you want to use my web interface instead of another one for some reason. I've mainly made this because my train.py worked in a T5 encoder-decoder model, which is not *widely* supported, so I've made the train.py customized for that. But editing the code file itself was getting pretty stale, so I've made the web interface.

## Features

* Modern React + TypeScript frontend
* FastAPI backend
* Automatic conversion of JSON parameters into CLI arguments
* Compatible with existing Python training scripts
* No modifications required to `train.py` as long as it accepts command-line arguments (and follows the expected format)

## How it Works

```
React UI
    │
    ▼
JSON Configuration
    │
    ▼
FastAPI (/train)
    │
    ▼
Build command:
python train.py --epochs 4 --steps 20 ...
    │
    ▼
subprocess.run(...)
    │
    ▼
Return stdout, stderr and exit code
```

## Project Structure

```
project/
│
├── frontend/
│   ├── src/
│   ├── App.tsx
│   └── ...
│
├── backend/
│   ├── main.py
│   └── train.py
│
└── README.md
```

## Installation

### Backend

Install the required packages:

```bash
pip install fastapi uvicorn
```

Start the server:

```bash
python main.py
```

The API will be available at:

```
http://127.0.0.1:5000
```

### Frontend

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

By default, Vite runs on:

```
http://localhost:5173
```

## API

### POST `/train`

Receives a JSON object containing the training configuration.

Example:

```json
{
  "epochs": 4,
  "steps": 20,
  "batch_size": 8,
  "learning_rate": 0.0001,
  "fp16": true,
  "resume": false
}
```

The backend automatically converts this into:

```bash
python train.py \
    --epochs 4 \
    --steps 20 \
    --batch-size 8 \
    --learning-rate 0.0001 \
    --fp16
```

Boolean values are treated as flags:

* `true` → `--flag`
* `false` → omitted

## Backend

The FastAPI server:

* receives the JSON configuration
* converts each key into a command-line argument
* executes `train.py`
* returns:

  * generated command
  * stdout
  * stderr
  * process return code


## License

MIT
