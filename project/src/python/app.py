from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
#for communication with vite
app.add_middleware(
    #all of this is probably unsafe ngl
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Job(BaseModel):
    input: str
    output: str
    quality: int

@app.get("/")
def home():
    return{"status": "running"}

@app.post("/train")
def train(config:dict):
    command = ["python", "train.py"]
    
    ignored = {
    "skip_steps",
    "fp16",
    "resume",
    "cache_dataset",
    "shuffle_dataset",
    }

    for key, value in config.items():
        if key in ignored:
            continue

        flag = "--" + key.replace("_", "-")
        command.extend([flag, str(value)])   
            

        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    return {
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5000) #change if necessary, mine works fine
