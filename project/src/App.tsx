import { useRef, useState } from "react";
import SliderInput from "./SliderInput";
import type { TrainingConfig } from './types.ts'

export default function App() {

  const [consoleText, setConsoleText] = useState("Waiting for training...\n")


  const formRef = useRef<HTMLFormElement | null>(null);

  async function handleStartTraining(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();

    const form = formRef.current;
    if (!form) return;

    const formData = new FormData(form);

    const config: TrainingConfig = {
      dataset: (formData.get("dataset") as File) ?? null,
      output_dir: String(formData.get("output_dir") ?? ""),
      epochs: Number(formData.get("epochs") ?? 0),
      steps: Number(formData.get("steps") ?? 0),
      batch_size: Number(formData.get("batch_size") ?? 0),
      learning_rate: Number(formData.get("learning_rate") ?? 0),
      save_every: Number(formData.get("save_every") ?? 0),
      warmup: Number(formData.get("warmup") ?? 0),
      seed: Number(formData.get("seed") ?? 0),
      fp16: formData.get("fp16") === "on",
      resume: formData.get("resume") === "on",
      cache_dataset: formData.get("cache_dataset") === "on",
      shuffle_dataset: formData.get("shuffle_dataset") === "on",
    };

    console.log("Training config:", config);

     try {
        const response = await fetch("http://127.0.0.1:5000/train", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(config),
        });

        if (!response.ok) {
            throw new Error("Failed to start training");
        }

        const result = await response.json();
        setConsoleText(
            `$ ${result.command.join(" ")}
            ${result.stdout}
            ${result.stderr}`
        )
    } catch (err) {
        setConsoleText(            
            `${err}`)
    }
  }

  return (
    <div className="app">
      <form ref={formRef} className="training-form" onSubmit={handleStartTraining}>
        <aside className="sidebar">
          <h1>Model Trainer</h1>

          <section className="group">
            <h2>Files</h2>

            <label className="field">
              Dataset
              <input type="file" name="dataset" />
            </label>

            <label className="field">
              Output Folder
              <input
                type="text"
                name="output_dir"
                placeholder="Select folder..."
                defaultValue=""
              />
            </label>
          </section>

          <section className="group">
            <h2>Training</h2>

            <SliderInput
              label="Epochs"
              name="epochs"
              min={1}
              max={100}
              defaultValue={4}
            />

            <SliderInput
              label="Steps"
              name="steps"
              min={1}
              max={5000}
              defaultValue={20}
            />

            <SliderInput
              label="Batch Size"
              name="batch_size"
              min={1}
              max={64}
              defaultValue={8}
            />

            <SliderInput
              label="Learning Rate"
              name="learning_rate"
              min={0.000001}
              max={0.01}
              step={0.000001}
              defaultValue={0.0001}
            />

            <SliderInput
              label="Save Every"
              name="save_every"
              min={50}
              max={5000}
              defaultValue={500}
            />

            <SliderInput
              label="Warmup"
              name="warmup"
              min={0}
              max={1000}
              defaultValue={100}
            />

            <SliderInput
              label="Seed"
              name="seed"
              min={0}
              max={999999}
              defaultValue={42}
            />
          </section>

          <section className="group">
            <label className="check">
              <input type="checkbox" name="fp16" />
              Mixed Precision (FP16)
            </label>

            <label className="check">
              <input type="checkbox" name="resume" />
              Resume Training
            </label>

            <label className="check">
              <input type="checkbox" name="cache_dataset" />
              Cache Dataset
            </label>

            <label className="check">
              <input type="checkbox" name="shuffle_dataset" defaultChecked />
              Shuffle Dataset
            </label>
          </section>

          <button className="start" type="submit">
            Start Training
          </button>
        </aside>

        <main className="console">
          <div className="consoleHeader">Console</div>
          <pre>{`${consoleText}`}</pre>
        </main>
      </form>
    </div>
  );
}
