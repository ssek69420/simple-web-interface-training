import { useRef, useState } from "react";
import SliderInput from "./SliderInput";
import type { TrainingConfig } from './types.ts'
import "./App.css"

export default function App() {

  const [consoleText, setConsoleText] = useState("Waiting for training...\n")
 

  const formRef = useRef<HTMLFormElement | null>(null);

  async function handleStartTraining(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();

    const form = formRef.current;
    if (!form) return;

    const formData = new FormData(form);

    const config: TrainingConfig = {
      model: String(formData.get("model") ?? ""),
      dataset: (formData.get("dataset") as File) ?? null,
      output_dir: String(formData.get("output_dir") ?? ""),
      best_model_dir: String(formData.get("best_model_dir") ?? ""),
      prefix: String(formData.get("prefix") ?? ""),
      input_column: String(formData.get("input_column") ?? "input"),
      target_column: String(formData.get("target_column") ?? "target"),
      max_input_length: Number(formData.get("max_input_length") ?? 512),
      max_target_length: Number(formData.get("max_target_length") ?? 512),
      test_size: Number(formData.get("test_size") ?? 0.1),
      learning_rate: Number(formData.get("learning_rate") ?? 0.0001),
      epochs: Number(formData.get("epochs") ?? 4),
      train_batch_size: Number(formData.get("train_batch_size") ?? 8),
      eval_batch_size: Number(formData.get("eval_batch_size") ?? 8),
      gradient_accumulation_steps: Number(formData.get("gradient_accumulation_steps") ?? 1),
      weight_decay: Number(formData.get("weight_decay") ?? 0.01),
      warmup_ratio: Number(formData.get("warmup_ratio") ?? 0.0),
      logging_steps: Number(formData.get("logging_steps") ?? 10),
      save_total_limit: Number(formData.get("save_total_limit") ?? 3),
      patience: Number(formData.get("patience") ?? 3),
      seed: Number(formData.get("seed") ?? 42),
      max_train_samples: Number(formData.get("max_train_samples") ?? 0),
      max_eval_samples: Number(formData.get("max_eval_samples") ?? 0),
      fp16: formData.get("fp16") === "on",
      resume: formData.get("resume") === "on",
      cache_dataset: formData.get("cache_dataset") === "on",
      shuffle_dataset: formData.get("shuffle_dataset") === "on",
      skip_steps: formData.get("skip_steps") === "on",
    };

    console.log("Prefix: ", config.prefix);
    console.log("Training config:", config);

    const skipSteps = formData.get("skip_steps") === "on";

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
          <input className="prefixInput"
            type="text"
            name="prefix"
            placeholder="Add a prefix for model training"
            required
          />
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

            <label className="field">
              Model
              <input
                type="text"
                name="model"
                placeholder="HuggingFace model name/path"
              />
            </label>

            <label className="field">
              Best Model Folder
              <input
                type="text"
                name="best_model_dir"
                placeholder="Best model output folder"
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
              label="Train Batch Size"
              name="train_batch_size"
              min={1}
              max={64}
              defaultValue={8}
            />

            <SliderInput
              label="Eval Batch Size"
              name="eval_batch_size"
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
              label="Warmup Ratio"
              name="warmup_ratio"
              min={0}
              max={1}
              step={0.01}
              defaultValue={0.1}
            />

            <SliderInput
              label="Weight Decay"
              name="weight_decay"
              min={0}
              max={1}
              step={0.01}
              defaultValue={0.01}
            />

            <SliderInput
              label="Gradient Accumulation Steps"
              name="gradient_accumulation_steps"
              min={1}
              max={64}
              defaultValue={1}
            />

            <SliderInput
              label="Max Input Length"
              name="max_input_length"
              min={32}
              max={4096}
              defaultValue={512}
            />

            <SliderInput
              label="Max Target Length"
              name="max_target_length"
              min={32}
              max={4096}
              defaultValue={512}
            />

            <SliderInput
              label="Test Size (%)"
              name="test_size"
              min={0}
              max={1}
              step={0.01}
              defaultValue={0.1}
            />

            <SliderInput
              label="Logging Steps"
              name="logging_steps"
              min={1}
              max={1000}
              defaultValue={10}
            />

            <SliderInput
              label="Save Total Limit"
              name="save_total_limit"
              min={1}
              max={20}
              defaultValue={3}
            />

            <SliderInput
              label="Patience"
              name="patience"
              min={0}
              max={20}
              defaultValue={3}
            />

            <SliderInput
              label="Seed"
              name="seed"
              min={0}
              max={999999}
              defaultValue={42}
            />

            <SliderInput
              label="Max Train Samples"
              name="max_train_samples"
              min={0}
              max={100000}
            defaultValue={0}
          />

          <SliderInput
            label="Max Eval Samples"
            name="max_eval_samples"
            min={0}
            max={100000}
            defaultValue={0}
          />
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
