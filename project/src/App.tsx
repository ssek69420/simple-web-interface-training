import SliderInput from "./SliderInput";

export default function App() {
    return (
        <div className="app">

            <aside className="sidebar">

                <h1>Model Trainer</h1>

                <section className="group">
                    <h2>Files</h2>

                    <label>
                        Dataset
                        <input type="file" />
                    </label>

                    <label>
                        Output Folder
                        <input type="text" placeholder="Select folder..." />
                    </label>
                </section>

                <section className="group">
                    <h2>Training</h2>

                    <SliderInput
                        label="Epochs"
                        min={1}
                        max={100}
                        value={4}
                    />

                    <SliderInput
                        label="Steps"
                        min={1}
                        max={5000}
                        value={20}
                    />

                    <SliderInput
                        label="Batch Size"
                        min={1}
                        max={64}
                        value={8}
                    />

                    <SliderInput
                        label="Learning Rate"
                        min={0.000001}
                        max={0.01}
                        step={0.000001}
                        value={0.0001}
                    />

                    <SliderInput
                        label="Save Every"
                        min={50}
                        max={5000}
                        value={500}
                    />

                    <SliderInput
                        label="Warmup"
                        min={0}
                        max={1000}
                        value={100}
                    />
                </section>

                <section className="group">

                    <label>
                        <input type="checkbox" />
                        Mixed Precision (FP16)
                    </label>

                    <label>
                        <input type="checkbox" />
                        Resume Training
                    </label>

                    <label>
                        <input type="checkbox" />
                        Cache Dataset
                    </label>

                    <label>
                        <input type="checkbox" />
                        Shuffle Dataset
                    </label>

                </section>

                <button className="start">
                    Start Training
                </button>

            </aside>

            <main className="console">

                <div className="consoleHeader">
                    Console
                </div>

                <pre>
{`Waiting for training...
`}
                </pre>

            </main>

        </div>
    );
}