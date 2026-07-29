import { useState } from "react";

export default function SliderInput({
    label,
    min,
    max,
    value,
    step = 1
}) {

    const [val, setVal] = useState(value);

    return (
        <div className="slider">

            <div className="sliderTop">

                <label>{label}</label>

                <input
                    type="number"
                    value={val}
                    onChange={(e) => setVal(e.target.value)}
                />

            </div>

            <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={val}
                onChange={(e) => setVal(e.target.value)}
            />

        </div>
    );
}