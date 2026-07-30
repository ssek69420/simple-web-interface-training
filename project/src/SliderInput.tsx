import { useState } from "react";

type SliderInputProps = {
  label: string;
  name: string;
  min: number;
  max: number;
  defaultValue: number;
  step?: number;
};

export default function SliderInput({
  label,
  name,
  min,
  max,
  defaultValue,
  step = 1,
}: SliderInputProps) {
  const [value, setValue] = useState<string>(String(defaultValue));

  function handleRangeChange(e: React.ChangeEvent<HTMLInputElement>) {
    setValue(e.target.value);
  }

  function handleNumberChange(e: React.ChangeEvent<HTMLInputElement>) {
    setValue(e.target.value);
  }

  return (
    <div className="slider" data-name={name}>
      <div className="sliderTop">
        <label htmlFor={`${name}-number`}>{label}</label>

        <input
          id={`${name}-number`}
          name={name}
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={handleNumberChange}
        />
      </div>

      <input
        id={`${name}-range`}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleRangeChange}
      />
    </div>
  );
}