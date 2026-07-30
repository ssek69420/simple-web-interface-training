export type TrainingConfig = {
  dataset: File | null;
  output_dir: string;
  epochs: number;
  steps: number;
  batch_size: number;
  learning_rate: number;
  save_every: number;
  warmup: number;
  seed: number;
  fp16: boolean;
  resume: boolean;
  cache_dataset: boolean;
  shuffle_dataset: boolean;
};