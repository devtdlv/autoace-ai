// Mirrors inference/pipeline/schema.py's Prediction model exactly — keep in
// sync if the Python schema changes.

export type EmotionalTone = "neutral" | "satisfied" | "frustrated" | "upset" | "distressed";
export type EmotionalIntensity = "low" | "medium" | "high";
export type BackgroundNoiseSeverity = "none" | "low" | "medium" | "high";
export type AudioQuality = "clear" | "slightly_impaired" | "severely_impaired";

export type Prediction = {
  emotional_tone: EmotionalTone;
  emotional_intensity: EmotionalIntensity;
  background_noise_present: boolean;
  background_noise_type: string;
  background_noise_severity: BackgroundNoiseSeverity;
  audio_quality: AudioQuality;
  speaker_overlap_present: boolean;
  long_silence_present: boolean;
  confidence: number;
};

export type CallStatus = "pending" | "processing" | "done" | "failed";
export type BatchStatus = "pending" | "processing" | "done" | "failed";

// Ground truth from a manifest's result_json column, when the batch was
// labeled. Untrusted/evaluator-supplied — not assumed to match our exact
// enum values, so it's typed loosely rather than as `Prediction`.
export type ExpectedResult = Record<string, unknown>;

export type Call = {
  id: string;
  batch_id: string;
  filename: string;
  status: CallStatus;
  result: Prediction | null;
  expected: ExpectedResult | null;
  error: string | null;
};

export type Batch = {
  id: string;
  created_at: string;
  status: BatchStatus;
  manifest_name: string;
  total_calls: number;
  completed_calls: number;
};
