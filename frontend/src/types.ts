export type Status =
  | "READY"
  | "RUNNING"
  | "UNAVAILABLE"
  | "ERROR"
  | "WAITING"
  | "COMPLETED"
  | "WARNING"
  | "FAILED"
  | "SKIPPED";
export interface ComponentStatus {
  status: Status;
  detail: string;
}
export interface SystemStatus {
  components: Record<string, ComponentStatus>;
  totals: Record<string, number>;
  privacy: string;
}
export interface DocumentInfo {
  id: string;
  name: string;
  safe_name: string;
  media_type: string;
  size: number;
  pages: number;
  created_at: string;
  is_demo: boolean;
  preview_url: string;
}
export interface Stage {
  key: string;
  number: number;
  name: string;
  status: Status;
  input: unknown;
  method: string;
  parameters: Record<string, unknown>;
  output: unknown;
  diagnostics: Record<string, unknown>;
  next_step: string | null;
  runtime_ms: number | null;
  warnings: string[];
  error: string | null;
}
export interface OCRToken {
  id: string;
  page: number;
  text: string;
  confidence: number;
  ocr_confidence?: number;
  confidence_source: string;
  confidence_calibration?: {
    applied: boolean;
    reason: string;
    candidate: string;
    similarity: number;
  } | null;
  bbox: number[];
  decision?: "KEEP" | "CORRECT";
  context?: string;
}
export interface GraphData {
  schema_version?: string;
  domain?: string;
  nodes: Array<{
    id: string;
    label: string;
    type: string;
    pages: number[];
    mentions: number;
    confidence: number;
    token_ids?: string[];
    provenance?: Array<Record<string, unknown>>;
    properties?: Record<string, unknown>;
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
    type: string;
    provenance?: string;
    token_ids?: string[];
    pages?: number[];
    properties?: Record<string, unknown>;
  }>;
  persisted_to_neo4j: boolean;
}
export interface RunResult {
  document?: DocumentInfo;
  tokens?: OCRToken[];
  confidence?: {
    threshold: number;
    average: number;
    low_count: number;
    high_count: number;
    distribution: Array<{ range: string; count: number }>;
  };
  low_confidence?: Array<{
    id: string;
    text: string;
    confidence: number;
    ocr_confidence?: number;
    confidence_calibration?: Record<string, unknown>;
    page: number;
    reason: string;
  }>;
  preprocessing?: Array<{
    source: string;
    output: string;
    operations: string[];
    width: number;
    height: number;
  }>;
  layout?: unknown[];
  reading_order?: unknown[];
  candidates?: Array<{
    token_id: string;
    original: string;
    candidates: Array<{ value: string; source: string; similarity: number }>;
  }>;
  semantic_evidence?: Evidence[];
  graph_evidence?: Array<{ token_id: string; results: unknown[] }>;
  combined_evidence?: unknown[];
  graph?: GraphData;
  prompts?: Array<{ token_id: string; rendered: string }>;
  slm_decisions?: Array<Record<string, unknown>>;
  validations?: Array<Record<string, unknown>>;
  original_text?: string;
  corrected_text?: string;
  export_pdf_url?: string;
  export_pdf_path?: string;
  provenance_json_path?: string;
  traces?: Trace[];
  stats?: Record<string, number>;
}
export interface Evidence {
  token_id: string;
  query: string;
  results: Array<{
    id: string;
    page: number;
    text: string;
    similarity: number;
    source: string;
  }>;
}
export interface Trace {
  token_id: string;
  original: string;
  final: string;
  selection: string;
  status: string;
  confidence: number;
  changed: boolean;
  why_selected: string;
  candidates: Array<{ value: string; source: string; similarity: number }>;
  semantic_evidence: Array<{
    text: string;
    similarity: number;
    source: string;
  }>;
  graph_evidence: unknown[];
  prompt: string;
  slm_decision: Record<string, unknown> | null;
  checks: string[];
}
export interface Run {
  id: string;
  document_id: string;
  document_name?: string;
  status: string;
  created_at: string;
  updated_at: string;
  config: Record<string, unknown>;
  result: RunResult;
  stages: Stage[];
  is_demo?: boolean;
  pages?: number;
}

export interface EvaluationDocument {
  document_id: string;
  run_id: string;
  filename: string;
  file_type: string;
  number_of_pages: number;
  ocr_engine: string | null;
  extracted_character_count: number;
  extracted_word_count: number;
  ocr_confidence: number | null;
  cer: number | null;
  wer: number | null;
  noise_ratio: number | null;
  valid_word_ratio: number | null;
  dictionary_match_ratio: number | null;
  language_consistency: number | null;
  structural_consistency: number | null;
  final_quality_score: number;
  quality_class: string;
  ground_truth_available: boolean;
}
export interface EvaluationReport {
  evaluation_id: string;
  status: string;
  started_at: string;
  completed_at: string;
  mode: "WITH_GROUND_TRUTH" | "WITHOUT_GROUND_TRUTH" | "MIXED";
  input_signature: string;
  documents: EvaluationDocument[];
  summary: {
    number_of_documents: number;
    documents_with_ground_truth: number;
    ground_truth_coverage: number;
    quality_classes: Record<string, { count: number; percentage: number }>;
    metrics: Record<string, { count: number; mean: number | null; median: number | null }>;
    aggregate_error_rates: Record<string, number | null>;
  };
  charts: Array<{ name: string; title: string; file: string }>;
  skipped_documents: Array<{ run_id: string; reason: string }>;
  output_directory: string;
}
