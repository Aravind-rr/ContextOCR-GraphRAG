import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BookOpen,
  ChevronRight,
  Database,
  FileCheck2,
  FilePlus2,
  FlaskConical,
  History,
  LayoutDashboard,
  LockKeyhole,
  Menu,
  Network,
  Play,
  RefreshCw,
  Search,
  Server,
  Settings2,
  UploadCloud,
  X,
} from "lucide-react";
import { api, API } from "./api";
import {
  Badge,
  ConfidenceChart,
  DocumentViewer,
  GraphViewer,
  PipelineStepper,
  ResultsCompare,
  StageDetail,
  TokenTable,
  TracePanel,
} from "./components";
import type { DocumentInfo, EvaluationReport, OCRToken, Run, SystemStatus, Trace } from "./types";

type View = "dashboard" | "workspace" | "evaluation" | "history" | "system";
const nav: Array<{ id: View; label: string; icon: typeof Activity }> = [
  { id: "dashboard", label: "Overview", icon: LayoutDashboard },
  { id: "workspace", label: "Pipeline workspace", icon: Activity },
  { id: "evaluation", label: "Evaluation", icon: FlaskConical },
  { id: "history", label: "Run history", icon: History },
  { id: "system", label: "System status", icon: Server },
];

export function App() {
  const [view, setView] = useState<View>("dashboard"),
    [status, setStatus] = useState<SystemStatus>(),
    [history, setHistory] = useState<Run[]>([]),
    [run, setRun] = useState<Run>(),
    [document, setDocument] = useState<DocumentInfo>(),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [mobile, setMobile] = useState(false);
  const refresh = async () => {
    try {
      const [s, h] = await Promise.all([api.status(), api.history()]);
      setStatus(s);
      setHistory(h);
    } catch (e) {
      setError(message(e));
    }
  };
  useEffect(() => {
    void refresh();
  }, []);
  useEffect(() => {
    if (!run || !["QUEUED", "RUNNING"].includes(run.status)) return;
    const source = new EventSource(`${API}/api/runs/${run.id}/events`);
    source.onmessage = async () => {
      try {
        setRun(await api.run(run.id));
      } catch {}
    };
    source.onerror = () => source.close();
    return () => source.close();
  }, [run?.id, run?.status]);
  const loadRun = async (id: string) => {
    try {
      setBusy(true);
      setRun(await api.run(id));
      setView("workspace");
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  };
  const upload = async (file: File) => {
    try {
      setBusy(true);
      setError("");
      setDocument(await api.upload(file));
      setView("workspace");
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  };
  const loadDemo = async () => {
    try {
      setBusy(true);
      setDocument(await api.demo());
      setView("workspace");
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  };
  const start = async (threshold: number, domain: string) => {
    if (!document) return;
    try {
      setBusy(true);
      setRun(await api.start(document.id, threshold, domain));
      setError("");
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobile ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">
            <Network size={21} />
          </div>
          <div>
            <strong>ContextOCR</strong>
            <small>GraphRAG workbench</small>
          </div>
          <button className="mobile-close" onClick={() => setMobile(false)}>
            <X />
          </button>
        </div>
        <nav>
          {nav.map((n) => (
            <button
              key={n.id}
              className={view === n.id ? "active" : ""}
              onClick={() => {
                setView(n.id);
                setMobile(false);
              }}
            >
              <n.icon size={18} />
              {n.label}
            </button>
          ))}
        </nav>
        <div className="privacy">
          <LockKeyhole size={17} />
          <div>
            <strong>Local by design</strong>
            <span>Core document processing stays on this machine.</span>
          </div>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <button className="menu" onClick={() => setMobile(true)}>
            <Menu />
          </button>
          <div>
            <span className="eyebrow">Research prototype · Review 3</span>
            <h1>{nav.find((n) => n.id === view)?.label}</h1>
          </div>
          <div className="top-actions">
            <span className="api-live">
              <i /> API connected
            </span>
            <button
              className="icon-btn"
              onClick={() => void refresh()}
              aria-label="Refresh"
            >
              <RefreshCw size={17} />
            </button>
            <button
              className="button primary"
              onClick={() => {
                setDocument(undefined);
                setRun(undefined);
                setView("workspace");
              }}
            >
              <FilePlus2 size={17} /> Process document
            </button>
          </div>
        </header>
        {error && (
          <div className="global-error">
            <AlertIcon />
            {error}
            <button onClick={() => setError("")}>Dismiss</button>
          </div>
        )}
        <div className="content">
          {view === "dashboard" && (
            <Dashboard
              status={status}
              history={history}
              onNew={() => setView("workspace")}
              onRun={loadRun}
            />
          )}{" "}
          {view === "workspace" && (
            <Workspace
              document={document}
              run={run}
              busy={busy}
              onUpload={upload}
              onDemo={loadDemo}
              onStart={start}
            />
          )}{" "}
          {view === "evaluation" && <Evaluation run={run} />}{" "}
          {view === "history" && <HistoryView runs={history} onRun={loadRun} />}{" "}
          {view === "system" && <SystemView status={status} />}
        </div>
      </main>
    </div>
  );
}
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));
function AlertIcon() {
  return <span>!</span>;
}

function Dashboard({
  status,
  history,
  onNew,
  onRun,
}: {
  status?: SystemStatus;
  history: Run[];
  onNew: () => void;
  onRun: (id: string) => void;
}) {
  const totals = status?.totals || {};
  return (
    <>
      <section className="hero">
        <div>
          <span className="eyebrow">
            Selective · Evidence-grounded · Private
          </span>
          <h2>
            Context-Aware Zero-Shot Prompt Engineering for Small Language Model
            Post-Correction Using GraphRAG
          </h2>
          <p>
            Confidence-aware OCR post-correction using hybrid semantic and graph
            retrieval with a local Small Language Model.
          </p>
          <button className="button primary large" onClick={onNew}>
            Process new document <ChevronRight size={18} />
          </button>
        </div>
        <div className="research-note">
          <BookOpen size={22} />
          <strong>Research principle</strong>
          <blockquote>
            “Correct only when necessary, and only when supported by evidence.”
          </blockquote>
          <div className="mini-flow">
            <span>OCR</span>
            <i>→</i>
            <span>Confidence</span>
            <i>→</i>
            <span>Evidence</span>
            <i>→</i>
            <span>Local SLM</span>
            <i>→</i>
            <span>Validate</span>
          </div>
        </div>
      </section>
      <section>
        <div className="section-heading">
          <div>
            <span className="eyebrow">Live environment</span>
            <h2>System readiness</h2>
          </div>
          <button className="text-button" onClick={onNew}>
            Open workspace <ChevronRight size={15} />
          </button>
        </div>
        <div className="status-grid">
          {status ? (
            Object.entries(status.components).map(([key, item]) => (
              <article className="status-card" key={key}>
                <div>
                  <ComponentIcon name={key} />
                  <span>{key.replace("_", " ")}</span>
                </div>
                <Badge status={item.status} />
                <p>{item.detail}</p>
              </article>
            ))
          ) : (
            <Skeleton count={6} />
          )}
        </div>
      </section>
      <section>
        <div className="section-heading">
          <div>
            <span className="eyebrow">Measured activity</span>
            <h2>Processing summary</h2>
          </div>
        </div>
        <div className="metrics-grid">
          {[
            ["Documents processed", "documents_processed"],
            ["OCR tokens", "ocr_tokens"],
            ["Low-confidence", "low_confidence_tokens"],
            ["Corrections applied", "corrections_applied"],
            ["Corrections rejected", "corrections_rejected"],
            ["SLM calls", "slm_calls"],
          ].map(([label, key]) => (
            <article className="metric" key={key}>
              <span>{label}</span>
              <strong>{totals[key] ?? "—"}</strong>
            </article>
          ))}
        </div>
      </section>
      <section>
        <div className="section-heading">
          <div>
            <span className="eyebrow">Recent work</span>
            <h2>Run history</h2>
          </div>
        </div>
        <RunList runs={history.slice(0, 5)} onRun={onRun} />
      </section>
    </>
  );
}
function ComponentIcon({ name }: { name: string }) {
  return name === "neo4j" ? (
    <Network size={18} />
  ) : name === "faiss" || name === "embedding_model" ? (
    <Database size={18} />
  ) : name === "ocr_engine" ? (
    <Search size={18} />
  ) : (
    <Server size={18} />
  );
}
function Skeleton({ count }: { count: number }) {
  return (
    <>
      {Array.from({ length: count }, (_, i) => (
        <div className="skeleton" key={i} />
      ))}
    </>
  );
}

function UploadZone({
  busy,
  onUpload,
  onDemo,
}: {
  busy: boolean;
  onUpload: (f: File) => void;
  onDemo: () => void;
}) {
  const [drag, setDrag] = useState(false);
  const pick = (files: FileList | null) => files?.[0] && onUpload(files[0]);
  return (
    <div
      className={`upload-zone ${drag ? "drag" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        pick(e.dataTransfer.files);
      }}
    >
      <UploadCloud size={35} />
      <h2>{busy ? "Reading document…" : "Drop a document here"}</h2>
      <p>PDF, PNG, JPG or TIFF · Maximum 25 MB</p>
      <label className="button primary">
        <input
          type="file"
          accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff"
          onChange={(e) => pick(e.target.files)}
        />
        Choose document
      </label>
      <button className="text-button" onClick={onDemo}>
        or use clearly labeled demo data
      </button>
    </div>
  );
}
function Workspace({
  document,
  run,
  busy,
  onUpload,
  onDemo,
  onStart,
}: {
  document?: DocumentInfo;
  run?: Run;
  busy: boolean;
  onUpload: (f: File) => void;
  onDemo: () => void;
  onStart: (t: number, d: string) => void;
}) {
  const [threshold, setThreshold] = useState(0.9),
    [domain, setDomain] = useState("general"),
    [selectedStage, setSelectedStage] = useState(1),
    [tab, setTab] = useState<
      "stage" | "ocr" | "trace" | "evidence" | "graph" | "output"
    >("stage"),
    [token, setToken] = useState<OCRToken>();
  useEffect(() => {
    const active = run?.stages.find((s) => s.status === "RUNNING");
    if (active) setSelectedStage(active.number);
  }, [run?.updated_at]);
  useEffect(() => {
    if (document?.is_demo && !run) {
      setThreshold(0.9);
      setDomain("government");
    }
  }, [document?.id, document?.is_demo, run]);
  const traces = run?.result.traces || [];
  const selectedTrace =
    traces.find((t) => t.token_id === token?.id) || traces[0];
  if (!document && !run)
    return <UploadZone busy={busy} onUpload={onUpload} onDemo={onDemo} />;
  if (document && !run)
    return (
      <div className="document-ready">
        <div className="ready-preview">
          {document.media_type.startsWith("image/") ? (
            <img src={`${API}${document.preview_url}`} alt="Document preview" />
          ) : (
            <FileCheck2 size={60} />
          )}
        </div>
        <div>
          <span className="eyebrow">
            Document ready{" "}
            {document.is_demo && <em className="demo">DEMO DATA</em>}
          </span>
          <h2>{document.name}</h2>
          <div className="file-facts">
            <span>{formatBytes(document.size)}</span>
            <span>
              {document.pages} page{document.pages !== 1 ? "s" : ""}
            </span>
            <span>{document.media_type}</span>
          </div>
          <div className="config-row">
            <label>
              Confidence threshold <strong>{threshold.toFixed(3)}</strong>
              <input
                type="range"
                min="0"
                max="1"
                step=".005"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
            </label>
            <label>
              Domain
              <select
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
              >
                {[
                  "general",
                  "medical",
                  "legal",
                  "government",
                  "financial",
                  "institutional",
                ].map((x) => (
                  <option key={x}>{x}</option>
                ))}
              </select>
            </label>
          </div>
          <button
            className="button primary large"
            disabled={busy}
            onClick={() => onStart(threshold, domain)}
          >
            <Play size={18} /> Run pipeline
          </button>
          <p className="fine-print">
            Recommended: 0.900. PaddleOCR's raw line score is retained while lexical calibration isolates suspicious words for correction.
          </p>
        </div>
      </div>
    );
  if (!run) return null;
  return (
    <div className="workspace">
      <div className="workspace-head">
        <div>
          <span className="eyebrow">
            Run {run.id.slice(0, 8)}{" "}
            {run.result.document?.is_demo && (
              <em className="demo">DEMO DATA</em>
            )}
          </span>
          <h2>{run.result.document?.name || "Pipeline execution"}</h2>
        </div>
        <Badge status={run.status} />
      </div>
      <div className="pipeline-layout">
        <PipelineStepper
          stages={run.stages}
          selected={selectedStage}
          onSelect={(n) => {
            setSelectedStage(n);
            setTab("stage");
          }}
        />
        <div className="inspection">
          <div className="tabs">
            {(
              ["stage", "ocr", "trace", "evidence", "graph", "output"] as const
            ).map((x) => (
              <button
                className={tab === x ? "active" : ""}
                onClick={() => setTab(x)}
                key={x}
              >
                {x}
              </button>
            ))}
          </div>
          {tab === "stage" && (
            <StageDetail stage={run.stages[selectedStage - 1]} run={run} />
          )}{" "}
          {tab === "ocr" && (
            <div className="ocr-grid">
              <DocumentViewer run={run} token={token} />
              <div>
                <div className="section-title">
                  <div>
                    <span className="eyebrow">OCR inspection</span>
                    <h2>{run.result.tokens?.length || 0} tokens</h2>
                  </div>
                </div>
                {run.result.confidence && (
                  <ConfidenceChart data={run.result.confidence.distribution} />
                )}
                <TokenTable
                  tokens={run.result.tokens || []}
                  selected={token?.id}
                  onSelect={(t) => setToken(t)}
                />
              </div>
            </div>
          )}{" "}
          {tab === "trace" && (
            <div className="trace-grid">
              <div className="trace-select">
                <h3>Low-confidence tokens</h3>
                {traces.map((t) => (
                  <button
                    className={
                      selectedTrace?.token_id === t.token_id ? "active" : ""
                    }
                    onClick={() =>
                      setToken(
                        run.result.tokens?.find((x) => x.id === t.token_id),
                      )
                    }
                    key={t.token_id}
                  >
                    <span>{t.original}</span>
                    <small>{(t.confidence * 100).toFixed(1)}%</small>
                  </button>
                ))}
                {!traces.length && (
                  <p className="muted">No correction traces yet.</p>
                )}
              </div>
              <TracePanel trace={selectedTrace} />
            </div>
          )}{" "}
          {tab === "evidence" && (
            <EvidenceView run={run} trace={selectedTrace} />
          )}{" "}
          {tab === "graph" && <GraphViewer graph={run.result.graph} />}{" "}
          {tab === "output" && <ResultsCompare run={run} />}
        </div>
      </div>
    </div>
  );
}
const formatBytes = (n: number) =>
  n < 1024
    ? `${n} B`
    : n < 1048576
      ? `${(n / 1024).toFixed(1)} KB`
      : `${(n / 1048576).toFixed(1)} MB`;
function EvidenceView({ run, trace }: { run: Run; trace?: Trace }) {
  const sem =
    trace?.semantic_evidence ||
    run.result.semantic_evidence?.flatMap((x) => x.results).slice(0, 5) ||
    [];
  return (
    <div>
      <div className="section-title">
        <div>
          <span className="eyebrow">Hybrid GraphRAG</span>
          <h2>Evidence inspection</h2>
        </div>
      </div>
      <div className="evidence-grid">
        <article>
          <h3>Semantic evidence</h3>
          {sem.length ? (
            sem.map((x, i) => (
              <div className="evidence" key={i}>
                <strong>{x.source}</strong>
                <span>{(x.similarity * 100).toFixed(1)}% similarity</span>
                <p>{x.text}</p>
              </div>
            ))
          ) : (
            <p className="muted">No semantic evidence retrieved.</p>
          )}
        </article>
        <article>
          <h3>Graph evidence</h3>
          {trace?.graph_evidence?.length ? (
            <pre>{JSON.stringify(trace.graph_evidence, null, 2)}</pre>
          ) : (
            <p className="muted">No graph entity supports this candidate.</p>
          )}
        </article>
        <article>
          <h3>Combined evidence</h3>
          <p>
            {run.result.combined_evidence?.length
              ? `Evidence sets retained for ${run.result.combined_evidence.length} selected token(s).`
              : "No combined evidence available."}
          </p>
        </article>
      </div>
    </div>
  );
}

function Evaluation({ run }: { run?: Run }) {
  const [truth, setTruth] = useState(""),
    [report, setReport] = useState<EvaluationReport>(),
    [experiments, setExperiments] = useState<
      Array<{
        name: string;
        status: string;
        metrics: Record<string, number> | null;
        reason?: string;
      }>
    >([]),
    [evaluationBusy, setEvaluationBusy] = useState(false),
    [error, setError] = useState("");
  useEffect(() => {
    setReport(undefined);
    setExperiments([]);
    setTruth("");
    setError("");
  }, [run?.id]);
  if (!run?.result.corrected_text)
    return (
      <EmptyState
        icon={FlaskConical}
        title="Ground truth not configured — evaluation unavailable."
        body="Ground truth is optional. CER and WER remain unavailable when no verified reference is supplied."
      />
    );
  const execute = async (dataset = false) => {
    try {
      setEvaluationBusy(true);
      const evaluation = dataset ? await api.evaluateDataset() : await api.evaluate(run.id, truth);
      setReport(evaluation);
      if (!dataset && truth.trim()) {
        const controlled = await api.experiments(run.id, truth);
        setExperiments(controlled.experiments);
      } else setExperiments([]);
      setError("");
    } catch (e) {
      setError(message(e));
    } finally {
      setEvaluationBusy(false);
    }
  };
  return (
    <div className="evaluation">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Dynamic quality analysis</span>
          <h2>Current document and dataset evaluation</h2>
          <p className="muted">Each evaluation creates a new ID and recalculates metrics and charts from current pipeline outputs.</p>
        </div>
      </div>
      <label className="ground-truth">
        Verified ground-truth transcription (optional)
        <textarea
          value={truth}
          onChange={(e) => setTruth(e.target.value)}
          placeholder="Paste manually verified ground truth here…"
        />
      </label>
      <div className="evaluation-actions">
        <button className="button primary" disabled={evaluationBusy} onClick={() => void execute(false)}>{evaluationBusy ? "Evaluating…" : "Evaluate current document"}</button>
        <button className="button" disabled={evaluationBusy} onClick={() => void execute(true)}>Evaluate latest completed dataset</button>
      </div>
      {error && <p className="error-text">{error}</p>}
      {report && <>
        <div className="evaluation-run-banner">
          <div><span>Current evaluation ID</span><strong>{report.evaluation_id}</strong></div>
          <div><span>Timestamp</span><strong>{new Date(report.completed_at).toLocaleString()}</strong></div>
          <div><span>Mode</span><strong>{report.mode.replaceAll("_", " ")}</strong></div>
          <div><span>Input signature</span><strong>{report.input_signature.slice(0, 16)}…</strong></div>
        </div>
        <div className="metrics-grid evaluation-metrics">
          <article className="metric"><span>Documents</span><strong>{report.summary.number_of_documents}</strong></article>
          <article className="metric"><span>With ground truth</span><strong>{report.summary.documents_with_ground_truth}</strong></article>
          <article className="metric"><span>Average quality</span><strong>{formatMetric(report.summary.metrics.final_quality_score?.mean, "score")}</strong></article>
          <article className="metric"><span>Median quality</span><strong>{formatMetric(report.summary.metrics.final_quality_score?.median, "score")}</strong></article>
          <article className="metric"><span>Average confidence</span><strong>{formatMetric(report.summary.metrics.ocr_confidence?.mean)}</strong></article>
          <article className="metric"><span>Average noise</span><strong>{formatMetric(report.summary.metrics.noise_ratio?.mean)}</strong></article>
          <article className="metric"><span>Average CER</span><strong>{formatMetric(report.summary.metrics.cer?.mean)}</strong></article>
          <article className="metric"><span>Average WER</span><strong>{formatMetric(report.summary.metrics.wer?.mean)}</strong></article>
        </div>
        <div className="table-wrap evaluation-table">
          <table>
            <thead><tr><th>Document</th><th>Score</th><th>Class</th><th>OCR confidence</th><th>Noise</th><th>Valid words</th><th>CER</th><th>WER</th></tr></thead>
            <tbody>{report.documents.map((document) => <tr key={document.run_id}>
              <td>{document.filename}</td><td>{document.final_quality_score.toFixed(2)}</td>
              <td><span className={`quality-pill ${document.quality_class.split(" ")[0].toLowerCase()}`}>{document.quality_class}</span></td>
              <td>{formatMetric(document.ocr_confidence)}</td><td>{formatMetric(document.noise_ratio)}</td>
              <td>{formatMetric(document.valid_word_ratio)}</td><td>{formatMetric(document.cer)}</td><td>{formatMetric(document.wer)}</td>
            </tr>)}</tbody>
          </table>
        </div>
        <div className="evaluation-charts">
          {report.charts.map((chart) => <figure key={chart.name}>
            <img src={api.evaluationChart(report.evaluation_id, chart.name)} alt={chart.title}/><figcaption>{chart.title}</figcaption>
          </figure>)}
        </div>
        <p className="evaluation-location">Machine-readable JSON, CSV, graph source data, charts and logs: <code>{report.output_directory}</code></p>
        {report.skipped_documents.length > 0 && <div className="warning-box"><p>{report.skipped_documents.length} run(s) were skipped. See the evaluation log for exact reasons.</p></div>}
      </>}
      {experiments.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Status</th>
                <th>CER</th>
                <th>WER</th>
                <th>Over-correction</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((x) => (
                <tr key={x.name}>
                  <td>{x.name}</td>
                  <td>
                    <Badge status={x.status} />
                  </td>
                  <td>{pct(x.metrics?.cer)}</td>
                  <td>{pct(x.metrics?.wer)}</td>
                  <td>{pct(x.metrics?.over_correction_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
const pct = (n?: number) => (n == null ? "—" : `${(n * 100).toFixed(2)}%`);
const formatMetric = (value?: number | null, kind: "ratio" | "score" = "ratio") => value == null ? "Unavailable" : kind === "score" ? value.toFixed(2) : `${(value * 100).toFixed(2)}%`;
function HistoryView({
  runs,
  onRun,
}: {
  runs: Run[];
  onRun: (id: string) => void;
}) {
  return (
    <>
      <div className="section-heading">
        <div>
          <span className="eyebrow">Persistent local metadata</span>
          <h2>Processing runs</h2>
        </div>
      </div>
      <RunList runs={runs} onRun={onRun} />
    </>
  );
}
function RunList({
  runs,
  onRun,
}: {
  runs: Run[];
  onRun: (id: string) => void;
}) {
  if (!runs.length)
    return (
      <EmptyState
        icon={History}
        title="No processing runs yet"
        body="Upload a real document or use the labeled demo fixture to begin."
      />
    );
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Document</th>
            <th>Started</th>
            <th>Status</th>
            <th>Pages</th>
            <th>Tokens</th>
            <th>Low confidence</th>
            <th>Runtime</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} onClick={() => onRun(r.id)} tabIndex={0}>
              <td>
                <strong>
                  {r.document_name || r.result.document?.name || r.document_id}
                </strong>
                {r.is_demo && <em className="demo">DEMO</em>}
              </td>
              <td>{new Date(r.created_at).toLocaleString()}</td>
              <td>
                <Badge status={r.status} />
              </td>
              <td>{r.pages || r.result.document?.pages || "—"}</td>
              <td>{r.result.stats?.ocr_tokens ?? "—"}</td>
              <td>{r.result.stats?.low_confidence_tokens ?? "—"}</td>
              <td>{sumRuntime(r.stages)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
const sumRuntime = (s: Run["stages"]) => {
  const n = s.reduce((a, x) => a + (x.runtime_ms || 0), 0);
  return n ? `${(n / 1000).toFixed(2)} s` : "—";
};
function SystemView({ status }: { status?: SystemStatus }) {
  if (!status) return <Skeleton count={6} />;
  return (
    <>
      <div className="section-heading">
        <div>
          <span className="eyebrow">Verified at request time</span>
          <h2>Component availability</h2>
        </div>
      </div>
      <div className="system-list">
        {Object.entries(status.components).map(([name, item]) => (
          <article key={name}>
            <ComponentIcon name={name} />
            <div>
              <h3>{name.replaceAll("_", " ")}</h3>
              <p>{item.detail}</p>
            </div>
            <Badge status={item.status} />
          </article>
        ))}
      </div>
      <div className="privacy-banner">
        <LockKeyhole />
        <div>
          <strong>{status.privacy}</strong>
          <p>No cloud language-model API is used by the correction pipeline.</p>
        </div>
      </div>
    </>
  );
}
function EmptyState({
  icon: Icon,
  title,
  body,
}: {
  icon: typeof History;
  title: string;
  body: string;
}) {
  return (
    <div className="empty-state">
      <Icon size={36} />
      <h2>{title}</h2>
      <p>{body}</p>
    </div>
  );
}
