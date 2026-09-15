import { useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  Clock,
  Download,
  FileSearch,
  LoaderCircle,
  Minus,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { API } from "./api";
import type { GraphData, OCRToken, Run, Stage, Status, Trace } from "./types";

export function Badge({ status }: { status: Status | string }) {
  return <span className={`badge ${status.toLowerCase()}`}>{status}</span>;
}
export function PipelineStepper({
  stages,
  selected,
  onSelect,
}: {
  stages: Stage[];
  selected: number;
  onSelect: (n: number) => void;
}) {
  return (
    <aside className="pipeline">
      <div className="eyebrow">Execution pipeline</div>
      {stages.map((s) => (
        <button
          key={s.key}
          className={`stage-row ${selected === s.number ? "active" : ""}`}
          onClick={() => onSelect(s.number)}
        >
          <span className="stage-index">
            {String(s.number).padStart(2, "0")}
          </span>
          <span className="stage-name">
            {s.name}
            <small>
              {s.runtime_ms == null ? "—" : `${s.runtime_ms.toFixed(0)} ms`}
            </small>
          </span>
          <StageIcon status={s.status} />
        </button>
      ))}
    </aside>
  );
}
function StageIcon({ status }: { status: Status }) {
  if (status === "RUNNING") return <LoaderCircle className="spin" size={17} />;
  if (status === "COMPLETED") return <Check size={17} />;
  if (status === "WARNING" || status === "FAILED")
    return <AlertTriangle size={17} />;
  if (status === "SKIPPED") return <Minus size={17} />;
  return <Clock size={16} />;
}
export function StageDetail({ stage, run }: { stage?: Stage; run: Run }) {
  if (!stage) return <div className="empty">Select a pipeline stage.</div>;
  return (
    <section className="stage-detail">
      <div className="section-title">
        <div>
          <span className="eyebrow">
            Stage {String(stage.number).padStart(2, "0")}
          </span>
          <h2>{stage.name}</h2>
        </div>
        <Badge status={stage.status} />
      </div>
      <StageOutput stage={stage} run={run} />
      <Detail label="Input" value={stage.input} />
      <Detail label="Method" value={stage.method || "Waiting for execution"} />
      <Detail label="Parameters" value={stage.parameters} />
      <Detail label="Complete output / provenance" value={stage.output} />
      <Detail
        label="Diagnostics"
        value={{ ...stage.diagnostics, runtime_ms: stage.runtime_ms }}
      />
      {stage.warnings?.length > 0 && (
        <div className="warning-box">
          {stage.warnings.map((w) => (
            <p key={w}>{w}</p>
          ))}
        </div>
      )}
      {stage.error && <div className="error-box">{stage.error}</div>}
      <Detail
        label="Next step"
        value={stage.next_step || "Pipeline complete"}
      />
    </section>
  );
}

type LayoutRegion = {
  id: string;
  class: string;
  raw_class?: string;
  bbox: number[];
  confidence: number | null;
  source: string;
  order?: number;
};
type LayoutPage = { page: number; regions: LayoutRegion[]; method?: string };
function StageOutput({ stage, run }: { stage: Stage; run: Run }) {
  const result = run.result;
  if (stage.status === "WAITING")
    return (
      <div className="stage-visual-empty">
        <Clock size={20} />
        <span>Waiting for the previous stage to complete.</span>
      </div>
    );
  if (stage.status === "RUNNING")
    return (
      <div className="stage-visual-empty">
        <LoaderCircle className="spin" size={20} />
        <span>Processing real stage output…</span>
      </div>
    );
  switch (stage.key) {
    case "document_input":
      return (
        <div className="stage-summary">
          <FileSearch />
          <div>
            <strong>{result.document?.name}</strong>
            <span>
              {result.document?.pages} page(s) · {result.document?.media_type}
            </span>
          </div>
        </div>
      );
    case "preprocessing":
      return <BeforeAfterStage run={run} />;
    case "layout":
      return (
        <OverlayStage
          run={run}
          pages={(result.layout || []) as LayoutPage[]}
          mode="layout"
        />
      );
    case "reading_order":
      return (
        <OverlayStage
          run={run}
          pages={(result.reading_order || []) as LayoutPage[]}
          mode="order"
        />
      );
    case "ocr":
      return (
        <div className="stage-visual">
          <h3>Extracted OCR tokens</h3>
          <TokenTable
            tokens={(result.tokens || []).slice(0, 30)}
            onSelect={() => {}}
          />
        </div>
      );
    case "confidence":
      return result.confidence ? (
        <div className="stage-visual">
          <div className="inline-stats">
            <span>
              Average{" "}
              <strong>{(result.confidence.average * 100).toFixed(1)}%</strong>
            </span>
            <span>
              Below τ <strong>{result.confidence.low_count}</strong>
            </span>
            <span>
              Kept <strong>{result.confidence.high_count}</strong>
            </span>
          </div>
          <ConfidenceChart data={result.confidence.distribution} />
        </div>
      ) : null;
    case "low_confidence":
      return <LowConfidenceStage run={run} />;
    case "candidates":
      return <CandidateStage run={run} />;
    case "semantic":
      return <SemanticStage run={run} />;
    case "graph":
      return <GraphViewer graph={result.graph} />;
    case "graphrag":
      return <CombinedStage run={run} />;
    case "prompt":
      return <PromptStage run={run} />;
    case "slm":
      return <DecisionStage run={run} />;
    case "validation":
      return <ValidationStage run={run} />;
    case "output":
      return <ResultsCompare run={run} />;
    default:
      return null;
  }
}
function BeforeAfterStage({ run }: { run: Run }) {
  const page = run.result.preprocessing?.[0];
  if (!page)
    return <div className="stage-visual-empty">No preview produced.</div>;
  return (
    <div className="before-after-stage">
      <figure>
        <figcaption>Original rendered page</figcaption>
        <img
          src={`${API}/files/processed/${run.id}/original/page-1.png`}
          alt="Original rendered page"
        />
      </figure>
      <span>→</span>
      <figure>
        <figcaption>Processed page</figcaption>
        <img
          src={`${API}/files/processed/${run.id}/preprocessed/page-1.png`}
          alt="Preprocessed page"
        />
      </figure>
      <div className="operation-list">
        {page.operations.map((op) => (
          <span key={op}>
            <Check size={13} />
            {op}
          </span>
        ))}
      </div>
    </div>
  );
}
function OverlayStage({
  run,
  pages,
  mode,
}: {
  run: Run;
  pages: LayoutPage[];
  mode: "layout" | "order";
}) {
  const page = pages[0],
    dimensions = run.result.preprocessing?.[0];
  if (!page || !dimensions)
    return <div className="stage-visual-empty">No regions produced.</div>;
  return (
    <div className="overlay-stage">
      <div className="overlay-canvas">
        <img
          src={`${API}/files/processed/${run.id}/preprocessed/page-1.png`}
          alt={`${mode} overlay`}
        />
        {page.regions.map((region, i) => (
          <span
            key={region.id}
            className={`region-box region-${i % 5}`}
            style={{
              left: `${(region.bbox[0] / dimensions.width) * 100}%`,
              top: `${(region.bbox[1] / dimensions.height) * 100}%`,
              width: `${((region.bbox[2] - region.bbox[0]) / dimensions.width) * 100}%`,
              height: `${((region.bbox[3] - region.bbox[1]) / dimensions.height) * 100}%`,
            }}
          >
            <b>{mode === "order" ? region.order : region.class}</b>
          </span>
        ))}
      </div>
      <div className="region-list">
        {page.regions.map((region) => (
          <div key={region.id}>
            <strong>
              {mode === "order" ? `#${region.order}` : region.class}
            </strong>
            <span>
              {region.confidence == null
                ? "geometric"
                : `${(region.confidence * 100).toFixed(1)}%`}{" "}
              · {region.source}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
function LowConfidenceStage({ run }: { run: Run }) {
  const selectedIds = new Set(
    (run.result.low_confidence || []).map((item) => item.id),
  );
  const selected = (run.result.tokens || []).filter((token) =>
    selectedIds.has(token.id),
  );
  return (
    <div className="stage-visual">
      <div className="inline-stats">
        <span>
          Threshold{" "}
          <strong>{run.result.confidence?.threshold.toFixed(3)}</strong>
        </span>
        <span>
          Selected <strong>{selected.length}</strong>
        </span>
        <span>
          Whole-document calls <strong>0</strong>
        </span>
      </div>
      {selected.length ? (
        <>
          <TokenTable tokens={selected} onSelect={() => {}} />
          <div className="calibration-list">
            {(run.result.low_confidence || []).map((item) => (
              <p key={item.id}><strong>{item.text}</strong> {item.reason}</p>
            ))}
          </div>
        </>
      ) : (
        <p className="muted">
          No token fell below the configured threshold; later correction stages
          correctly receive an empty input.
        </p>
      )}
    </div>
  );
}
function CandidateStage({ run }: { run: Run }) {
  const values = run.result.candidates || [];
  return (
    <div className="candidate-stage">
      {values.length ? (
        values.map((item) => (
          <article key={item.token_id}>
            <h3>{item.original}</h3>
            {item.candidates.map((c) => (
              <div key={c.value}>
                <strong>{c.value}</strong>
                <span>
                  {(c.similarity * 100).toFixed(1)}% · {c.source}
                </span>
              </div>
            ))}
          </article>
        ))
      ) : (
        <p className="muted">No candidate sets were necessary.</p>
      )}
    </div>
  );
}
function SemanticStage({ run }: { run: Run }) {
  const values = run.result.semantic_evidence || [];
  return (
    <div className="stage-visual">
      {values.length ? (
        values.map((set) => (
          <article className="retrieval-set" key={set.token_id}>
            <h3>Query</h3>
            <p>{set.query}</p>
            {set.results.map((hit) => (
              <div className="evidence" key={hit.id}>
                <strong>{hit.source}</strong>
                <span>{(hit.similarity * 100).toFixed(1)}%</span>
                <p>{hit.text}</p>
              </div>
            ))}
          </article>
        ))
      ) : (
        <p className="muted">No low-confidence query required retrieval.</p>
      )}
    </div>
  );
}
function CombinedStage({ run }: { run: Run }) {
  const values = (run.result.combined_evidence || []) as Array<{
    token_id: string;
    semantic: unknown[];
    graph: unknown[];
  }>;
  return (
    <div className="candidate-stage">
      {values.length ? (
        values.map((item) => (
          <article key={item.token_id}>
            <h3>{item.token_id}</h3>
            <div>
              <strong>Semantic evidence</strong>
              <span>{item.semantic.length} result(s)</span>
            </div>
            <div>
              <strong>Graph evidence</strong>
              <span>{item.graph.length} result(s)</span>
            </div>
            <pre>{JSON.stringify(item, null, 2)}</pre>
          </article>
        ))
      ) : (
        <p className="muted">No hybrid evidence set was required.</p>
      )}
    </div>
  );
}
function PromptStage({ run }: { run: Run }) {
  const prompts = run.result.prompts || [];
  return (
    <div className="stage-visual">
      {prompts.length ? (
        prompts.map((prompt) => (
          <article key={prompt.token_id}>
            <h3>Prompt for {prompt.token_id}</h3>
            <pre>{prompt.rendered}</pre>
          </article>
        ))
      ) : (
        <p className="muted">
          No prompt was constructed because no token passed the confidence gate.
        </p>
      )}
    </div>
  );
}
function DecisionStage({ run }: { run: Run }) {
  const decisions = run.result.slm_decisions || [];
  return (
    <div className="decision-grid">
      {decisions.length ? (
        decisions.map((decision, i) => (
          <article key={String(decision.token_id || i)}>
            <Badge status={String(decision.status)} />
            <h3>{String(decision.selection || "No selection")}</h3>
            <p>{String(decision.reason || "No rationale returned")}</p>
            <span>
              {decision.latency_ms
                ? `${Number(decision.latency_ms).toFixed(0)} ms`
                : "Not invoked"}
            </span>
          </article>
        ))
      ) : (
        <p className="muted">
          Qwen was not invoked because there were no low-confidence tokens.
        </p>
      )}
    </div>
  );
}
function ValidationStage({ run }: { run: Run }) {
  const values = run.result.validations || [];
  return (
    <div className="decision-grid">
      {values.length ? (
        values.map((value, i) => (
          <article key={String(value.token_id || i)}>
            <Badge status={String(value.status)} />
            <h3>
              {String(value.original)} → {String(value.final)}
            </h3>
            <pre>{JSON.stringify(value.checks, null, 2)}</pre>
          </article>
        ))
      ) : (
        <p className="muted">No correction decisions required validation.</p>
      )}
    </div>
  );
}
function Detail({ label, value }: { label: string; value: unknown }) {
  const empty =
    value == null ||
    (typeof value === "object" && Object.keys(value as object).length === 0);
  return (
    <div className="detail-block">
      <h3>{label}</h3>
      {empty ? (
        <p className="muted">Not available yet.</p>
      ) : typeof value === "string" ? (
        <p>{value}</p>
      ) : (
        <pre>{JSON.stringify(value, null, 2)}</pre>
      )}
    </div>
  );
}
export function ConfidenceChart({
  data,
}: {
  data: Array<{ range: string; count: number }>;
}) {
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="range" fontSize={11} />
          <YAxis allowDecimals={false} fontSize={11} />
          <Tooltip />
          <Bar
            dataKey="count"
            name="OCR tokens"
            fill="#246b61"
            radius={[3, 3, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
      <p className="chart-note">Token count by effective confidence; the original PaddleOCR line score remains visible in the token table.</p>
    </div>
  );
}
export function TokenTable({
  tokens,
  selected,
  onSelect,
}: {
  tokens: OCRToken[];
  selected?: string;
  onSelect: (t: OCRToken) => void;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Token</th>
            <th>Page</th>
            <th>Confidence</th>
            <th>Raw OCR</th>
            <th>Decision</th>
          </tr>
        </thead>
        <tbody>
          {tokens.map((t) => (
            <tr
              key={t.id}
              className={selected === t.id ? "selected" : ""}
              onClick={() => onSelect(t)}
              tabIndex={0}
            >
              <td>{t.text}</td>
              <td>{t.page}</td>
              <td>{(t.confidence * 100).toFixed(1)}%</td>
              <td>{((t.ocr_confidence ?? t.confidence) * 100).toFixed(1)}%</td>
              <td>
                <span
                  className={`decision ${(t.decision || "KEEP").toLowerCase()}`}
                >
                  {t.decision || "KEEP"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function DocumentViewer({ run, token }: { run: Run; token?: OCRToken }) {
  const [zoom, setZoom] = useState(1);
  const doc = run.result.document;
  const page = token?.page || 1;
  const details = run.result.preprocessing?.[page - 1];
  const source = details
    ? `${API}/files/processed/${run.id}/preprocessed/page-${page}.png`
    : doc && doc.media_type.startsWith("image/")
      ? `${API}${doc.preview_url}`
      : "";
  const box =
    token && details
      ? {
          left: `${(token.bbox[0] / details.width) * 100}%`,
          top: `${(token.bbox[1] / details.height) * 100}%`,
          width: `${((token.bbox[2] - token.bbox[0]) / details.width) * 100}%`,
          height: `${((token.bbox[3] - token.bbox[1]) / details.height) * 100}%`,
        }
      : undefined;
  return (
    <div className="document-viewer">
      <div className="viewer-tools">
        <span>
          Page {page} of {doc?.pages || 1}
        </span>
        <div>
          <button
            aria-label="Zoom out"
            onClick={() => setZoom(Math.max(0.5, zoom - 0.1))}
          >
            <ZoomOut size={16} />
          </button>
          <span>{Math.round(zoom * 100)}%</span>
          <button
            aria-label="Zoom in"
            onClick={() => setZoom(Math.min(2, zoom + 0.1))}
          >
            <ZoomIn size={16} />
          </button>
        </div>
      </div>
      <div className="page-canvas">
        <div className="page-image" style={{ transform: `scale(${zoom})` }}>
          {source ? (
            <img src={source} alt="Processed document page" />
          ) : (
            <FileSearch size={42} />
          )}{" "}
          {box && <span className="bbox-highlight" style={box} />}
        </div>
        {token && (
          <div className="selection-note">
            Selected: “{token.text}” · bbox [
            {token.bbox.map((n) => Math.round(n)).join(", ")}]
          </div>
        )}
      </div>
    </div>
  );
}
export function TracePanel({ trace }: { trace?: Trace }) {
  if (!trace)
    return (
      <div className="empty">
        Select a low-confidence token to inspect its complete correction trace.
      </div>
    );
  const steps = [
    ["Original token", trace.original],
    ["OCR confidence", `${(trace.confidence * 100).toFixed(1)}%`],
    ["Why selected", trace.why_selected],
    ["Candidates", trace.candidates],
    ["Semantic evidence", trace.semantic_evidence],
    ["Graph evidence", trace.graph_evidence],
    ["Dynamic prompt", trace.prompt],
    ["SLM decision", trace.slm_decision],
    ["Validation", { status: trace.status, checks: trace.checks }],
    ["Final result", trace.final],
  ];
  return (
    <div className="trace">
      <div className="section-title">
        <div>
          <span className="eyebrow">Correction trace</span>
          <h2>
            {trace.original} → {trace.final}
          </h2>
        </div>
        <Badge status={trace.status} />
      </div>
      {steps.map(([name, value], i) => (
        <div className="trace-step" key={String(name)}>
          <span className="trace-count">{i + 1}</span>
          <div>
            <h3>{String(name)}</h3>
            {typeof value === "string" ? (
              <p>{value || "No evidence returned"}</p>
            ) : (
              <pre>{JSON.stringify(value, null, 2)}</pre>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
export function GraphViewer({ graph }: { graph?: GraphData }) {
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [selection, setSelection] = useState("");
  const drag = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const coords = useMemo(
    () => nodes.map((node) => {
      if (node.type === "Document") return { ...node, x: 450, y: 260 };
      const others = nodes.filter((item) => item.type !== "Document");
      const position = others.findIndex((item) => item.id === node.id);
      const angle = Math.PI * 2 * position / Math.max(1, others.length) - Math.PI / 2;
      const outer = others.length > 8 && position % 2;
      return {
        ...node,
        x: 450 + Math.cos(angle) * (outer ? 335 : 255),
        y: 260 + Math.sin(angle) * (outer ? 210 : 165),
      };
    }),
    [nodes],
  );
  const selectedNode = nodes.find((node) => selection === "node:" + node.id) || (!selection ? nodes[0] : undefined);
  const selectedEdge = edges.find((edge) => selection === "edge:" + edge.id);
  const nodeLabel = (id: string) => nodes.find((node) => node.id === id)?.label || id;
  if (!nodes.length)
    return (
      <div className="empty">
        No meaningful high-confidence entities were extracted.
      </div>
    );
  return (
    <div className="graph-view">
      <div className="graph-header">
        <div><span className="eyebrow">Structured {graph?.domain} graph</span><strong>{nodes.length} nodes · {edges.length} directed relationships</strong></div>
        <div className="graph-tools">
          <button onClick={() => setZoom(Math.max(.55, zoom - .15))}><ZoomOut size={16} /></button>
          <span>{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom(Math.min(2.2, zoom + .15))}><ZoomIn size={16} /></button>
          <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>Reset</button>
        </div>
      </div>
      <div className="graph-canvas">
        <svg viewBox="0 0 900 520" role="img" aria-label="Interactive extracted knowledge graph"
          onWheel={(event) => { event.preventDefault(); setZoom(Math.min(2.2, Math.max(.55, zoom - event.deltaY * .001))); }}
          onPointerDown={(event) => { drag.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y }; event.currentTarget.setPointerCapture(event.pointerId); }}
          onPointerMove={(event) => { if (drag.current) setPan({ x: drag.current.panX + (event.clientX - drag.current.x) / zoom, y: drag.current.panY + (event.clientY - drag.current.y) / zoom }); }}
          onPointerUp={() => { drag.current = null; }}>
          <defs><marker id="graph-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" /></marker></defs>
          <g transform={"translate(" + pan.x + " " + pan.y + ") scale(" + zoom + ")"}>
            {edges.map((edge) => {
              const source = coords.find((node) => node.id === edge.source);
              const target = coords.find((node) => node.id === edge.target);
              if (!source || !target) return null;
              return (
                <g key={edge.id} className={selection === "edge:" + edge.id ? "graph-edge selected" : "graph-edge"}
                  onClick={(event) => { event.stopPropagation(); setSelection("edge:" + edge.id); }}>
                  <line x1={source.x} y1={source.y} x2={target.x} y2={target.y} markerEnd="url(#graph-arrow)" />
                  <text x={(source.x + target.x) / 2} y={(source.y + target.y) / 2 - 7}>{edge.type.replaceAll("_", " ")}</text>
                </g>
              );
            })}
            {coords.map((node) => (
              <g key={node.id} className={"graph-node" + (node.type === "Document" ? " root" : "") + (selection === "node:" + node.id ? " selected" : "")}
                onClick={(event) => { event.stopPropagation(); setSelection("node:" + node.id); }}>
                <rect x={node.x - 68} y={node.y - 29} width="136" height="58" rx="14" />
                <text className="node-label" x={node.x} y={node.y - 2}>{node.label.length > 20 ? node.label.slice(0, 18) + "…" : node.label}</text>
                <text className="node-type" x={node.x} y={node.y + 15}>{node.type}</text>
              </g>
            ))}
          </g>
        </svg>
      </div>
      <div className="graph-inspector">
        {selectedEdge ? <>
          <span className="eyebrow">Selected relationship</span>
          <h3>{nodeLabel(selectedEdge.source)} → {nodeLabel(selectedEdge.target)}</h3>
          <strong>{selectedEdge.type.replaceAll("_", " ")}</strong>
          <p>{selectedEdge.provenance}</p>
          <small>Pages: {selectedEdge.pages?.join(", ") || "document"} · Token evidence: {selectedEdge.token_ids?.join(", ") || "metadata"}</small>
        </> : selectedNode ? <>
          <span className="eyebrow">Selected node</span><h3>{selectedNode.label}</h3><strong>{selectedNode.type}</strong>
          <p>Confidence: {(selectedNode.confidence * 100).toFixed(1)}% · Mentions: {selectedNode.mentions} · Pages: {selectedNode.pages.join(", ")}</p>
          <small>Token provenance: {selectedNode.token_ids?.join(", ") || "document metadata"}</small>
        </> : <p>Select a node or relationship to inspect its provenance.</p>}
      </div>
      <p className={graph?.persisted_to_neo4j ? "graph-status persisted" : "graph-status warning"}>
        {graph?.persisted_to_neo4j ? "Persisted to Neo4j" : "Neo4j unavailable for this run; showing the actual in-run graph"}
      </p>
    </div>
  );
}
export function ResultsCompare({ run }: { run: Run }) {
  const r = run.result;
  return (
    <div className="output-panel">
      <div className="output-actions">
        <div>
          <span className="eyebrow">Final deliverable</span>
          <h2>Corrected searchable PDF</h2>
          <p>Select, copy, and search the corrected text in the exported document.</p>
        </div>
        {r.export_pdf_url ? (
          <a className="button primary" href={`${API}${r.export_pdf_url}`} download>
            <Download size={17} /> Download PDF
          </a>
        ) : (
          <span className="muted">PDF is generated after validation completes.</span>
        )}
      </div>
      <div className="compare">
        <article>
          <span className="eyebrow">Original OCR</span>
          <p>{r.original_text || "Output pending."}</p>
        </article>
        <article>
          <span className="eyebrow">Corrected output</span>
          <p>{r.corrected_text || "Output pending."}</p>
        </article>
      </div>
    </div>
  );
}
