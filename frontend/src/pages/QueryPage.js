// src/pages/QueryPage.js
import React, { useState, useRef, useEffect } from "react";
import { api } from "../utils/api";

function RiskBadge({ level }) {
  return <span className={`badge badge-${level?.toLowerCase() || "low"}`}>{level || "low"}</span>;
}

function ReasoningTrace({ steps }) {
  const [open, setOpen] = useState(false);
  if (!steps?.length) return null;
  return (
    <div className="reasoning-trace">
      <div className="trace-header" onClick={() => setOpen(!open)}>
        <span>◎</span>
        <span>Reasoning trace ({steps.length} steps)</span>
        <span style={{ marginLeft: "auto" }}>{open ? "▲" : "▼"}</span>
      </div>
      {open && (
        <div className="trace-steps">
          {steps.map((step, i) => (
            <div className="trace-step" key={i}>
              <div className="trace-step-num">0{i + 1}</div>
              <div className="trace-step-content">
                <div className="trace-step-name">{step.step_name}</div>
                <div className="trace-step-summary">{step.output_summary}</div>
                {step.data_used?.length > 0 && (
                  <div className="trace-step-data">
                    Data used: {step.data_used.slice(0, 5).join(", ")}{step.data_used.length > 5 ? ` +${step.data_used.length - 5} more` : ""}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function GapList({ gaps }) {
  if (!gaps?.length) return null;
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 11, color: "var(--text-2)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>Gaps Detected</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {gaps.map((g, i) => (
          <div key={i} style={{ display: "flex", gap: 8, alignItems: "flex-start", padding: "8px 12px", background: "var(--bg-3)", borderRadius: 8, fontSize: 12 }}>
            <RiskBadge level={g.severity} />
            <span style={{ color: "var(--text-1)", flex: 1 }}>{g.description}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function RecList({ recs }) {
  if (!recs?.length) return null;
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 11, color: "var(--text-2)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>Recommendations</div>
      <div className="rec-list">
        {recs.slice(0, 3).map((r, i) => (
          <div key={i} className="rec-card" style={{ padding: "10px 14px" }}>
            <div className={`rec-priority-bar ${r.priority}`} />
            <div className="rec-body">
              <div className="rec-title" style={{ fontSize: 13 }}>{r.title}</div>
              <div className="rec-desc" style={{ fontSize: 12 }}>{r.description}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatAnswer(text) {
  if (!text) return null;
  const lines = text.split("\n");
  return lines.map((line, i) => {
    if (line.startsWith("**") && line.endsWith("**")) {
      return <div key={i} style={{ fontWeight: 600, color: "var(--text-0)", margin: "8px 0 4px" }}>{line.replace(/\*\*/g, "")}</div>;
    }
    if (line.startsWith("- ")) {
      return <div key={i} style={{ paddingLeft: 16, color: "var(--text-1)", margin: "2px 0" }}>• {line.slice(2)}</div>;
    }
    if (line === "") return <div key={i} style={{ height: 6 }} />;
    return <div key={i} style={{ margin: "2px 0" }}>{line}</div>;
  });
}

const SUGGESTED = [
  "What are the healthcare gaps in Texas?",
  "Find hospitals without emergency services in California",
  "Which regions have low doctor density?",
  "Recommend where to deploy doctors in New York",
  "Show high risk hospitals in Florida",
  "What specialties are missing in rural states?",
];

export default function QueryPage() {
  const [messages, setMessages] = useState([
    {
      role: "ai",
      content: "Hello! I'm your Healthcare Intelligence Agent. I can analyse hospital data, detect gaps, and generate recommendations.\n\nTry asking about specific states, hospital types, or healthcare capabilities.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [stateFilter, setStateFilter] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendQuery = async (queryText) => {
    const q = (queryText || input).trim();
    if (!q || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setLoading(true);
    try {
      const resp = await api.query(q, { state: stateFilter || null, includeReasoning: true, maxResults: 12 });
      setMessages((m) => [...m, { role: "ai", content: resp.answer, meta: resp }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "ai", content: `⚠ Error: ${err.message}`, error: true }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Header */}
      <div style={{ padding: "20px 24px 0", borderBottom: "1px solid var(--border)", background: "var(--bg-1)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
          <div>
            <h1 className="page-title" style={{ fontSize: 20 }}>AI Healthcare Query</h1>
            <p className="page-sub">Multi-step reasoning agent with gap detection</p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 11, color: "var(--text-2)", textTransform: "uppercase" }}>State Filter</span>
            <input
              className="input-field"
              style={{ width: 80 }}
              placeholder="e.g. TX"
              value={stateFilter}
              onChange={(e) => setStateFilter(e.target.value.toUpperCase())}
              maxLength={2}
            />
          </div>
        </div>
        {/* Suggested queries */}
        <div style={{ display: "flex", gap: 6, overflowX: "auto", paddingBottom: 12 }}>
          {SUGGESTED.map((s) => (
            <button
              key={s}
              onClick={() => sendQuery(s)}
              style={{
                padding: "5px 12px", borderRadius: 99, background: "var(--bg-3)",
                border: "1px solid var(--border)", fontSize: 11, color: "var(--text-1)",
                whiteSpace: "nowrap", cursor: "pointer", transition: "var(--transition)",
              }}
              onMouseEnter={(e) => e.target.style.borderColor = "var(--accent-mid)"}
              onMouseLeave={(e) => e.target.style.borderColor = "var(--border)"}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "20px 24px", display: "flex", flexDirection: "column", gap: 20 }}>
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className={`msg-avatar ${msg.role}`}>
              {msg.role === "ai" ? "⬡" : "↑"}
            </div>
            <div className="msg-body">
              <div className={`msg-bubble ${msg.error ? "error-bubble" : ""}`} style={msg.error ? { borderColor: "var(--risk-critical)" } : {}}>
                <div style={{ lineHeight: 1.65, fontSize: 13.5 }}>{formatAnswer(msg.content)}</div>
                {msg.meta && (
                  <>
                    <GapList gaps={msg.meta.gaps_identified} />
                    <RecList recs={msg.meta.recommendations} />
                    <ReasoningTrace steps={msg.meta.reasoning_steps} />
                    {msg.meta.hospitals_referenced?.length > 0 && (
                      <div style={{ marginTop: 10, fontSize: 11, color: "var(--text-2)" }}>
                        Data sources: {msg.meta.data_sources?.slice(0, 3).join(", ")}
                        {msg.meta.hospitals_referenced.length > 3 ? ` +${msg.meta.hospitals_referenced.length - 3} more` : ""}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="message ai">
            <div className="msg-avatar ai">⬡</div>
            <div className="msg-body">
              <div className="msg-bubble">
                <div style={{ display: "flex", gap: 8, alignItems: "center", color: "var(--text-2)" }}>
                  <div className="spinner" />
                  Running reasoning pipeline…
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="chat-input-bar">
        <div className="chat-input-wrap">
          <textarea
            className="chat-textarea"
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about hospitals, gaps, risk levels, or recommendations…"
          />
        </div>
        <button
          className="btn btn-primary"
          onClick={() => sendQuery()}
          disabled={loading || !input.trim()}
          style={{ height: 48, padding: "0 20px" }}
        >
          {loading ? <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> : "Send ◈"}
        </button>
      </div>
    </div>
  );
}
