// src/pages/ParsePage.js
import React, { useState } from "react";
import { api } from "../utils/api";

const EXAMPLES = [
  {
    label: "Complete record",
    text: `St. Catherine Medical Center is located in Austin, TX 78701. 
The facility is a Level I Trauma Center with full emergency services, ICU, surgery, maternity ward, and pediatric unit. 
Staff includes 45 doctors, 80 nurses. Hospital rating: 4. 200 beds available.`,
  },
  {
    label: "Minimal info",
    text: `Community clinic in rural AR. Some emergency capabilities. 2 doctors on staff.`,
  },
  {
    label: "Messy / contradictory",
    text: `Downtown Health Center - we have surgery suites but currently no surgical staff (0 doctors). 
Located in Chicago IL. Emergency room operational. ICU being renovated. Rating 2/5.`,
  },
  {
    label: "Missing fields",
    text: `New Hope Hospital. Has ICU, cardiac care unit. About 30 physicians. 
No emergency services at this time.`,
  },
];

function CapabilityGrid({ caps }) {
  if (!caps) return null;
  return (
    <div className="cap-grid" style={{ marginTop: 8 }}>
      {Object.entries(caps).map(([k, v]) => (
        <span key={k} className={`cap-chip ${v ? "has" : "no"}`}>
          {v ? "✓" : "✗"} {k.replace(/_/g, " ")}
        </span>
      ))}
    </div>
  );
}

function QualityRow({ label, value }) {
  if (!value) return null;
  const color = value.includes("Below") ? "var(--risk-critical)" : value.includes("Above") ? "var(--risk-low)" : "var(--text-2)";
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", fontSize: 12 }}>
      <span style={{ color: "var(--text-2)" }}>{label}</span>
      <span style={{ color }}>{value}</span>
    </div>
  );
}

function ValidationPanel({ result }) {
  if (!result) return null;
  const risk = result.risk_level;
  const valid = result.valid;
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div className="card-header">
        <span className="card-title">Validation Result</span>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span className={`badge badge-${risk}`}>{risk} risk</span>
          <span style={{ fontSize: 12, color: valid ? "var(--risk-low)" : "var(--risk-critical)" }}>
            {valid ? "✓ Valid" : "✗ Issues found"}
          </span>
        </div>
      </div>

      {result.issues?.length === 0 ? (
        <div style={{ color: "var(--risk-low)", fontSize: 13 }}>✓ No issues detected</div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {result.issues.map((issue, i) => (
            <div key={i} style={{
              padding: "10px 12px",
              background: "var(--bg-2)",
              borderRadius: 8,
              borderLeft: `3px solid var(--risk-${issue.severity})`,
              fontSize: 12,
            }}>
              <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 3 }}>
                <span className={`badge badge-${issue.severity}`} style={{ fontSize: 10 }}>{issue.severity}</span>
                <span style={{ color: "var(--text-2)", fontFamily: "var(--font-mono)", fontSize: 11 }}>{issue.field}</span>
              </div>
              <div style={{ color: "var(--text-1)" }}>{issue.issue}</div>
            </div>
          ))}
        </div>
      )}

      {result.warnings?.length > 0 && (
        <div style={{ marginTop: 10, fontSize: 12, color: "var(--text-2)" }}>
          {result.warnings.map((w, i) => <div key={i}>{w}</div>)}
        </div>
      )}

      <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-2)", display: "flex", gap: 16 }}>
        <span>Facility: <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-1)" }}>{result.facility_id}</span></span>
        <span>Confidence: <span style={{ color: result.confidence_score >= 0.7 ? "var(--risk-low)" : "var(--risk-high)" }}>{(result.confidence_score * 100).toFixed(0)}%</span></span>
      </div>
    </div>
  );
}

export default function ParsePage() {
  const [text, setText] = useState(EXAMPLES[0].text);
  const [parseResult, setParseResult] = useState(null);
  const [validResult, setValidResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [strictMode, setStrictMode] = useState(false);

  const handleParse = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setValidResult(null);
    try {
      const parsed = await api.parse(text, strictMode);
      setParseResult(parsed);
      if (parsed.success && parsed.hospital) {
        const validation = await api.validateRecord(parsed.hospital);
        setValidResult(validation);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const h = parseResult?.hospital;

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Intelligent Document Parser</h1>
        <p className="page-sub">Extract structured hospital data from unstructured text</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, alignItems: "start" }}>
        {/* Input panel */}
        <div>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-header">
              <span className="card-title">Input Text</span>
              <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-2)", cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={strictMode}
                  onChange={(e) => setStrictMode(e.target.checked)}
                  style={{ accentColor: "var(--accent)" }}
                />
                Strict Mode
              </label>
            </div>
            <textarea
              className="input-field"
              style={{ minHeight: 220, resize: "vertical" }}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste hospital description text here…"
            />
            <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
              <button
                className="btn btn-primary"
                onClick={handleParse}
                disabled={loading || !text.trim()}
              >
                {loading ? <><div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Parsing…</> : "⟐ Parse & Validate"}
              </button>
              <button className="btn btn-secondary" onClick={() => { setParseResult(null); setValidResult(null); setText(""); }}>
                Clear
              </button>
            </div>
          </div>

          {/* Example templates */}
          <div className="card">
            <div className="card-header"><span className="card-title">Example Inputs</span></div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.label}
                  className="btn btn-secondary"
                  style={{ justifyContent: "flex-start", fontSize: 12, padding: "8px 12px" }}
                  onClick={() => { setText(ex.text); setParseResult(null); setValidResult(null); }}
                >
                  <span style={{ color: "var(--accent)", marginRight: 6 }}>⟐</span>
                  {ex.label}
                </button>
              ))}
            </div>
          </div>

          {error && (
            <div style={{ marginTop: 14, padding: 12, background: "#ff6b6b14", border: "1px solid #ff6b6b30", borderRadius: 8, color: "var(--risk-critical)", fontSize: 13 }}>
              ⚠ {error}
            </div>
          )}
        </div>

        {/* Output panel */}
        <div>
          {!parseResult && !loading && (
            <div className="empty" style={{ padding: 60 }}>
              <div className="empty-icon">⟐</div>
              Enter hospital text and click "Parse & Validate"
            </div>
          )}

          {parseResult && (
            <>
              {/* Status */}
              <div style={{ display: "flex", gap: 10, marginBottom: 16, padding: "10px 14px", background: parseResult.success ? "var(--accent-dim)" : "#ff6b6b14", border: `1px solid ${parseResult.success ? "var(--accent-mid)" : "#ff6b6b30"}`, borderRadius: 10, alignItems: "center" }}>
                <span style={{ fontSize: 18 }}>{parseResult.success ? "✓" : "✗"}</span>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 13, color: parseResult.success ? "var(--accent)" : "var(--risk-critical)" }}>
                    {parseResult.success ? "Parsing successful" : "Parsing failed"}
                  </div>
                  {parseResult.processing_notes?.length > 0 && (
                    <div style={{ fontSize: 12, color: "var(--text-2)", marginTop: 2 }}>
                      {parseResult.processing_notes.join(" · ")}
                    </div>
                  )}
                </div>
                {h && (
                  <div style={{ marginLeft: "auto", textAlign: "right" }}>
                    <div style={{ fontSize: 11, color: "var(--text-2)" }}>Confidence</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: h.confidence_score >= 0.7 ? "var(--risk-low)" : "var(--risk-high)", fontFamily: "var(--font-mono)" }}>
                      {(h.confidence_score * 100).toFixed(0)}%
                    </div>
                  </div>
                )}
              </div>

              {h && (
                <div className="card">
                  <div className="card-header">
                    <span className="card-title">Extracted Record</span>
                    <span className="td-mono" style={{ fontSize: 11 }}>{h.facility_id}</span>
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    <div>
                      <h3 style={{ fontFamily: "var(--font-display)", fontSize: 17, marginBottom: 4 }}>{h.facility_name}</h3>
                      <p style={{ color: "var(--text-2)", fontSize: 12 }}>
                        {[h.city, h.state, h.zip_code].filter(Boolean).join(", ") || "Location unknown"}
                      </p>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                      {[
                        ["Hospital Type", h.hospital_type],
                        ["Ownership", h.ownership],
                        ["Phone", h.phone],
                        ["Doctor Count", h.doctor_count > 0 ? h.doctor_count : null],
                        ["Overall Rating", h.quality?.overall_rating ? `${h.quality.overall_rating}/5` : null],
                        ["Notes", h.notes],
                      ].filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => (
                        <div key={k} style={{ background: "var(--bg-2)", borderRadius: 8, padding: "8px 12px" }}>
                          <div style={{ fontSize: 10, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.08em" }}>{k}</div>
                          <div style={{ fontSize: 13, color: "var(--text-0)", marginTop: 2 }}>{v}</div>
                        </div>
                      ))}
                    </div>

                    <div>
                      <div style={{ fontSize: 11, color: "var(--text-2)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>Capabilities</div>
                      <CapabilityGrid caps={h.capabilities} />
                    </div>
                  </div>
                </div>
              )}

              <ValidationPanel result={validResult} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
