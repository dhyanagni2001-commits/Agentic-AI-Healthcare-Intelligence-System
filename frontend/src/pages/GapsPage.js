// src/pages/GapsPage.js
import React, { useState } from "react";
import { api } from "../utils/api";

const US_STATES = [
  "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
  "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
  "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
  "TX","UT","VT","VA","WA","WV","WI","WY",
];

const GAP_ICONS = {
  no_emergency_services: "🚨",
  no_icu: "🏥",
  low_doctor_density: "👨‍⚕️",
  no_specialist: "🔬",
  low_quality_metrics: "📉",
  isolated_region: "📍",
};

const REC_ICONS = {
  invest_in_facility: "💰",
  patient_transfer: "🚑",
  deploy_staff: "👨‍⚕️",
  upgrade_capability: "⬆️",
  close_service_gap: "🎯",
};

function RiskBadge({ level }) {
  return <span className={`badge badge-${level?.toLowerCase() || "low"}`}>{level}</span>;
}

export default function GapsPage() {
  const [state, setState] = useState("TX");
  const [city, setCity] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("gaps");

  const runAnalysis = async () => {
    if (!state && !city) return;
    setLoading(true);
    setError(null);
    try {
      // Get gaps
      const gapResult = await api.gaps(state || null, city || null);
      // Get recommendations via query
      const recQuery = `What are the healthcare gaps and recommendations for ${city || ""} ${state || ""}?`.trim();
      const agentResult = await api.query(recQuery, { state: state || null, city: city || null, maxResults: 20 });
      setResult({ ...gapResult, recommendations: agentResult.recommendations });
      setActiveTab("gaps");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const overallRisk = result?.overall_risk;

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Healthcare Gap Analysis</h1>
        <p className="page-sub">Identify medical deserts and under-served regions</p>
      </div>

      {/* Controls */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
          <div style={{ flex: 1, minWidth: 160 }}>
            <label style={{ display: "block", fontSize: 11, color: "var(--text-2)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>State</label>
            <select className="select-field" value={state} onChange={(e) => setState(e.target.value)}>
              <option value="">Select state…</option>
              {US_STATES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div style={{ flex: 1, minWidth: 160 }}>
            <label style={{ display: "block", fontSize: 11, color: "var(--text-2)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>City (optional)</label>
            <input
              className="input-field"
              placeholder="e.g. Houston"
              value={city}
              onChange={(e) => setCity(e.target.value)}
            />
          </div>
          <button
            className="btn btn-primary"
            onClick={runAnalysis}
            disabled={loading || (!state && !city)}
            style={{ height: 42, padding: "0 24px" }}
          >
            {loading ? <><div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Analysing…</> : "Run Analysis"}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: 14, background: "#ff6b6b14", border: "1px solid #ff6b6b30", borderRadius: 10, marginBottom: 20, color: "var(--risk-critical)", fontSize: 13 }}>
          ⚠ {error}
        </div>
      )}

      {result && (
        <>
          {/* Summary header */}
          <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
            <div className="card" style={{ flex: 1, minWidth: 180 }}>
              <div className="stat-label">Region</div>
              <div className="stat-value" style={{ fontSize: 22 }}>
                {[city, state].filter(Boolean).join(", ") || "All"}
              </div>
            </div>
            <div className="card" style={{ flex: 1, minWidth: 180 }}>
              <div className="stat-label">Facilities Analysed</div>
              <div className="stat-value accent">{result.facilities_analyzed}</div>
            </div>
            <div className="card" style={{ flex: 1, minWidth: 180 }}>
              <div className="stat-label">Gaps Found</div>
              <div className={`stat-value ${result.gaps?.length > 0 ? "danger" : "accent"}`}>
                {result.gaps?.length ?? 0}
              </div>
            </div>
            <div className="card" style={{ flex: 1, minWidth: 180 }}>
              <div className="stat-label">Overall Risk</div>
              <div style={{ marginTop: 8 }}>
                <RiskBadge level={overallRisk} />
              </div>
              <div className="stat-sub" style={{ marginTop: 6 }}>{result.summary}</div>
            </div>
          </div>

          {/* Tabs */}
          <div style={{ display: "flex", gap: 4, marginBottom: 20, background: "var(--bg-2)", padding: 4, borderRadius: 10, width: "fit-content" }}>
            {["gaps", "recommendations"].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  padding: "7px 18px",
                  borderRadius: 7,
                  fontSize: 13,
                  fontWeight: 500,
                  background: activeTab === tab ? "var(--bg-3)" : "transparent",
                  color: activeTab === tab ? "var(--text-0)" : "var(--text-2)",
                  border: activeTab === tab ? "1px solid var(--border)" : "1px solid transparent",
                  transition: "var(--transition)",
                }}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
                <span style={{ marginLeft: 6, background: "var(--bg-4)", borderRadius: 99, padding: "1px 6px", fontSize: 11 }}>
                  {tab === "gaps" ? result.gaps?.length : result.recommendations?.length}
                </span>
              </button>
            ))}
          </div>

          {activeTab === "gaps" && (
            result.gaps?.length === 0 ? (
              <div className="card" style={{ textAlign: "center", padding: 40 }}>
                <div style={{ fontSize: 32, marginBottom: 12 }}>✅</div>
                <div style={{ color: "var(--accent)", fontWeight: 600 }}>No critical gaps detected</div>
                <div style={{ color: "var(--text-2)", fontSize: 13, marginTop: 4 }}>
                  This region meets the baseline healthcare coverage thresholds.
                </div>
              </div>
            ) : (
              <div className="gap-grid">
                {result.gaps.map((gap, i) => (
                  <div key={i} className="gap-card">
                    <div className="gap-card-header">
                      <div>
                        <div className="gap-type">{GAP_ICONS[gap.gap_type] || "⚠"} {gap.gap_type?.replace(/_/g, " ")}</div>
                        <div className="gap-desc" style={{ marginTop: 6 }}>{gap.description}</div>
                      </div>
                      <RiskBadge level={gap.severity} />
                    </div>
                    {gap.population_impact && (
                      <div className="gap-impact">⚠ {gap.population_impact}</div>
                    )}
                    {gap.affected_facilities?.length > 0 && (
                      <div className="gap-facilities">
                        {gap.affected_facilities.slice(0, 6).map((id) => (
                          <span key={id} className="facility-chip">{id}</span>
                        ))}
                        {gap.affected_facilities.length > 6 && (
                          <span className="facility-chip">+{gap.affected_facilities.length - 6}</span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )
          )}

          {activeTab === "recommendations" && (
            result.recommendations?.length === 0 ? (
              <div className="card" style={{ textAlign: "center", padding: 40 }}>
                <div style={{ fontSize: 32, marginBottom: 12 }}>✅</div>
                <div style={{ color: "var(--accent)", fontWeight: 600 }}>No recommendations needed</div>
              </div>
            ) : (
              <div className="rec-list">
                {result.recommendations.map((rec, i) => (
                  <div key={i} className="rec-card">
                    <div className={`rec-priority-bar ${rec.priority}`} />
                    <div style={{ fontSize: 20, marginTop: 2, flexShrink: 0 }}>
                      {REC_ICONS[rec.type] || "📋"}
                    </div>
                    <div className="rec-body">
                      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                        <div className="rec-title">{rec.title}</div>
                        <RiskBadge level={rec.priority} />
                      </div>
                      <div className="rec-desc">{rec.description}</div>
                      {rec.rationale && <div className="rec-rationale">Rationale: {rec.rationale}</div>}
                      {rec.estimated_impact && <div className="rec-impact">→ {rec.estimated_impact}</div>}
                    </div>
                  </div>
                ))}
              </div>
            )
          )}
        </>
      )}

      {!result && !loading && (
        <div className="empty">
          <div className="empty-icon">◎</div>
          Select a state or city above and click "Run Analysis" to detect healthcare gaps and get recommendations.
        </div>
      )}
    </div>
  );
}
