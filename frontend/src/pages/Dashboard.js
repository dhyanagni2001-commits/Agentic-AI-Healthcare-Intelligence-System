// src/pages/Dashboard.js
import React, { useEffect, useState } from "react";
import { api } from "../utils/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const QUICK_QUERIES = [
  "Which states have the most hospitals without emergency services?",
  "Show me high-risk hospitals in Texas",
  "What are the biggest healthcare gaps in rural areas?",
  "Find hospitals with low doctor density",
];

export default function Dashboard({ onNavigate }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.stats()
      .then(setStats)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="page">
      <div className="loader"><div className="spinner" /> Loading dashboard…</div>
    </div>
  );

  if (!stats) return (
    <div className="page">
      <div className="empty"><div className="empty-icon">⚠</div>Could not load statistics. Is the backend running?</div>
    </div>
  );

  const stateData = Object.entries(stats.top_states_by_hospital_count || {})
    .map(([state, count]) => ({ state, count }))
    .slice(0, 10);

  const typeData = Object.entries(stats.hospital_type_distribution || {})
    .map(([type, count]) => ({ type: type.replace("Hospital", "").trim(), count }))
    .slice(0, 8);

  const erPct = stats.total_hospitals
    ? Math.round((stats.hospitals_with_emergency / stats.total_hospitals) * 100)
    : 0;

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Healthcare Intelligence Overview</h1>
        <p className="page-sub">Real-time analysis of {stats.total_hospitals?.toLocaleString()} US hospital facilities</p>
      </div>

      {/* Stats */}
      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-label">Total Hospitals</div>
          <div className="stat-value accent">{stats.total_hospitals?.toLocaleString()}</div>
          <div className="stat-sub">Across {stats.states_covered} states</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">With Emergency</div>
          <div className="stat-value">{stats.hospitals_with_emergency?.toLocaleString()}</div>
          <div className="stat-sub">{erPct}% of all facilities</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Without Emergency</div>
          <div className="stat-value danger">{stats.hospitals_without_emergency?.toLocaleString()}</div>
          <div className="stat-sub">Potential coverage gaps</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg Rating</div>
          <div className="stat-value">{stats.average_rating?.toFixed(1) ?? "N/A"}<span style={{fontSize:14,color:"var(--text-2)"}}>/5</span></div>
          <div className="stat-sub">National average</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">High Risk</div>
          <div className="stat-value warning">{stats.high_risk_hospitals?.toLocaleString()}</div>
          <div className="stat-sub">Multiple below-avg metrics</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total Doctors</div>
          <div className="stat-value accent">{stats.total_doctors?.toLocaleString()}</div>
          <div className="stat-sub">Across all facilities</div>
        </div>
      </div>

      {/* Charts row */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 28 }}>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Hospitals by State (Top 10)</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={stateData} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
              <XAxis dataKey="state" tick={{ fill: "#5a7090", fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#5a7090", fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: "#141b25", border: "1px solid #1e2d44", borderRadius: 8, fontSize: 12 }}
                cursor={{ fill: "#1c2535" }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {stateData.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? "#00e5c3" : "#1c2535"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Hospital Types</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={typeData} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 10 }}>
              <XAxis type="number" tick={{ fill: "#5a7090", fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis dataKey="type" type="category" tick={{ fill: "#a8b8d8", fontSize: 11 }} axisLine={false} tickLine={false} width={120} />
              <Tooltip
                contentStyle={{ background: "#141b25", border: "1px solid #1e2d44", borderRadius: 8, fontSize: 12 }}
                cursor={{ fill: "#1c2535" }}
              />
              <Bar dataKey="count" fill="#7c6dfa" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Emergency coverage bar */}
      <div className="card" style={{ marginBottom: 28 }}>
        <div className="card-header">
          <span className="card-title">Emergency Service Coverage</span>
          <span style={{ fontSize: 12, color: "var(--text-2)" }}>{erPct}% covered</span>
        </div>
        <div style={{ background: "var(--bg-3)", borderRadius: 4, height: 8, overflow: "hidden" }}>
          <div style={{ width: `${erPct}%`, height: "100%", background: "linear-gradient(90deg, #00e5c3, #7c6dfa)", borderRadius: 4, transition: "width 1s ease" }} />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8, fontSize: 12, color: "var(--text-2)" }}>
          <span>{stats.hospitals_with_emergency?.toLocaleString()} with emergency services</span>
          <span>{stats.hospitals_without_emergency?.toLocaleString()} without</span>
        </div>
      </div>

      {/* Quick queries */}
      <div className="card">
        <div className="card-header"><span className="card-title">Quick AI Queries</span></div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {QUICK_QUERIES.map((q) => (
            <button
              key={q}
              className="btn btn-secondary"
              style={{ justifyContent: "flex-start", textAlign: "left" }}
              onClick={() => onNavigate("query")}
            >
              <span style={{ color: "var(--accent)", fontSize: 12 }}>◈</span>
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
