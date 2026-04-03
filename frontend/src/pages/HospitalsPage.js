// src/pages/HospitalsPage.js
import React, { useState, useEffect, useCallback } from "react";
import { api } from "../utils/api";

const US_STATES = [
  "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
  "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
  "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
  "TX","UT","VT","VA","WA","WV","WI","WY",
];

function RatingStars({ rating }) {
  if (!rating) return <span style={{ color: "var(--text-3)" }}>—</span>;
  return (
    <span style={{ fontSize: 13, color: "var(--accent-4)" }}>
      {"★".repeat(rating)}{"☆".repeat(5 - rating)}
      <span style={{ fontSize: 11, color: "var(--text-2)", marginLeft: 4 }}>{rating}/5</span>
    </span>
  );
}

function CapBadges({ caps }) {
  const entries = Object.entries(caps || {});
  const active = entries.filter(([, v]) => v).map(([k]) => k.replace(/_/g, " "));
  if (!active.length) return <span style={{ color: "var(--text-3)", fontSize: 11 }}>None</span>;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
      {active.slice(0, 4).map((c) => (
        <span key={c} className="cap-chip has" style={{ fontSize: 10, padding: "2px 7px" }}>{c}</span>
      ))}
      {active.length > 4 && <span className="cap-chip has" style={{ fontSize: 10, padding: "2px 7px" }}>+{active.length - 4}</span>}
    </div>
  );
}

function HospitalDrawer({ hospital, onClose }) {
  if (!hospital) return null;
  const h = hospital;
  return (
    <div style={{
      position: "fixed", top: 0, right: 0, bottom: 0, width: 420,
      background: "var(--bg-1)", borderLeft: "1px solid var(--border)",
      overflowY: "auto", zIndex: 100, padding: 24, animation: "fadeUp 0.2s ease",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 20, alignItems: "flex-start" }}>
        <div>
          <h2 style={{ fontFamily: "var(--font-display)", fontSize: 18, fontWeight: 700 }}>{h.facility_name}</h2>
          <p style={{ color: "var(--text-2)", fontSize: 12, marginTop: 4 }}>
            {h.address && `${h.address}, `}{h.city}, {h.state} {h.zip_code}
          </p>
        </div>
        <button onClick={onClose} style={{ color: "var(--text-2)", fontSize: 20, padding: 4 }}>✕</button>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {/* Info */}
        <div className="card" style={{ padding: 16 }}>
          <div className="card-title" style={{ marginBottom: 12 }}>Facility Info</div>
          {[
            ["Type", h.hospital_type || "—"],
            ["Ownership", h.ownership || "—"],
            ["Phone", h.phone || "—"],
            ["County", h.county || "—"],
            ["Facility ID", h.facility_id],
          ].map(([k, v]) => (
            <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid var(--border)", fontSize: 12 }}>
              <span style={{ color: "var(--text-2)" }}>{k}</span>
              <span style={{ color: "var(--text-0)", textAlign: "right", maxWidth: 220 }}>{v}</span>
            </div>
          ))}
        </div>

        {/* Quality */}
        <div className="card" style={{ padding: 16 }}>
          <div className="card-title" style={{ marginBottom: 12 }}>Quality Metrics</div>
          <div style={{ marginBottom: 10 }}>
            <RatingStars rating={h.quality?.overall_rating} />
          </div>
          {[
            ["Mortality", h.quality?.mortality_comparison],
            ["Safety", h.quality?.safety_comparison],
            ["Readmission", h.quality?.readmission_comparison],
            ["Patient Experience", h.quality?.patient_experience_comparison],
            ["Effectiveness", h.quality?.effectiveness_comparison],
            ["Timeliness", h.quality?.timeliness_comparison],
          ].map(([k, v]) => {
            if (!v) return null;
            const color = v.includes("Below") ? "var(--risk-critical)" : v.includes("Above") ? "var(--risk-low)" : "var(--text-2)";
            return (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", fontSize: 12 }}>
                <span style={{ color: "var(--text-2)" }}>{k}</span>
                <span style={{ color, fontSize: 11 }}>{v}</span>
              </div>
            );
          })}
        </div>

        {/* Capabilities */}
        <div className="card" style={{ padding: 16 }}>
          <div className="card-title" style={{ marginBottom: 12 }}>Capabilities</div>
          <div className="cap-grid">
            {Object.entries(h.capabilities || {}).map(([k, v]) => (
              <span key={k} className={`cap-chip ${v ? "has" : "no"}`} style={{ fontSize: 11 }}>
                {v ? "✓" : "✗"} {k.replace(/_/g, " ")}
              </span>
            ))}
          </div>
        </div>

        {/* Staffing */}
        <div className="card" style={{ padding: 16 }}>
          <div className="card-title" style={{ marginBottom: 12 }}>Staffing</div>
          <div style={{ display: "flex", gap: 20 }}>
            <div>
              <div style={{ fontSize: 24, fontWeight: 700, color: "var(--accent)", fontFamily: "var(--font-display)" }}>{h.doctor_count}</div>
              <div style={{ fontSize: 11, color: "var(--text-2)" }}>Total Doctors</div>
            </div>
            <div>
              <div style={{ fontSize: 24, fontWeight: 700, color: "var(--accent-2)", fontFamily: "var(--font-display)" }}>{h.department_count}</div>
              <div style={{ fontSize: 11, color: "var(--text-2)" }}>Departments</div>
            </div>
          </div>
          {h.departments?.length > 0 && (
            <div style={{ marginTop: 12, display: "flex", flexWrap: "wrap", gap: 4 }}>
              {h.departments.slice(0, 12).map((d) => (
                <span key={d} style={{ background: "var(--bg-3)", border: "1px solid var(--border)", borderRadius: 4, padding: "2px 8px", fontSize: 11, color: "var(--text-1)" }}>{d}</span>
              ))}
              {h.departments.length > 12 && (
                <span style={{ fontSize: 11, color: "var(--text-2)" }}>+{h.departments.length - 12} more</span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function HospitalsPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(null);
  const [page, setPage] = useState(1);
  const [filters, setFilters] = useState({ state: "", has_emergency: "", min_rating: "", hospital_type: "" });

  const load = useCallback(() => {
    setLoading(true);
    const params = { page, per_page: 20, ...filters };
    Object.keys(params).forEach((k) => { if (!params[k]) delete params[k]; });
    api.listHospitals(params)
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [page, filters]);

  useEffect(() => { load(); }, [load]);

  const setFilter = (k, v) => { setPage(1); setFilters((f) => ({ ...f, [k]: v })); };

  const totalPages = data ? Math.ceil(data.total / 20) : 1;

  return (
    <div className="page" style={{ maxWidth: "100%" }}>
      <div className="page-header">
        <h1 className="page-title">Hospital Directory</h1>
        <p className="page-sub">{data ? `${data.total?.toLocaleString()} hospitals matching filters` : "Loading…"}</p>
      </div>

      {/* Filters */}
      <div className="filters-bar">
        <div className="filter-group">
          <span className="filter-label">State</span>
          <select className="select-field" style={{ width: 90 }} value={filters.state} onChange={(e) => setFilter("state", e.target.value)}>
            <option value="">All</option>
            {US_STATES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className="filter-group">
          <span className="filter-label">Emergency</span>
          <select className="select-field" style={{ width: 100 }} value={filters.has_emergency} onChange={(e) => setFilter("has_emergency", e.target.value)}>
            <option value="">Any</option>
            <option value="true">Has ER</option>
            <option value="false">No ER</option>
          </select>
        </div>
        <div className="filter-group">
          <span className="filter-label">Min Rating</span>
          <select className="select-field" style={{ width: 80 }} value={filters.min_rating} onChange={(e) => setFilter("min_rating", e.target.value)}>
            <option value="">Any</option>
            {[1, 2, 3, 4, 5].map((r) => <option key={r} value={r}>{r}★+</option>)}
          </select>
        </div>
        <button className="btn btn-secondary" onClick={() => { setPage(1); setFilters({ state: "", has_emergency: "", min_rating: "", hospital_type: "" }); }}>
          Clear
        </button>
      </div>

      {loading ? (
        <div className="loader"><div className="spinner" /> Loading hospitals…</div>
      ) : (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Hospital</th>
                  <th>Location</th>
                  <th>Type</th>
                  <th>Emergency</th>
                  <th>Rating</th>
                  <th>Doctors</th>
                  <th>Capabilities</th>
                </tr>
              </thead>
              <tbody>
                {data?.hospitals?.map((h) => (
                  <tr key={h.facility_id} onClick={() => setSelected(h)} style={{ cursor: "pointer" }}>
                    <td>
                      <div className="td-name">{h.facility_name}</div>
                      <div className="td-mono">{h.facility_id}</div>
                    </td>
                    <td>
                      <div style={{ color: "var(--text-0)" }}>{h.city}</div>
                      <div style={{ color: "var(--text-2)", fontSize: 11 }}>{h.state} {h.zip_code}</div>
                    </td>
                    <td style={{ color: "var(--text-1)", maxWidth: 140 }}>
                      <div style={{ fontSize: 12, lineHeight: 1.4 }}>{h.hospital_type || "—"}</div>
                    </td>
                    <td>
                      {h.capabilities?.emergency_services ? (
                        <span className="badge badge-low">✓ Yes</span>
                      ) : (
                        <span className="badge badge-critical">✗ No</span>
                      )}
                    </td>
                    <td><RatingStars rating={h.quality?.overall_rating} /></td>
                    <td>
                      <span style={{ color: h.doctor_count > 0 ? "var(--accent)" : "var(--text-3)", fontFamily: "var(--font-mono)", fontSize: 13 }}>
                        {h.doctor_count}
                      </span>
                    </td>
                    <td><CapBadges caps={h.capabilities} /></td>
                  </tr>
                ))}
                {!data?.hospitals?.length && (
                  <tr><td colSpan={7} className="empty">No hospitals match the current filters</td></tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="pagination">
            <span className="pagination-info">Page {page} of {totalPages}</span>
            <button className="page-btn" onClick={() => setPage(1)} disabled={page === 1}>«</button>
            <button className="page-btn" onClick={() => setPage((p) => p - 1)} disabled={page === 1}>‹</button>
            {[page - 1, page, page + 1].filter((p) => p >= 1 && p <= totalPages).map((p) => (
              <button key={p} className={`page-btn ${p === page ? "active" : ""}`} onClick={() => setPage(p)}>{p}</button>
            ))}
            <button className="page-btn" onClick={() => setPage((p) => p + 1)} disabled={page === totalPages}>›</button>
            <button className="page-btn" onClick={() => setPage(totalPages)} disabled={page === totalPages}>»</button>
          </div>
        </>
      )}

      <HospitalDrawer hospital={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
