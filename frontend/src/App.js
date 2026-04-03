// src/App.js
import React, { useState, useEffect } from "react";
import Dashboard from "./pages/Dashboard";
import HospitalsPage from "./pages/HospitalsPage";
import QueryPage from "./pages/QueryPage";
import ParsePage from "./pages/ParsePage";
import GapsPage from "./pages/GapsPage";
import { api } from "./utils/api";
import "./App.css";

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: "⬡" },
  { id: "query", label: "AI Query", icon: "◈" },
  { id: "hospitals", label: "Hospitals", icon: "⊞" },
  { id: "gaps", label: "Gap Analysis", icon: "◎" },
  { id: "parse", label: "Parse Text", icon: "⟐" },
];

export default function App() {
  const [page, setPage] = useState("dashboard");
  const [systemStatus, setSystemStatus] = useState(null);

  useEffect(() => {
    api.health()
      .then(setSystemStatus)
      .catch(() => setSystemStatus({ status: "error" }));
  }, []);

  const isOnline = systemStatus?.status === "ok";

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-mark">⬡</div>
          <div className="logo-text">
            <span className="logo-title">HealthIQ</span>
            <span className="logo-sub">Intelligence Platform</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${page === item.id ? "active" : ""}`}
              onClick={() => setPage(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className={`status-dot ${isOnline ? "online" : "offline"}`} />
          <span className="status-text">
            {systemStatus === null
              ? "Connecting…"
              : isOnline
              ? `${systemStatus.hospitals_loaded?.toLocaleString()} hospitals loaded`
              : "Backend offline"}
          </span>
        </div>
      </aside>

      <main className="main-content">
        {page === "dashboard" && <Dashboard onNavigate={setPage} />}
        {page === "query" && <QueryPage />}
        {page === "hospitals" && <HospitalsPage />}
        {page === "gaps" && <GapsPage />}
        {page === "parse" && <ParsePage />}
      </main>
    </div>
  );
}
