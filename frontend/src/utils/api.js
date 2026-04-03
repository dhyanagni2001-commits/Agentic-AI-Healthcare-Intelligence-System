// src/utils/api.js
// Centralised API client — all calls go through here.

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  // Health
  health: () => request("/health"),

  // IDP
  parse: (text, strict_mode = false) =>
    request("/parse", {
      method: "POST",
      body: JSON.stringify({ text, strict_mode }),
    }),

  // Validation
  validate: (facility_id) =>
    request("/validate", {
      method: "POST",
      body: JSON.stringify({ facility_id }),
    }),

  validateRecord: (hospital) =>
    request("/validate", {
      method: "POST",
      body: JSON.stringify({ hospital }),
    }),

  // Agent query
  query: (queryText, options = {}) =>
    request("/query", {
      method: "POST",
      body: JSON.stringify({
        query: queryText,
        state_filter: options.state || null,
        city_filter: options.city || null,
        include_reasoning: options.includeReasoning !== false,
        max_results: options.maxResults || 10,
      }),
    }),

  // Hospital data
  listHospitals: (params = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== "") qs.set(k, v); });
    return request(`/hospitals?${qs}`);
  },

  getHospital: (id) => request(`/hospitals/${encodeURIComponent(id)}`),

  // Analytics
  stats: () => request("/stats"),

  gaps: (state, city) => {
    const qs = new URLSearchParams();
    if (state) qs.set("state", state);
    if (city) qs.set("city", city);
    return request(`/gaps?${qs}`);
  },
};
