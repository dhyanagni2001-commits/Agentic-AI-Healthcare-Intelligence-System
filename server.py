"""
Healthcare Intelligence Platform — HTTP Server
Pure Python stdlib: http.server + json + urllib.parse
Runs with: python3 server.py

Endpoints:
  GET  /health
  GET  /stats
  GET  /hospitals          ?page=1&per_page=20&state=TX&has_emergency=true&min_rating=3
  GET  /hospitals/<id>
  GET  /gaps               ?state=TX&city=Houston
  POST /parse
  POST /validate
  POST /query
"""
from __future__ import annotations
import json, sys, logging, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from collections import Counter
from typing import Any, Dict, List, Optional
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Lazy globals (populated at startup) ───────────────────────────────────────
_hospitals: Dict[str, Any] = {}
_index: Any = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("server")


def _load():
    global _hospitals, _index
    from backend.services.data_loader import load_all_hospitals
    from backend.services.rag_service import build_index
    log.info("Loading hospital data…")
    _hospitals = load_all_hospitals()
    log.info(f"Loaded {len(_hospitals)} hospitals — building index…")
    _index = build_index(_hospitals)
    log.info("Ready.")


# ── Response helpers ───────────────────────────────────────────────────────────

def _json(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, default=str).encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info(f"{self.address_string()} {fmt % args}")

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _send(self, code: int, data: Any):
        body = _json(data)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return {}

    # ── Routing ────────────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        qs     = parse_qs(parsed.query)

        def q(k, default=None):
            vals = qs.get(k)
            return vals[0] if vals else default

        if path == "/health":
            self._send(200, {
                "status": "ok",
                "hospitals_loaded": len(_hospitals),
                "index_ready": _index is not None and _index.total > 0,
            })

        elif path == "/stats":
            self._handle_stats()

        elif path == "/hospitals":
            self._handle_list(qs, q)

        elif path.startswith("/hospitals/"):
            fid = path[len("/hospitals/"):]
            h = _hospitals.get(fid)
            if h:
                self._send(200, h.to_dict())
            else:
                self._send(404, {"detail": f"Hospital '{fid}' not found"})

        elif path == "/gaps":
            self._handle_gaps(q("state"), q("city"))

        else:
            self._send(404, {"detail": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        body   = self._body()

        if path == "/parse":
            self._handle_parse(body)
        elif path == "/validate":
            self._handle_validate(body)
        elif path == "/query":
            self._handle_query(body)
        else:
            self._send(404, {"detail": "Not found"})

    # ── Handlers ───────────────────────────────────────────────────────────────

    def _handle_stats(self):
        items = list(_hospitals.values())
        total = len(items)
        if not total:
            self._send(503, {"detail": "No data loaded"}); return
        ratings = [h.quality.overall_rating for h in items if h.quality.overall_rating]
        avg = round(sum(ratings)/len(ratings), 2) if ratings else None

        def is_high_risk(h):
            metrics = [h.quality.mortality_comparison, h.quality.safety_comparison,
                       h.quality.readmission_comparison, h.quality.effectiveness_comparison]
            return sum(1 for m in metrics if m and "below" in m.lower()) >= 2

        state_c = Counter(h.state for h in items if h.state)
        type_c  = Counter(h.hospital_type for h in items if h.hospital_type)
        self._send(200, {
            "total_hospitals": total,
            "states_covered": len(state_c),
            "hospitals_with_emergency":    sum(1 for h in items if h.capabilities.emergency_services),
            "hospitals_without_emergency": sum(1 for h in items if not h.capabilities.emergency_services),
            "average_rating": avg,
            "high_risk_hospitals": sum(1 for h in items if is_high_risk(h)),
            "total_doctors": sum(h.doctor_count for h in items),
            "total_departments": sum(h.department_count for h in items),
            "top_states_by_hospital_count": dict(state_c.most_common(10)),
            "hospital_type_distribution": dict(type_c.most_common(15)),
        })

    def _handle_list(self, qs, q):
        items = list(_hospitals.values())
        state = q("state")
        city  = q("city")
        has_er = q("has_emergency")
        min_r  = q("min_rating")
        h_type = q("hospital_type")

        if state:  items = [h for h in items if (h.state or "").upper() == state.upper()]
        if city:   items = [h for h in items if city.lower() in (h.city or "").lower()]
        if has_er == "true":  items = [h for h in items if h.capabilities.emergency_services]
        if has_er == "false": items = [h for h in items if not h.capabilities.emergency_services]
        if min_r:
            try:
                mr = int(min_r)
                items = [h for h in items if h.quality.overall_rating and h.quality.overall_rating >= mr]
            except: pass
        if h_type: items = [h for h in items if h_type.lower() in (h.hospital_type or "").lower()]

        total    = len(items)
        page     = max(1, int(q("page") or 1))
        per_page = min(100, max(1, int(q("per_page") or 20)))
        start    = (page - 1) * per_page
        self._send(200, {
            "total": total, "page": page, "per_page": per_page,
            "hospitals": [h.to_dict() for h in items[start:start + per_page]],
        })

    def _handle_gaps(self, state: Optional[str], city: Optional[str]):
        from backend.services.gap_detection import analyse_region
        items = list(_hospitals.values())
        if state: items = [h for h in items if (h.state or "").upper() == state.upper()]
        if city:  items = [h for h in items if city.lower() in (h.city or "").lower()]
        if not items:
            self._send(404, {"detail": "No hospitals found for the specified region"}); return
        result = analyse_region(items, state=state, city=city)
        self._send(200, result.to_dict())

    def _handle_parse(self, body: Dict):
        from backend.services.idp_service import parse_hospital_text
        from backend.services.validation_service import validate_hospital
        text = body.get("text", "")
        strict = bool(body.get("strict_mode", False))
        if not text or len(text.strip()) < 3:
            self._send(400, {"detail": "text field is required (min 3 chars)"}); return
        result = parse_hospital_text(text, strict_mode=strict)
        d = result.to_dict()
        # Also attach validation if parse succeeded
        if result.success and result.hospital:
            vr = validate_hospital(result.hospital)
            d["validation"] = vr.to_dict()
        self._send(200, d)

    def _handle_validate(self, body: Dict):
        from backend.services.validation_service import validate_hospital
        from backend.models.schemas import HospitalRecord
        fid  = body.get("facility_id")
        hosp = body.get("hospital")
        if fid:
            h = _hospitals.get(fid)
            if not h:
                self._send(404, {"detail": f"Hospital '{fid}' not found"}); return
        elif hosp:
            try: h = HospitalRecord.from_dict(hosp)
            except Exception as e:
                self._send(400, {"detail": f"Invalid hospital object: {e}"}); return
        else:
            self._send(400, {"detail": "Provide facility_id or hospital"}); return
        self._send(200, validate_hospital(h).to_dict())

    def _handle_query(self, body: Dict):
        from backend.agents.healthcare_agent import run_agent
        if _index is None or _index.total == 0:
            self._send(503, {"detail": "Search index not ready"}); return
        query = body.get("query", "").strip()
        if not query:
            self._send(400, {"detail": "query field is required"}); return
        resp = run_agent(
            query=query,
            index=_index,
            state_filter=body.get("state_filter"),
            city_filter=body.get("city_filter"),
            include_reasoning=body.get("include_reasoning", True),
            max_results=min(50, max(1, int(body.get("max_results", 10)))),
        )
        self._send(200, resp.to_dict())


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    # Load data in background so server starts immediately
    t = threading.Thread(target=_load, daemon=True)
    t.start()
    t.join()  # wait for data before accepting connections

    server = HTTPServer(("0.0.0.0", port), Handler)
    log.info(f"Server running on http://0.0.0.0:{port}")
    log.info("API docs: see README.md")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
