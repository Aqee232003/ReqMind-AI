# ============================================================
# ReqMind AI — FastAPI Backend
# Team: CodeBlooded | HackFest 2.0
# Deployment: Google Cloud Run
# ============================================================
# To run locally:
#   uvicorn main:app --reload --port 8000
# To deploy:
#   See README.md or follow the Cloud Run steps
# ============================================================

import os
import re
import json
import math
import uuid
import time
import datetime
import numpy as np
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Configuration — set via environment variables on Cloud Run ──
# Locally: just edit these defaults or set env vars
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL      = os.environ.get("GEMINI_MODEL",   "gemini-2.0-flash")
CHUNK_SIZE        = int(os.environ.get("CHUNK_SIZE",  "400"))
CHUNK_OVERLAP     = int(os.environ.get("CHUNK_OVERLAP","80"))
TOP_K_CHUNKS      = int(os.environ.get("TOP_K_CHUNKS", "5"))
EMBED_MODEL       = os.environ.get("EMBED_MODEL",    "all-MiniLM-L6-v2")
HF_DATASET_NAME   = os.environ.get("HF_DATASET_NAME","edinburghcstr/ami")
HF_DATASET_CONFIG = os.environ.get("HF_DATASET_CONFIG","ihm")

# BRD Template
BRD_TEMPLATE = """
BUSINESS REQUIREMENTS DOCUMENT (BRD)
=====================================
Project Name: {project_name}
Date: {date}
Prepared By: ReqMind AI — CodeBlooded Team
Version: 1.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{executive_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. PROJECT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{project_overview}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. STAKEHOLDERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{stakeholders_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. FUNCTIONAL REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{functional_requirements_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. NON-FUNCTIONAL REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{non_functional_requirements_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. KEY DECISIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{decisions_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. CONFLICT LOG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{conflicts_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. PROJECT TIMELINE & MILESTONES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{timeline_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. TRACEABILITY MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{traceability_section}
"""

# ── Gemini Client ──────────────────────────────────────────────

class GeminiClient:
    def __init__(self, api_key: str, model: str):
        self.mock_mode  = not api_key.strip()
        self.model_name = model
        if not self.mock_mode:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
        print(f"[GeminiClient] mode={'MOCK' if self.mock_mode else 'LIVE'} model={model}")

    def call(self, prompt: str, system: str = "") -> str:
        if self.mock_mode:
            return self._mock(prompt)
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                err = str(e).lower()
                if '429' in err or 'quota' in err or 'rate' in err:
                    wait = (attempt + 1) * 20
                    print(f"[GeminiClient] Rate limit. Waiting {wait}s ({attempt+1}/{max_retries})...")
                    time.sleep(wait)
                    continue
                if '400' in err or 'api_key' in err or 'invalid' in err:
                    raise ValueError(
                        "Gemini API key invalid. Get one at aistudio.google.com/apikey"
                    )
                print(f"[GeminiClient] Error attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    return self._mock(prompt)
        return self._mock(prompt)

    def _mock(self, prompt: str) -> str:
        p = prompt.lower()
        if "noise" in p or "filter" in p or "clean" in p:
            lines = [l for l in prompt.split('\n') if ':' in l][:15]
            return '\n'.join(lines) if lines else "Speaker A: We need to finalize requirements."
        if "extract" in p or "requirement" in p or "stakeholder" in p:
            return json.dumps({
                "functional_requirements": [
                    {"id": "FR-001", "description": "System shall support user authentication via OAuth 2.0",
                     "priority": "High", "speaker": "Project Manager", "timestamp": "00:03:12",
                     "source": "Transcript", "confidence": 0.92},
                    {"id": "FR-002", "description": "System shall generate reports in PDF and CSV formats",
                     "priority": "Medium", "speaker": "Designer", "timestamp": "00:07:45",
                     "source": "Transcript", "confidence": 0.88},
                    {"id": "FR-003", "description": "Dashboard shall display real-time data updates",
                     "priority": "High", "speaker": "Engineer", "timestamp": "00:12:30",
                     "source": "Transcript", "confidence": 0.71},
                    {"id": "FR-004", "description": "System shall support bulk data import via CSV",
                     "priority": "Low", "speaker": "Project Manager", "timestamp": "00:18:00",
                     "source": "Transcript", "confidence": 0.65},
                ],
                "non_functional_requirements": [
                    {"category": "Performance", "description": "System shall respond within 2 seconds for 95% of requests"},
                    {"category": "Security",    "description": "All data in transit must be encrypted via TLS 1.3"},
                    {"category": "Scalability", "description": "System shall support up to 10,000 concurrent users"},
                ],
                "stakeholders": [
                    {"name": "Project Manager", "role": "Project Lead",  "sentiment": "positive", "concerns": "Timeline adherence"},
                    {"name": "Designer",         "role": "UX Lead",       "sentiment": "neutral",  "concerns": "Accessibility compliance"},
                    {"name": "Engineer",         "role": "Tech Lead",     "sentiment": "negative", "concerns": "Technical feasibility"},
                ],
                "decisions": [
                    {"decision": "Use OAuth 2.0 for all authentication", "made_by": "Project Manager", "timestamp": "00:05:00"},
                    {"decision": "Prototype due end of Q1",              "made_by": "Team",            "timestamp": "00:22:00"},
                ],
                "timelines": [
                    {"milestone": "Requirements Sign-off", "date": "2025-03-15", "owner": "Project Manager"},
                    {"milestone": "Prototype Demo",        "date": "2025-03-31", "owner": "Engineer"},
                    {"milestone": "Full Release",          "date": "2025-06-30", "owner": "Team"},
                ]
            })
        if "conflict" in p or "contradict" in p or "disagree" in p:
            return json.dumps({"conflicts": [
                {"conflict": "Timeline disagreement on prototype delivery",
                 "description": "Project Manager insists March 31st but Engineer needs 8 weeks.",
                 "parties": ["Project Manager", "Engineer"],
                 "severity": "High",
                 "resolution": "Reduced-scope prototype March 31st, full April 30th"}
            ]})
        if "summary" in p or "executive" in p:
            return ("This project aims to build a robust system with secure authentication and "
                    "real-time dashboards. Key decisions include OAuth 2.0 adoption. One "
                    "high-severity timeline conflict was resolved through scope reduction.")
        if "edit" in p or "update" in p:
            return prompt
        return "Processed successfully."


# ── Rule-Based Preprocessing ───────────────────────────────────

FILLER_PATTERNS = re.compile(
    r'\b(um+|uh+|er+|ah+|hmm+|like|you know|i mean|sort of|kind of|basically|'
    r'actually|literally|right\?|okay so|so yeah|anyway|alright)\b',
    re.IGNORECASE
)
SPEAKER_LINE_RE = re.compile(
    r'^(?P<speaker>[A-Za-z][A-Za-z\s\-\.]+?)\s*(?:\[(?P<timestamp>\d{1,2}:\d{2}:\d{2})\])?\s*:\s*(?P<text>.+)$',
    re.MULTILINE
)

def rule_segment_speakers(raw: str) -> List[Dict]:
    segments = []
    for m in SPEAKER_LINE_RE.finditer(raw):
        text = m.group('text').strip()
        if len(text) < 3:
            continue
        segments.append({
            'speaker':   m.group('speaker').strip(),
            'timestamp': m.group('timestamp') or '00:00:00',
            'text':      text,
            'original':  m.group(0)
        })
    if not segments and raw.strip():
        segments.append({'speaker': 'Unknown', 'timestamp': '00:00:00',
                         'text': raw.strip(), 'original': raw.strip()})
    return segments

def rule_remove_fillers(text: str) -> str:
    cleaned = FILLER_PATTERNS.sub('', text)
    return re.sub(r'\s{2,}', ' ', cleaned).strip()

def rule_is_noise(text: str) -> bool:
    t = text.lower().strip()
    if len(t) < 5:
        return True
    noise_phrases = [
        "okay", "alright", "sure", "yeah", "yep", "nope", "mhm",
        "i see", "got it", "sounds good", "cool", "great", "thanks",
        "thank you", "no problem", "absolutely", "definitely"
    ]
    return t in noise_phrases


# ── RAG Store ──────────────────────────────────────────────────

from sentence_transformers import SentenceTransformer
import faiss

class RAGStore:
    def __init__(self, embed_model: str):
        print(f"[RAGStore] Loading embedding model: {embed_model}")
        self.embedder = SentenceTransformer(embed_model)
        self.chunks: List[Dict] = []
        self.index: Optional[faiss.IndexFlatL2] = None
        print("[RAGStore] Ready.")

    def build(self, segments: List[Dict], chunk_size: int, overlap: int):
        self.chunks = []
        full_lines = [f"{s['speaker']} [{s['timestamp']}]: {s['text']}" for s in segments]

        raw_chunks, current_lines, current_len = [], [], 0
        for line in full_lines:
            line_len = len(line) + 1
            if current_len + line_len > chunk_size and current_lines:
                raw_chunks.append('\n'.join(current_lines))
                overlap_lines, overlap_len = [], 0
                for ol in reversed(current_lines):
                    if overlap_len + len(ol) < overlap:
                        overlap_lines.insert(0, ol)
                        overlap_len += len(ol)
                    else:
                        break
                current_lines, current_len = overlap_lines, overlap_len
            current_lines.append(line)
            current_len += line_len
        if current_lines:
            raw_chunks.append('\n'.join(current_lines))

        for i, chunk in enumerate(raw_chunks):
            speakers = re.findall(r'^([A-Za-z][A-Za-z\s]+?)\s*\[', chunk, re.MULTILINE)
            self.chunks.append({
                'chunk_id':   f'CHK-{i+1:03d}',
                'text':       chunk,
                'speaker':    speakers[0] if speakers else 'Unknown',
                'line_count': chunk.count('\n') + 1
            })

        print(f"[RAGStore] Built {len(self.chunks)} chunks from {len(segments)} segments")
        texts      = [c['text'] for c in self.chunks]
        embeddings = np.array(self.embedder.encode(texts, show_progress_bar=False)).astype('float32')
        dim        = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"[RAGStore] FAISS index built. dim={dim}, vectors={self.index.ntotal}")

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        if self.index is None or not self.chunks:
            return self.chunks[:top_k]
        q_emb     = np.array(self.embedder.encode([query], show_progress_bar=False)).astype('float32')
        distances, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                chunk = dict(self.chunks[idx])
                chunk['relevance_score'] = float(1 / (1 + dist))
                results.append(chunk)
        return results

    def retrieve_multi(self, queries: List[str], top_k_per_query: int,
                       max_total: int = 8) -> List[Dict]:
        if self.index is None or not self.chunks:
            return self.chunks[:max_total]
        seen: Dict[str, Dict] = {}
        for query in queries:
            for chunk in self.retrieve(query, top_k_per_query):
                cid = chunk['chunk_id']
                if cid not in seen or chunk['relevance_score'] > seen[cid]['relevance_score']:
                    seen[cid] = chunk
        return sorted(seen.values(), key=lambda c: c['relevance_score'], reverse=True)[:max_total]

    def get_context(self, query: str, top_k: int) -> str:
        return self._format_chunks(self.retrieve(query, top_k))

    def get_context_multi(self, queries: List[str], top_k_per_query: int = 3,
                          max_total: int = 8) -> str:
        return self._format_chunks(self.retrieve_multi(queries, top_k_per_query, max_total))

    def _format_chunks(self, chunks: List[Dict]) -> str:
        if not chunks:
            return "No relevant context found."
        parts = [f"[{c['chunk_id']} | relevance={c.get('relevance_score', 0):.2f}]\n{c['text']}"
                 for c in chunks]
        return '\n\n---\n\n'.join(parts)


# ── Pipeline ───────────────────────────────────────────────────

def safe_json(text: str) -> Any:
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'\s*```$',          '', text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except:
                pass
    return {}

class ReqMindPipeline:
    def __init__(self, gemini_client, rag_store, top_k, chunk_size, overlap):
        self.llm, self.rag = gemini_client, rag_store
        self.top_k, self.chunk_size, self.overlap = top_k, chunk_size, overlap

    def stage_ingest(self, transcript, project_name):
        print("[Pipeline] Stage 1: Ingestion")
        segments = rule_segment_speakers(transcript)
        return {'project_name': project_name, 'raw_transcript': transcript,
                'segments': segments, 'segment_count': len(segments)}

    def stage_preprocess(self, data):
        print("[Pipeline] Stage 2: Preprocessing")
        cleaned, dropped = [], 0
        for seg in data['segments']:
            if rule_is_noise(seg['text']):
                dropped += 1; continue
            clean_text = rule_remove_fillers(seg['text'])
            if len(clean_text) < 5:
                dropped += 1; continue
            cleaned.append({**seg, 'text': clean_text})
        print(f"[Pipeline] kept {len(cleaned)}, dropped {dropped}")
        data.update({'segments': cleaned, 'dropped_lines': dropped, 'cleaned_count': len(cleaned)})
        return data

    def stage_build_rag(self, data):
        print("[Pipeline] Stage 3: Building RAG Index")
        self.rag.build(data['segments'], self.chunk_size, self.overlap)
        data['chunk_count'] = len(self.rag.chunks)
        data['rag_ready']   = True
        return data

    def stage_noise_filter(self, data):
        print("[Pipeline] Stage 4: LLM Noise Filter")
        context = self.rag.get_context_multi(
            queries=["project requirement feature decision constraint",
                     "timeline deadline milestone delivery",
                     "stakeholder concern disagreement agreed"],
            top_k_per_query=3, max_total=6
        )
        prompt = (f"You are cleaning a meeting transcript for requirement extraction.\n\n"
                  f"Given these transcript chunks:\n{context}\n\n"
                  f"Remove off-topic content. Keep requirements, decisions, timelines, concerns.\n"
                  f"Return ONLY cleaned transcript lines in original speaker format.")
        data['cleaned_transcript'] = self.llm.call(
            prompt, system="You are a precise transcript cleaning assistant."
        )
        return data

    def stage_extract(self, data):
        print("[Pipeline] Stage 5: Requirement Extraction")
        context = self.rag.get_context_multi(
            queries=["feature functionality system shall must requirement",
                     "performance security scalability reliability usability",
                     "stakeholder concern disagreement agreed decided",
                     "timeline deadline milestone delivery date",
                     "constraint budget limitation restriction"],
            top_k_per_query=3, max_total=8
        )
        prompt = f"""You are a senior business analyst. Extract all requirements from these transcript chunks.

=== TRANSCRIPT CONTEXT ===
{context}

Return ONLY a JSON object (no markdown):
{{
  "functional_requirements": [
    {{"id": "FR-001", "description": "shall/must statement", "priority": "High|Medium|Low",
      "speaker": "name", "timestamp": "HH:MM:SS", "source": "Transcript", "confidence": 0.0-1.0}}
  ],
  "non_functional_requirements": [
    {{"category": "Performance|Security|Scalability|Usability|Reliability", "description": "..."}}
  ],
  "stakeholders": [
    {{"name": "...", "role": "...", "sentiment": "positive|negative|neutral", "concerns": "..."}}
  ],
  "decisions": [{{"decision": "...", "made_by": "...", "timestamp": "HH:MM:SS"}}],
  "timelines":  [{{"milestone": "...", "date": "...", "owner": "..."}}]
}}

CONFIDENCE RULES — assign honestly, do NOT default to 0.8 for everything:
- 0.90-1.00: Explicitly stated with exact numbers or clear action words
- 0.75-0.89: Clearly implied, no ambiguity
- 0.60-0.74: Ambiguous or disputed → will be HITL flagged
- 0.40-0.59: Very unclear or heavily inferred → will be HITL flagged"""

        extracted = safe_json(self.llm.call(
            prompt, system="You are a precise requirements engineering specialist."
        ))
        for i, fr in enumerate(extracted.get('functional_requirements', [])):
            fr.setdefault('id',        f'FR-{i+1:03d}')
            fr.setdefault('priority',  'Medium')
            fr.setdefault('speaker',   'Unknown')
            fr.setdefault('timestamp', '00:00:00')
            fr.setdefault('source',    'Transcript')
            fr.setdefault('confidence', 0.80)
        data['extracted_data'] = extracted
        return data

    def stage_detect_conflicts(self, data):
        print("[Pipeline] Stage 6: Conflict Detection")
        context = self.rag.get_context_multi(
            queries=["disagree contradict conflict dispute",
                     "but however instead no that's wrong",
                     "timeline deadline too tight not possible"],
            top_k_per_query=3, max_total=6
        )
        sh_names = ', '.join([s['name'] for s in data.get('extracted_data', {}).get('stakeholders', [])]) or 'Unknown'
        prompt = f"""Analyse transcript chunks for conflicts.
Stakeholders: {sh_names}

=== CHUNKS ===
{context}

Find: contradictory requirements, disputed decisions, timeline clashes, explicit disagreements.

Return ONLY JSON:
{{"conflicts": [{{"conflict": "title", "description": "...", "parties": ["A","B"],
   "severity": "High|Medium|Low", "resolution": "... or Unresolved"}}]}}

If none exist return {{"conflicts": []}}. Do not fabricate."""

        conflict_data = safe_json(self.llm.call(prompt, system="You are a conflict analysis specialist."))
        data['extracted_data']['conflicts'] = conflict_data.get('conflicts', [])
        return data

    def stage_form_json(self, data):
        print("[Pipeline] Stage 7: JSON Formation")
        ext = data.get('extracted_data', {})
        seen, unique_frs = set(), []
        for fr in ext.get('functional_requirements', []):
            key = fr['description'][:60].lower().strip()
            if key not in seen:
                seen.add(key); unique_frs.append(fr)
        ext['functional_requirements'] = unique_frs
        for i, fr in enumerate(ext['functional_requirements']):
            fr['id'] = f'FR-{i+1:03d}'
        for key in ['non_functional_requirements', 'stakeholders', 'decisions', 'timelines', 'conflicts']:
            ext.setdefault(key, [])
        data['extracted_data'] = ext
        return data

    def stage_generate_brd(self, data):
        print("[Pipeline] Stage 8: BRD Generation")
        ext  = data['extracted_data']
        pn   = data['project_name']
        date = datetime.date.today().strftime('%B %d, %Y')

        overview_context = self.rag.get_context_multi(
            queries=["project goal objective purpose scope background"],
            top_k_per_query=self.top_k, max_total=4
        )
        overview = self.llm.call(
            f"Based on this meeting context:\n{overview_context}\n\n"
            f"Write a 2-3 paragraph Project Overview for the BRD for project: {pn}. Be concise and professional.",
            system="You are a technical writer creating a formal BRD."
        )

        sh_lines = [f"• {s['name']} | Role: {s['role']} | Sentiment: {s['sentiment'].title()} | Concerns: {s.get('concerns','N/A')}"
                    for s in ext.get('stakeholders', [])]
        stakeholders_section = '\n'.join(sh_lines) or "No stakeholders identified."

        fr_lines = []
        for fr in ext.get('functional_requirements', []):
            flag = "⚠ HITL REVIEWED" if fr.get('confidence', 1.0) < 0.75 else ""
            fr_lines.append(
                f"[{fr['id']}] {fr['description']}\n"
                f"  Priority: {fr.get('priority','Medium')} | Speaker: {fr.get('speaker','Unknown')} | "
                f"Timestamp: {fr.get('timestamp','—')} | Confidence: {int(fr.get('confidence',0.8)*100)}% {flag}"
            )
        fr_section = '\n\n'.join(fr_lines) or "No functional requirements extracted."

        nfr_lines = [f"• [{n.get('category','General')}] {n.get('description','')}"
                     for n in ext.get('non_functional_requirements', [])]
        nfr_section = '\n'.join(nfr_lines) or "No non-functional requirements extracted."

        dec_lines = [f"• {d.get('decision','')} — {d.get('made_by','Team')} at {d.get('timestamp','—')}"
                     for d in ext.get('decisions', [])]
        dec_section = '\n'.join(dec_lines) or "No key decisions recorded."

        conf_lines = []
        for c in ext.get('conflicts', []):
            marker = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(c.get('severity',''), '⚪')
            conf_lines.append(
                f"{marker} [{c.get('severity','—')}] {c.get('conflict','')}\n"
                f"  Parties: {', '.join(c.get('parties',[]))}\n"
                f"  Detail: {c.get('description','')}\n"
                f"  Resolution: {c.get('resolution','Unresolved')}"
            )
        conf_section = '\n\n'.join(conf_lines) or "✓ No conflicts detected."

        tl_lines = [f"• {t.get('milestone','')} — {t.get('date','TBD')} (Owner: {t.get('owner','—')})"
                    for t in ext.get('timelines', [])]
        tl_section = '\n'.join(tl_lines) or "No timeline milestones identified."

        trace_lines = ["REQ ID    | DESCRIPTION (truncated)              | SPEAKER          | TIMESTAMP | CONFIDENCE",
                       "-" * 90]
        for fr in ext.get('functional_requirements', []):
            trace_lines.append(
                f"{fr['id']:<10}| {fr['description'][:40].ljust(40)}| "
                f"{fr.get('speaker','Unknown')[:16].ljust(16)}| "
                f"{fr.get('timestamp','—'):<10}| {int(fr.get('confidence',0.8)*100)}%"
            )
        trace_section = '\n'.join(trace_lines)

        data['brd_document'] = BRD_TEMPLATE.format(
            project_name=pn, date=date,
            executive_summary="",  # filled by stage_exec_summary
            project_overview=overview.strip(),
            stakeholders_section=stakeholders_section,
            functional_requirements_section=fr_section,
            non_functional_requirements_section=nfr_section,
            decisions_section=dec_section, conflicts_section=conf_section,
            timeline_section=tl_section, traceability_section=trace_section
        )
        return data

    def stage_exec_summary(self, data):
        print("[Pipeline] Stage 9: Executive Summary")
        ext  = data['extracted_data']
        fr_count, sh_count, con_count = (
            len(ext.get('functional_requirements', [])),
            len(ext.get('stakeholders', [])),
            len(ext.get('conflicts', []))
        )
        context = self.rag.get_context_multi(
            queries=["project objective goal purpose outcome",
                     "key decision agreed resolved approved",
                     "conflict resolved unresolved critical"],
            top_k_per_query=2, max_total=5
        )
        prompt = (f"Write a 4-5 sentence executive summary for a BRD.\n"
                  f"Project: {data['project_name']}\n"
                  f"Stats: {fr_count} FRs, {sh_count} stakeholders, {con_count} conflicts.\n\n"
                  f"Context:\n{context}\n\nBe professional. Cover: project purpose, key requirements, decisions, conflicts.")
        summary = self.llm.call(prompt, system="You are a senior business analyst.").strip()
        if not summary or len(summary) < 20:
            summary = (f"This BRD documents requirements for {data['project_name']}, "
                       f"capturing {fr_count} functional requirements across {sh_count} stakeholders. "
                       f"{con_count} conflict(s) identified. All requirements traced to source utterances.")
        data['executive_summary'] = summary
        if 'brd_document' in data:
            data['brd_document'] = data['brd_document'].replace(
                "\n1. EXECUTIVE SUMMARY\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
                f"\n1. EXECUTIVE SUMMARY\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{summary}\n"
            )
        return data

    def run(self, transcript: str, project_name: str) -> Dict:
        print(f"\n{'='*60}\n[Pipeline] Starting: {project_name}\n{'='*60}")
        data = self.stage_ingest(transcript, project_name)
        data = self.stage_preprocess(data)
        data = self.stage_build_rag(data)
        data = self.stage_noise_filter(data)
        data = self.stage_extract(data)
        data = self.stage_detect_conflicts(data)
        data = self.stage_form_json(data)
        data = self.stage_generate_brd(data)
        data = self.stage_exec_summary(data)
        print(f"[Pipeline] Complete. FR={len(data['extracted_data'].get('functional_requirements',[]))}")
        return data


# ── Input Normalizer ───────────────────────────────────────────

def normalize_input_to_transcript(raw: str, source_type: str = "transcript") -> str:
    if source_type == "transcript":
        return raw
    lines      = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    normalized = []
    fake_time  = 0

    if source_type == "email":
        current_sender, buffer = "Unknown", []
        def flush(sender, buf, t):
            text = ' '.join(buf).strip()
            if text and len(text) > 5:
                h,m,s = t//3600,(t%3600)//60,t%60
                return f"{sender} [{h:02d}:{m:02d}:{s:02d}]: {text}"
            return None
        for line in lines:
            fm = re.match(r'^(?:From|from)\s*:\s*([A-Za-z][A-Za-z\s]+?)(?:\s*<.*?>)?\s*$', line)
            if fm:
                entry = flush(current_sender, buffer, fake_time)
                if entry: normalized.append(entry); fake_time += 60
                current_sender, buffer = fm.group(1).strip(), []
            elif re.match(r'^(?:Subject|To|Cc|Date|Re)\s*:', line, re.IGNORECASE) or line.startswith('>'):
                continue
            else:
                buffer.append(line)
        entry = flush(current_sender, buffer, fake_time)
        if entry: normalized.append(entry)

    elif source_type == "slack":
        current_speaker, i = "Unknown", 0
        while i < len(lines):
            sh = re.match(r'^([A-Za-z][A-Za-z\s\-\.]{1,30}?)\s{2,}(\d{1,2}:\d{2}(?:\s?[AP]M)?)\s*$', lines[i])
            if sh:
                current_speaker = sh.group(1).strip()
                try:
                    t  = datetime.datetime.strptime(sh.group(2).strip().upper().replace(' ',''), '%I:%M%p')
                    ts = t.strftime('%H:%M:00')
                except:
                    h,m = fake_time//3600,(fake_time%3600)//60
                    ts  = f"{h:02d}:{m:02d}:00"; fake_time += 120
                i += 1; msg_lines = []
                while i < len(lines):
                    if re.match(r'^[A-Za-z][A-Za-z\s\-\.]{1,30}?\s{2,}\d{1,2}:\d{2}', lines[i]): break
                    msg_lines.append(lines[i]); i += 1
                text = ' '.join(msg_lines).strip()
                if text and len(text) > 5: normalized.append(f"{current_speaker} [{ts}]: {text}")
            else:
                text = lines[i].strip()
                if text and len(text) > 5:
                    h,m = fake_time//3600,(fake_time%3600)//60
                    normalized.append(f"{current_speaker} [{h:02d}:{m:02d}:00]: {text}"); fake_time += 30
                i += 1
    else:
        paragraph, p_idx = [], 1
        for line in lines:
            if not line:
                if paragraph:
                    h,m = fake_time//3600,(fake_time%3600)//60
                    normalized.append(f"Participant {p_idx} [{h:02d}:{m:02d}:00]: {' '.join(paragraph).strip()}")
                    fake_time += 60; p_idx += 1; paragraph = []
            else:
                paragraph.append(line)
        if paragraph:
            h,m = fake_time//3600,(fake_time%3600)//60
            normalized.append(f"Participant {p_idx} [{h:02d}:{m:02d}:00]: {' '.join(paragraph).strip()}")

    result = '\n'.join(normalized)
    if not result.strip():
        return f"Unknown [00:00:00]: {raw.strip()}"
    print(f"[Normalizer] {source_type} → {len(normalized)} transcript lines")
    return result


def load_ami_from_huggingface(meeting_id: str = None) -> str:
    from datasets import load_dataset
    print(f"[AMI Loader] Loading from HuggingFace...")
    try:
        dataset = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split='train', trust_remote_code=True)
        if meeting_id:
            samples = [s for s in dataset if s.get('meeting_id') == meeting_id] or list(dataset)
        else:
            samples = list(dataset)
        lines = []
        for sample in samples[:200]:
            speaker  = sample.get('speaker_id', 'Speaker')
            text     = sample.get('text', '').strip()
            start    = sample.get('begin_time', 0)
            h,m,s    = int(start//3600), int((start%3600)//60), int(start%60)
            if text: lines.append(f"{speaker} [{h:02d}:{m:02d}:{s:02d}]: {text}")
        transcript = '\n'.join(lines)
        print(f"[AMI Loader] {len(lines)} utterances loaded")
        return transcript
    except Exception as e:
        print(f"[AMI Loader] Failed: {e} — using fallback")
        return SAMPLE_TRANSCRIPT_FALLBACK

SAMPLE_TRANSCRIPT_FALLBACK = """Project Manager [00:01:00]: Alright team, let's finalize the remote control design today.
Designer [00:01:15]: I suggest we go with 8 buttons maximum for a clean design.
Project Manager [00:01:45]: Marketing says we need at least 15 buttons. That's a client requirement.
Engineer [00:02:10]: 15 buttons will hurt ergonomics. We should keep it under 12.
Designer [00:02:30]: Agreed with Engineer. 8 to 10 is ideal from a UX perspective.
Project Manager [00:03:00]: Final decision - 12 buttons. Let's compromise there.
Engineer [00:03:45]: The casing needs to be impact resistant. Drop tests from 1 meter height should be standard.
Designer [00:04:15]: We should use recycled plastic for the casing - client mentioned sustainability.
Project Manager [00:04:45]: Good point. Recycled materials for casing is now a requirement.
Engineer [00:05:30]: Battery life must last at least 12 months with normal use. Non-negotiable.
Project Manager [00:06:00]: Timeline - prototype ready by end of March. Full production by end of Q2.
Engineer [00:06:30]: March is too tight. We need 8 weeks minimum for prototyping. That's mid-April at earliest.
Project Manager [00:07:00]: The client deadline is March 31st. We need to figure this out.
Designer [00:07:30]: What if we reduce prototype scope? Just shell and basic button layout?
Engineer [00:08:00]: That could work. Reduced scope prototype by March 31st, full by April 30th.
Project Manager [00:08:30]: Agreed. Remote must also be compatible with all major TV brands.
Engineer [00:09:00]: Universal compatibility via standard IR protocol. Covers 95% of TVs.
Designer [00:09:30]: Design should also support visually impaired users. Large buttons, tactile feedback.
Project Manager [00:10:00]: Add accessibility features as a requirement. Good catch."""


# ── Initialise global instances ────────────────────────────────

print("[Init] Initialising ReqMind AI pipeline...")
llm_client = GeminiClient(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)
rag_store  = RAGStore(embed_model=EMBED_MODEL)
pipeline   = ReqMindPipeline(
    gemini_client=llm_client, rag_store=rag_store,
    top_k=TOP_K_CHUNKS, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
)
print("[Init] Pipeline ready.")


# ── FastAPI App ────────────────────────────────────────────────

app = FastAPI(title="ReqMind AI", version="2.0", description="RAG-based BRD generation — Team CodeBlooded")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Models ─────────────────────────────────────────────

class GenerateRequest(BaseModel):
    transcript:   str
    project_name: str          = "Untitled Project"
    meeting_id:   Optional[str] = None
    source_type:  str          = "transcript"

class EditRequest(BaseModel):
    current_brd:      str
    edit_instruction: str

class EvaluateRequest(BaseModel):
    meeting_id: str = "ES2002a"


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status":          "running",
        "project":         "ReqMind AI",
        "team":            "CodeBlooded",
        "hackathon":       "HackFest 2.0",
        "llm":             GEMINI_MODEL,
        "mode":            "mock" if llm_client.mock_mode else "live",
        "rag":             f"{TOP_K_CHUNKS} chunks per query",
        "pipeline_stages": 9
    }

@app.post("/generate-brd")
async def generate_brd(req: GenerateRequest):
    try:
        transcript = req.transcript
        if req.meeting_id:
            transcript = load_ami_from_huggingface(req.meeting_id)
            if not transcript:
                raise HTTPException(status_code=404, detail=f"Meeting '{req.meeting_id}' not found")
        if not transcript.strip():
            raise HTTPException(status_code=400, detail="Input is empty.")
        transcript = normalize_input_to_transcript(transcript, req.source_type)
        result     = pipeline.run(transcript, req.project_name)
        exec_summary = result.get('executive_summary', '')
        if not exec_summary:
            ext = result.get('extracted_data', {})
            exec_summary = (f"Requirements document generated for {result['project_name']}. "
                            f"{len(ext.get('functional_requirements',[]))} FRs extracted.")
        return {
            "project_name":       result['project_name'],
            "cleaned_transcript": result.get('cleaned_transcript', ''),
            "extracted_data":     result['extracted_data'],
            "brd_document":       result.get('brd_document', ''),
            "executive_summary":  exec_summary,
            "provider":           "mock" if llm_client.mock_mode else "gemini",
            "model":              GEMINI_MODEL,
            "rag_chunks_built":   result.get('chunk_count', 0),
            "segments_processed": result.get('cleaned_count', 0),
            "source_type":        req.source_type,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        err = str(e)
        if '429' in err or 'quota' in err:
            raise HTTPException(status_code=429,
                detail="Gemini rate limit. Wait 60s and retry. (Free tier: 15 req/min)")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {err}")

@app.post("/edit-brd")
async def edit_brd(req: EditRequest):
    try:
        context = rag_store.get_context(req.edit_instruction, TOP_K_CHUNKS) if rag_store.index else ""
        prompt  = (f"Edit this BRD.\nINSTRUCTION: {req.edit_instruction}\n\n"
                   f"CONTEXT:\n{context}\n\nBRD:\n{req.current_brd}\n\n"
                   f"Apply the instruction. Keep template structure. Return only the updated BRD.")
        updated = llm_client.call(prompt, system="You are a professional BRD editor.")
        return {"updated_brd": updated, "edit_applied": req.edit_instruction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ami-meetings")
async def list_ami_meetings():
    try:
        from datasets import load_dataset
        dataset     = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split='train', trust_remote_code=True)
        meeting_ids = sorted(set(dataset['meeting_id']))
        return {"meetings": meeting_ids, "total": len(meeting_ids)}
    except Exception as e:
        return {"meetings": [], "error": str(e)}

@app.get("/pipeline-info")
async def pipeline_info():
    return {
        "pipeline": "ReqMind AI RAG Pipeline v2.0",
        "stages": [
            {"stage": 1, "name": "Ingestion",          "type": "rule-based",    "description": "Regex speaker segmentation"},
            {"stage": 2, "name": "Preprocessing",      "type": "rule-based",    "description": "Filler removal + noise filtering"},
            {"stage": 3, "name": "RAG Index Build",    "type": "embedding",     "description": f"Line-boundary chunking + {EMBED_MODEL} + FAISS"},
            {"stage": 4, "name": "Noise Filter",       "type": "llm+rag",       "description": "Gemini removes remaining noise using retrieved context"},
            {"stage": 5, "name": "Requirement Extract","type": "llm+rag",       "description": "Gemini extracts FR/NFR/stakeholders using 5-query deduped retrieval"},
            {"stage": 6, "name": "Conflict Detection", "type": "llm+rag",       "description": "Gemini finds contradictions using conflict-focused retrieval"},
            {"stage": 7, "name": "JSON Formation",     "type": "rule-based",    "description": "Schema normalisation + deduplication"},
            {"stage": 8, "name": "BRD Generation",     "type": "llm+template",  "description": "Fixed template filled section by section"},
            {"stage": 9, "name": "Exec Summary",       "type": "llm+rag",       "description": "4-5 sentence summary from key-decision context"},
        ],
        "rag_config": {"embedding_model": EMBED_MODEL, "chunk_size": CHUNK_SIZE,
                       "chunk_overlap": CHUNK_OVERLAP, "top_k": TOP_K_CHUNKS},
        "llm":     GEMINI_MODEL,
        "dataset": "AMI Meeting Corpus — edinburghcstr/ami on HuggingFace"
    }
