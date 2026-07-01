import React, { useState, useEffect, useRef } from "react";

const COLORS: Record<string, string> = {
  bg: "#222225",
  bgAlt: "#1c1c1f",
  card: "#2a2a2f",
  cardBorder: "#3a3a40",
  text: "#e8e9ed",
  textDim: "#9aa0a8",
  textFaint: "#5a5e66",
  primary: "#62c4ff",
  success: "#7fd962",
  warn: "#ffb86c",
  danger: "#ff6b6b",
  wcTeal: "#4a8a9a",
  wcPurple: "#9a7ab0",
  wcGold: "#d4a04a",
};

interface Word { word: string; start: number; end: number; }
interface Boundary { ts: number; concept: number; step: number; }
interface Composition { id: string; }

const CONCEPT_COLORS = ["#62c4ff", "#ffb86c", "#7fd962", "#9a7ab0", "#d4a04a", "#ff6b6b", "#4a8a9a", "#6a9ab0"];

export default function App() {
  const [compositions, setCompositions] = useState<Composition[]>([]);
  const [selected, setSelected] = useState("");
  const [words, setWords] = useState<Word[]>([]);
  const [captions, setCaptions] = useState<any[]>([]);
  const [boundaries, setBoundaries] = useState<Boundary[]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [wsLog, setWsLog] = useState<string[]>([]);
  const [fixing, setFixing] = useState(false);
  const [videoKey, setVideoKey] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Resizable columns
  const [leftWidth, setLeftWidth] = useState(600);
  const draggingRef = useRef(false);
  const [agentOpen, setAgentOpen] = useState(true);
  const [agentPos, setAgentPos] = useState({ x: 100, y: 100 });
  const agentDragRef = useRef<{ startX: number; startY: number; origX: number; origY: number } | null>(null);

  // Load compositions
  useEffect(() => {
    fetch("/api/compositions").then(r => r.json()).then(setCompositions);
  }, []);

  // Load data when composition changes
  useEffect(() => {
    if (!selected) return;
    Promise.all([
      fetch(`/api/words/${selected}`).then(r => r.json()),
      fetch(`/api/captions/${selected}`).then(r => r.json()),
      fetch(`/api/boundaries/${selected}`).then(r => r.json()),
    ]).then(([w, c, b]) => {
      setWords(w);
      setCaptions(c);
      setBoundaries(b);
    });
  }, [selected]);

  // Auto-select first composition
  useEffect(() => {
    if (compositions.length > 0 && !selected) setSelected(compositions[0].id);
  }, [compositions]);

  // WebSocket for live agent feedback
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:4097");
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      if (data.type === "log") {
        setWsLog(prev => [...prev, data.text]);
      } else if (data.type === "done") {
        setFixing(false);
        setWsLog(prev => [...prev, `\n=== Agent ${data.status} ===\n`]);
        if (data.status === "done") {
          // Reload video by changing key
          setTimeout(() => {
            setVideoKey(k => k + 1);
            setWsLog(prev => [...prev, "Video reloaded.\n"]);
          }, 1000);
        }
      }
    };
    return () => ws.close();
  }, []);

  // Video time sync
  const onTimeUpdate = () => {
    if (videoRef.current) setCurrentTime(videoRef.current.currentTime);
  };

  const onLoadedMetadata = () => {
    if (videoRef.current) setDuration(videoRef.current.duration);
  };

  // Seek to a word
  const seekTo = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  // Submit fix to agent
  const submitFix = (feedback: string) => {
    if (!selected || !feedback.trim()) return;
    setFixing(true);
    setWsLog([`Sending to agent: "${feedback}"\n`]);
    fetch("/api/fix", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ compositionId: selected, feedback }),
    }).then(r => r.json()).then(data => {
      setWsLog(prev => [...prev, `Job ${data.jobId} started...\n`]);
    });
  };

  // Find current word index
  const currentWordIdx = words.findIndex((w, i) =>
    currentTime >= w.start && currentTime < (words[i + 1]?.start ?? w.end)
  );

  // Find current concept
  const currentBoundary = [...boundaries].reverse().find(b => currentTime >= b.ts);

  // Resize handlers
  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (draggingRef.current) {
        setLeftWidth(Math.max(300, Math.min(window.innerWidth - 300, e.clientX)));
      }
    };
    const onMouseUp = () => { draggingRef.current = false; document.body.style.cursor = ""; document.body.style.userSelect = ""; };
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => { window.removeEventListener("mousemove", onMouseMove); window.removeEventListener("mouseup", onMouseUp); };
  }, []);

  const startDrag = () => (e: React.MouseEvent) => {
    e.preventDefault();
    draggingRef.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  };

  // Agent panel drag
  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (agentDragRef.current) {
        const dx = e.clientX - agentDragRef.current.startX;
        const dy = e.clientY - agentDragRef.current.startY;
        setAgentPos({
          x: Math.max(0, Math.min(window.innerWidth - 100, agentDragRef.current.origX + dx)),
          y: Math.max(0, Math.min(window.innerHeight - 40, agentDragRef.current.origY + dy)),
        });
      }
    };
    const onMouseUp = () => { agentDragRef.current = null; };
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => { window.removeEventListener("mousemove", onMouseMove); window.removeEventListener("mouseup", onMouseUp); };
  }, []);

  const startAgentDrag = (e: React.MouseEvent) => {
    e.preventDefault();
    agentDragRef.current = { startX: e.clientX, startY: e.clientY, origX: agentPos.x, origY: agentPos.y };
  };

  return (
    <div style={{ display: "flex", height: "100vh", background: COLORS.cardBorder, position: "relative" }}>
      {/* LEFT: Transcript (Descript-style main panel) */}
      <div style={{ flex: `0 0 ${leftWidth}px`, background: COLORS.bgAlt, display: "flex", flexDirection: "column", minWidth: 300, overflow: "hidden" }}>
        {/* Header with composition selector */}
        <div style={{ padding: "8px 12px", borderBottom: `1px solid ${COLORS.cardBorder}`, display: "flex", alignItems: "center", gap: 8 }}>
          <select
            value={selected}
            onChange={e => setSelected(e.target.value)}
            style={{
              background: COLORS.card, color: COLORS.text, border: `1px solid ${COLORS.cardBorder}`,
              borderRadius: 4, padding: "4px 8px", fontFamily: "inherit", fontSize: 13,
            }}
          >
            {compositions.map(c => <option key={c.id} value={c.id}>{c.id}</option>)}
          </select>
          {currentBoundary && (
            <span style={{ fontSize: 11, color: COLORS.textDim }}>
              <span style={{ color: CONCEPT_COLORS[currentBoundary.concept % CONCEPT_COLORS.length] }}>
                {"\u25CF"} C{currentBoundary.concept}
              </span>
              {" / S"}{currentBoundary.step}
              {" / "}<span style={{ color: COLORS.textFaint }}>{currentTime.toFixed(1)}s</span>
            </span>
          )}
        </div>
        {/* Transcript content */}
        <div style={{ flex: 1, overflow: "auto", padding: "16px 20px" }}>
          <Transcript words={words} currentIdx={currentWordIdx} onSeek={seekTo} />
        </div>
      </div>

      {/* Divider */}
      <div
        onMouseDown={startDrag()}
        style={{ width: 4, background: COLORS.cardBorder, cursor: "col-resize", flexShrink: 0 }}
      />

      {/* RIGHT: Video player */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", background: COLORS.bgAlt, minWidth: 300 }}>
        {/* Video */}
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 }}>
          <video
            key={videoKey}
            ref={videoRef}
            src={`/api/video/${selected}`}
            onTimeUpdate={onTimeUpdate}
            onLoadedMetadata={onLoadedMetadata}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            controls
            style={{ maxWidth: "100%", maxHeight: "100%", borderRadius: 8 }}
          />
        </div>
        {/* Step boundary timeline */}
        <StepTimeline boundaries={boundaries} duration={duration} currentTime={currentTime} onSeek={seekTo} />
      </div>

      {/* AI AGENT OVERLAY (draggable) */}
      {agentOpen && (
        <div style={{
          position: "absolute", left: agentPos.x, top: agentPos.y, width: 380, height: "50vh",
          background: "rgba(34, 34, 37, 0.96)", backdropFilter: "blur(12px)",
          border: `1px solid ${COLORS.cardBorder}`, borderRadius: 12,
          display: "flex", flexDirection: "column", overflow: "hidden",
          boxShadow: "0 8px 32px rgba(0,0,0,0.4)", zIndex: 100,
        }}>
          {/* Header (drag handle) */}
          <div
            onMouseDown={startAgentDrag}
            style={{
              padding: "8px 12px", borderBottom: `1px solid ${COLORS.cardBorder}`,
              display: "flex", justifyContent: "space-between", alignItems: "center",
              fontSize: 11, fontWeight: 700, color: COLORS.textDim, textTransform: "uppercase", letterSpacing: 1,
              cursor: "move", userSelect: "none",
            }}
          >
            <span>{"\u2728"} AI Agent</span>
            <button onClick={() => setAgentOpen(false)} style={{ background: "none", border: "none", color: COLORS.textFaint, cursor: "pointer", fontSize: 16, lineHeight: 1 }}>
              {"\u2715"}
            </button>
          </div>
          <AgentPanel
            fixing={fixing}
            log={wsLog}
            onSubmit={submitFix}
            currentConcept={currentBoundary?.concept}
            currentTime={currentTime}
          />
        </div>
      )}

      {/* Floating reopen button when agent is closed */}
      {!agentOpen && (
        <button
          onClick={() => setAgentOpen(true)}
          style={{
            position: "absolute", bottom: 16, right: 16,
            background: COLORS.primary, color: COLORS.bg, border: "none",
            borderRadius: 24, padding: "10px 20px", fontFamily: "inherit", fontSize: 12,
            fontWeight: 700, cursor: "pointer", zIndex: 100,
            boxShadow: "0 4px 16px rgba(98, 196, 255, 0.3)",
          }}
        >
          {"\u2728"} AI Agent
        </button>
      )}
    </div>
  );
}

// --- Step Timeline component ---
function StepTimeline({ boundaries, duration, currentTime, onSeek }: {
  boundaries: Boundary[];
  duration: number;
  currentTime: number;
  onSeek: (t: number) => void;
}) {
  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    onSeek(pct * duration);
  };

  return (
    <div style={{ padding: "8px 12px", borderTop: `1px solid ${COLORS.cardBorder}` }}>
      {/* Timeline bar */}
      <div
        onClick={handleClick}
        style={{ position: "relative", height: 24, background: COLORS.card, borderRadius: 4, cursor: "pointer", overflow: "hidden" }}
      >
        {/* Concept color segments */}
        {boundaries.map((b, i) => {
          const next = boundaries[i + 1];
          const left = (b.ts / (duration || 1)) * 100;
          const width = ((next ? next.ts : duration) - b.ts) / (duration || 1) * 100;
          return (
            <div key={i} style={{
              position: "absolute", left: `${left}%`, width: `${width}%`, height: "100%",
              background: CONCEPT_COLORS[b.concept % CONCEPT_COLORS.length], opacity: 0.15,
              borderLeft: `1px solid ${CONCEPT_COLORS[b.concept % CONCEPT_COLORS.length]}`
            }} />
          );
        })}
        {/* Step tick marks */}
        {boundaries.map((b, i) => (
          <div key={i} style={{
            position: "absolute", left: `${(b.ts / (duration || 1)) * 100}%`, top: 0, bottom: 0,
            width: 1, background: CONCEPT_COLORS[b.concept % CONCEPT_COLORS.length], opacity: 0.4,
          }} />
        ))}
        {/* Playhead */}
        <div style={{
          position: "absolute", left: `${(currentTime / (duration || 1)) * 100}%`, top: 0, bottom: 0,
          width: 2, background: COLORS.primary,
        }} />
      </div>
      {/* Labels */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 8px", marginTop: 4, fontSize: 9, color: COLORS.textFaint }}>
        {boundaries.filter((b, i) => i === 0 || b.concept !== boundaries[i - 1].concept).map((b, i) => (
          <span key={i} style={{ color: CONCEPT_COLORS[b.concept % CONCEPT_COLORS.length], cursor: "pointer" }}
            onClick={() => onSeek(b.ts)}>
            C{b.concept}
          </span>
        ))}
      </div>
    </div>
  );
}

// --- Transcript component ---
function Transcript({ words, currentIdx, onSeek }: {
  words: Word[];
  currentIdx: number;
  onSeek: (t: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const currentRef = useRef<HTMLSpanElement>(null);

  // Auto-scroll to current word
  useEffect(() => {
    if (currentRef.current && containerRef.current) {
      const container = containerRef.current;
      const el = currentRef.current;
      const containerRect = container.getBoundingClientRect();
      const elRect = el.getBoundingClientRect();
      if (elRect.top < containerRect.top || elRect.bottom > containerRect.bottom) {
        container.scrollTop += (elRect.top - containerRect.top) - containerRect.height / 3;
      }
    }
  }, [currentIdx]);

  // Group words into sentences (split on period)
  const sentences: { text: string; start: number; wordIndices: number[] }[] = [];
  let current: { text: string; start: number; wordIndices: number[] } = { text: "", start: 0, wordIndices: [] };
  words.forEach((w, i) => {
    if (current.text === "") current.start = w.start;
    current.text += w.word + " ";
    current.wordIndices.push(i);
    if (w.word.endsWith(".") || w.word.endsWith("?") || w.word.endsWith("!")) {
      sentences.push(current);
      current = { text: "", start: 0, wordIndices: [] };
    }
  });
  if (current.text) sentences.push(current);

  return (
    <div ref={containerRef} style={{ lineHeight: 1.8, fontSize: 12 }}>
      {sentences.map((s, si) => {
        const isActive = s.wordIndices.includes(currentIdx);
        return (
          <span
            key={si}
            onClick={() => onSeek(s.start)}
            style={{
              cursor: "pointer",
              color: isActive ? COLORS.primary : COLORS.textDim,
              background: isActive ? "rgba(98, 196, 255, 0.1)" : "transparent",
              borderRadius: 3,
              padding: "1px 2px",
            }}
            ref={isActive ? currentRef : undefined}
          >
            {s.text.trim()}{" "}
          </span>
        );
      })}
    </div>
  );
}

// --- Agent Panel ---
function AgentPanel({ fixing, log, onSubmit, currentConcept, currentTime }: {
  fixing: boolean;
  log: string[];
  onSubmit: (feedback: string) => void;
  currentConcept: number | undefined;
  currentTime: number;
}) {
  const [input, setInput] = useState("");
  const logRef = useRef<HTMLDivElement>(null);
  const [elapsed, setElapsed] = useState(0);

  // Live timer when fixing
  useEffect(() => {
    if (!fixing) { setElapsed(0); return; }
    const start = Date.now();
    const timer = setInterval(() => setElapsed((Date.now() - start) / 1000), 100);
    return () => clearInterval(timer);
  }, [fixing]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const handleSubmit = () => {
    onSubmit(input);
    setInput("");
  };

  const quickFeedback = (text: string) => {
    onSubmit(`At ${currentTime.toFixed(1)}s (concept ${currentConcept}): ${text}`);
  };

  // Status line based on elapsed time and log content
  const hasLog = log.length > 1 || (log.length === 1 && log[0].length > 50);
  let statusText = "";
  if (fixing) {
    if (elapsed < 2) statusText = "Starting opencode...";
    else if (!hasLog && elapsed < 8) statusText = "Agent is thinking...";
    else if (log.join("").includes("tsc")) statusText = "Verifying TypeScript...";
    else if (log.join("").includes("remotion render")) statusText = "Rendering video...";
    else if (hasLog) statusText = "Agent is working...";
    else statusText = `Working... ${elapsed.toFixed(0)}s`;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Quick feedback buttons */}
      <div style={{ padding: 6, display: "flex", flexWrap: "wrap", gap: 3, borderBottom: `1px solid ${COLORS.cardBorder}` }}>
        <button onClick={() => quickFeedback("no movement for too long, violates the 5-second rule")} style={btnStyle}>No movement</button>
        <button onClick={() => quickFeedback("highlight is too early")} style={btnStyle}>Highlight early</button>
        <button onClick={() => quickFeedback("highlight is too late")} style={btnStyle}>Highlight late</button>
        <button onClick={() => quickFeedback("text overlaps the box")} style={btnStyle}>Overlap</button>
        <button onClick={() => quickFeedback("diagram doesn't match what the narration says")} style={btnStyle}>Mismatch</button>
      </div>

      {/* Input */}
      <div style={{ padding: 6, display: "flex", gap: 6 }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && !fixing) handleSubmit(); }}
          placeholder={fixing ? "Agent is working..." : "Describe what's wrong..."}
          disabled={fixing}
          style={{
            flex: 1, background: COLORS.card, color: COLORS.text, border: `1px solid ${COLORS.cardBorder}`,
            borderRadius: 4, padding: "5px 8px", fontFamily: "inherit", fontSize: 12,
          }}
        />
        <button
          onClick={handleSubmit}
          disabled={fixing || !input.trim()}
          style={{
            ...btnStyle,
            background: fixing ? COLORS.cardBorder : COLORS.primary,
            color: fixing ? COLORS.textFaint : COLORS.bg,
            cursor: fixing ? "not-allowed" : "pointer",
            padding: "5px 12px",
          }}
        >
          {fixing ? "..." : "Fix"}
        </button>
      </div>

      {/* Status bar */}
      {fixing && (
        <div style={{
          padding: "4px 8px", background: "rgba(98, 196, 255, 0.08)",
          borderBottom: `1px solid ${COLORS.cardBorder}`,
          fontSize: 10, color: COLORS.primary, display: "flex", alignItems: "center", gap: 6,
        }}>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: COLORS.primary, opacity: 0.4 + Math.abs(Math.sin(elapsed * 3)) * 0.6 }} />
          {statusText}
          <span style={{ marginLeft: "auto", color: COLORS.textFaint }}>{elapsed.toFixed(1)}s</span>
        </div>
      )}

      {/* Log */}
      <div ref={logRef} style={{ flex: 1, minHeight: 0, overflow: "auto", padding: 8, fontSize: 10, lineHeight: 1.5, color: COLORS.textDim, fontFamily: "'SF Mono', monospace", whiteSpace: "pre-wrap" }}>
        {log.length === 0 ? (
          <span style={{ color: COLORS.textFaint }}>
            Describe an issue or use a quick button above. The agent will edit the source code, re-render, and reload the video.
          </span>
        ) : (
          log.map((line, i) => {
            // Highlight lines that look like tool calls or results
            const isTool = line.includes("Reading") || line.includes("Editing") || line.includes("Running");
            const isError = line.includes("error") || line.includes("Error") || line.includes("FAIL");
            const isDone = line.includes("===") || line.includes("reloaded");
            return (
              <span key={i} style={{ color: isError ? COLORS.danger : isDone ? COLORS.success : isTool ? COLORS.primary : COLORS.textDim }}>
                {line}
              </span>
            );
          })
        )}
      </div>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  background: COLORS.card, color: COLORS.textDim, border: `1px solid ${COLORS.cardBorder}`,
  borderRadius: 4, padding: "3px 8px", fontFamily: "inherit", fontSize: 10, cursor: "pointer",
};
