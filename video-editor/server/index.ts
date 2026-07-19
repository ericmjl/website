import express from "express";
import cors from "cors";
import { spawn, execSync } from "child_process";
import { readdirSync, readFileSync, existsSync } from "fs";
import { join } from "path";
import { createServer } from "http";
import { WebSocketServer } from "ws";

const REMOTION_DIR = "/Users/ericmjl/github/website/remotion-videos";
const PORT = 4097;

const app = express();
app.use(cors());
app.use(express.json());

// --- Composition discovery ---
// Scans public/*-narration/ for words.json and matches to compositions
interface Composition {
  id: string;
  narrationDir: string;
  videoPath: string;
}

function discoverCompositions(): Composition[] {
  const comps: Composition[] = [];
  const publicDir = join(REMOTION_DIR, "public");
  if (!existsSync(publicDir)) return comps;

  for (const dir of readdirSync(publicDir, { withFileTypes: true })) {
    if (!dir.isDirectory() || !dir.name.endsWith("-narration")) continue;
    const narrationDir = join(publicDir, dir.name);
    if (!existsSync(join(narrationDir, "words.json"))) continue;

    // Derive composition ID from narration dir name
    // e.g. "autolearn-narration" -> "AutolearnVideo"
    const base = dir.name.replace("-narration", "");
    const videoPath = join(REMOTION_DIR, "out", `${base}-video.mp4`);
    if (!existsSync(videoPath)) continue;

    comps.push({ id: base, narrationDir, videoPath });
  }
  return comps;
}

// --- STEP_BOUNDARIES parser ---
// Extracts STEP_BOUNDARIES from the .tsx source using regex
function parseStepBoundaries(compositionId: string): any[] {
  // Try to find the video source file
  const PascalName = compositionId.charAt(0).toUpperCase() + compositionId.slice(1);
  const sourcePath = join(REMOTION_DIR, "src", `${PascalName}Video.tsx`);
  if (!existsSync(sourcePath)) return [];

  const source = readFileSync(sourcePath, "utf-8");
  const matches = [...source.matchAll(/\{\s*ts:\s*([\d.]+),\s*concept:\s*(\d+),\s*step:\s*(\d+)\s*\}/g)];
  return matches.map(m => ({
    ts: parseFloat(m[1]),
    concept: parseInt(m[2]),
    step: parseInt(m[3]),
  }));
}

// --- Routes ---

app.get("/api/compositions", (_req, res) => {
  res.json(discoverCompositions().map(c => ({ id: c.id })));
});

app.get("/api/words/:compositionId", (req, res) => {
  const comp = discoverCompositions().find(c => c.id === req.params.compositionId);
  if (!comp) return res.status(404).json({ error: "Not found" });
  res.sendFile(join(comp.narrationDir, "words.json"));
});

app.get("/api/captions/:compositionId", (req, res) => {
  const comp = discoverCompositions().find(c => c.id === req.params.compositionId);
  if (!comp) return res.status(404).json({ error: "Not found" });
  res.sendFile(join(comp.narrationDir, "captions.json"));
});

app.get("/api/boundaries/:compositionId", (req, res) => {
  res.json(parseStepBoundaries(req.params.compositionId));
});

app.get("/api/video/:compositionId", (req, res) => {
  const comp = discoverCompositions().find(c => c.id === req.params.compositionId);
  if (!comp) return res.status(404).json({ error: "Not found" });
  res.sendFile(comp.videoPath);
});

// --- WebSocket for live agent feedback ---
const server = createServer(app);
const wss = new WebSocketServer({ server });

interface FixJob {
  id: string;
  status: "running" | "done" | "error";
  log: string;
  compositionId: string;
}

const jobs = new Map<string, FixJob>();

app.post("/api/fix", (req, res) => {
  const { compositionId, feedback } = req.body;
  if (!compositionId || !feedback) return res.status(400).json({ error: "Missing compositionId or feedback" });

  const jobId = `fix-${Date.now()}`;
  const PascalName = compositionId.charAt(0).toUpperCase() + compositionId.slice(1);
  const job: FixJob = { id: jobId, status: "running", log: "", compositionId };
  jobs.set(jobId, job);

  const prompt = `Fix the Remotion video composition ${PascalName}Video. The user reports: "${feedback}".

Relevant files:
- src/${PascalName}Video.tsx (composition with STEP_BOUNDARIES)
- src/scenes/${compositionId}/transcript.ts (typed gems with delayFrames)
- src/scenes/${compositionId}/diagrams.tsx (SVG diagram components)

Read these files, understand the issue, make the fix, then verify with 'npx tsc --noEmit'. After fixing, render with 'npx remotion render src/index.ts ${PascalName}Video out/${compositionId}-video.mp4 --concurrency=4'.`;

  // Broadcast function
  const broadcast = (data: any) => {
    wss.clients.forEach(client => {
      if (client.readyState === 1) client.send(JSON.stringify(data));
    });
  };

  // Run opencode with print-logs for streaming output
  const proc = spawn("opencode", ["run", "--print-logs", prompt], {
    cwd: REMOTION_DIR,
    env: { ...process.env, AUTOLEARN_REVIEWER: "1" },
  });

  proc.stdout.on("data", (data) => {
    const text = data.toString();
    job.log += text;
    broadcast({ type: "log", jobId, text });
  });

  proc.stderr.on("data", (data) => {
    const text = data.toString();
    job.log += text;
    broadcast({ type: "log", jobId, text });
  });

  proc.on("close", (code) => {
    job.status = code === 0 ? "done" : "error";
    job.log += `\n[Process exited with code ${code}]\n`;
    broadcast({ type: "done", jobId, status: job.status, compositionId });
  });

  res.json({ jobId });
});

app.get("/api/jobs/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) return res.status(404).json({ error: "Job not found" });
  res.json(job);
});

server.listen(PORT, () => {
  console.log(`AI Video Editor backend on http://localhost:${PORT}`);
});
