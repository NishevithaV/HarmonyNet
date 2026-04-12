"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Step = "transcribing" | "quantizing" | "encoding" | "rendering" | "analysing";

const STEP_LABELS: Record<Step, string> = {
  transcribing: "Detecting notes…",
  quantizing: "Quantizing to musical grid…",
  encoding: "Encoding to MusicXML…",
  rendering: "Rendering PDF…",
  analysing: "Generating AI analysis…",
};

interface DoneState {
  num_notes: number;
  num_measures: number;
  output_type: string;
  ai_analysis: string | null;
}

export default function ResultPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const [status, setStatus] = useState<"pending" | "processing" | "done" | "failed">("pending");
  const [step, setStep] = useState<Step | null>(null);
  const [result, setResult] = useState<DoneState | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const poll = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/status/${jobId}`);
        const data = await res.json();

        setStatus(data.status);

        if (data.status === "processing") setStep(data.step ?? null);

        if (data.status === "done") {
          setResult(data);
          clearInterval(poll);
        }

        if (data.status === "failed") {
          setError(data.error ?? "Transcription failed.");
          clearInterval(poll);
        }
      } catch {
        setError("Lost connection to server.");
        clearInterval(poll);
      }
    }, 2000);

    return () => clearInterval(poll);
  }, [jobId]);

  const downloadUrl = `${API_URL}/result/${jobId}`;

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center px-4">
      <div className="w-full max-w-lg space-y-8 text-center">
        <h1 className="text-3xl font-bold">HarmonyNet</h1>

        {/* Pending / processing */}
        {(status === "pending" || status === "processing") && (
          <div className="space-y-4">
            <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-gray-300">
              {status === "pending"
                ? "Job queued…"
                : step
                ? STEP_LABELS[step] ?? "Processing…"
                : "Processing…"}
            </p>
          </div>
        )}

        {/* Error */}
        {status === "failed" && (
          <p className="text-red-400">{error}</p>
        )}

        {/* Done */}
        {status === "done" && result && (
          <div className="space-y-6">
            <div className="bg-gray-900 rounded-2xl p-6 space-y-2 text-left">
              <p className="text-gray-400 text-sm">Notes detected</p>
              <p className="text-2xl font-semibold">{result.num_notes}</p>
              <p className="text-gray-400 text-sm mt-3">Measures</p>
              <p className="text-2xl font-semibold">{result.num_measures}</p>
            </div>

            {result.ai_analysis && (
              <div className="bg-indigo-950 border border-indigo-800 rounded-2xl p-6 text-left space-y-2">
                <p className="text-indigo-300 text-sm font-medium">AI Analysis</p>
                <p className="text-gray-200 text-sm leading-relaxed">{result.ai_analysis}</p>
              </div>
            )}

            <a
              href={downloadUrl}
              className="block w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-3 rounded-xl transition-colors"
            >
              Download {result.output_type.toUpperCase()}
            </a>

            <a href="/" className="text-gray-500 hover:text-gray-300 text-sm">
              ← Transcribe another piece
            </a>
          </div>
        )}
      </div>
    </main>
  );
}
