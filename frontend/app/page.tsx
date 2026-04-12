"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const TIME_SIGNATURES = ["4/4", "3/4", "3/8", "2/4", "6/8"];

export default function Home() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [tempo, setTempo] = useState(120);
  const [timeSig, setTimeSig] = useState("4/4");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = (f: File) => {
    if (!f.name.match(/\.(mp3|wav|flac)$/i)) {
      setError("Please upload an MP3, WAV, or FLAC file.");
      return;
    }
    setError(null);
    setFile(f);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const form = new FormData();
    form.append("file", file);
    form.append("tempo", String(tempo));
    form.append("time_sig", timeSig);

    try {
      const res = await fetch(`${API_URL}/transcribe`, { method: "POST", body: form });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail ?? "Upload failed");
      }
      const { job_id } = await res.json();
      router.push(`/result/${job_id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center px-4">
      <div className="w-full max-w-lg space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">HarmonyNet</h1>
          <p className="text-gray-400">Upload solo piano audio and get sheet music.</p>
        </div>

        {/* Drop zone */}
        <div
          className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-colors ${
            dragging ? "border-indigo-400 bg-indigo-950" : "border-gray-700 hover:border-gray-500"
          }`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const f = e.dataTransfer.files[0];
            if (f) handleFile(f);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".mp3,.wav,.flac"
            className="hidden"
            onChange={(e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }}
          />
          {file ? (
            <p className="text-indigo-300 font-medium">{file.name}</p>
          ) : (
            <>
              <p className="text-gray-300 font-medium">Drop your audio file here</p>
              <p className="text-gray-500 text-sm mt-1">MP3, WAV, or FLAC</p>
            </>
          )}
        </div>

        {/* Settings */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-sm text-gray-400">Tempo (BPM)</label>
            <input
              type="number"
              min={20}
              max={300}
              value={tempo}
              onChange={(e) => setTempo(Number(e.target.value))}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <div className="space-y-1">
            <label className="text-sm text-gray-400">Time Signature</label>
            <select
              value={timeSig}
              onChange={(e) => setTimeSig(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {TIME_SIGNATURES.map((ts) => (
                <option key={ts}>{ts}</option>
              ))}
            </select>
          </div>
        </div>

        {error && <p className="text-red-400 text-sm text-center">{error}</p>}

        <button
          onClick={handleSubmit}
          disabled={!file || loading}
          className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-xl transition-colors"
        >
          {loading ? "Submitting…" : "Transcribe"}
        </button>
      </div>
    </main>
  );
}
