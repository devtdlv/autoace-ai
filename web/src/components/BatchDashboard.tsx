"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import type { Batch } from "@/lib/api";

const STATUS_STYLES: Record<Batch["status"], string> = {
  pending: "bg-zinc-100 text-zinc-600 dark:bg-zinc-500/10 dark:text-zinc-400",
  processing: "bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400",
  done: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400",
  failed: "bg-red-50 text-red-700 dark:bg-red-500/10 dark:text-red-400",
};

const POLL_INTERVAL_MS = 4000;

export default function BatchDashboard() {
  const router = useRouter();
  const [batches, setBatches] = useState<Batch[] | null>(null);
  const [manifestFile, setManifestFile] = useState<File | null>(null);
  const [audioFiles, setAudioFiles] = useState<FileList | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [refreshTick, setRefreshTick] = useState(0);
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);
  const manifestInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  // The loader lives entirely inside the effect (rather than a shared
  // useCallback invoked here too) — triggering a re-fetch elsewhere bumps
  // refreshTick instead of calling this directly, which is what satisfies
  // react-hooks/set-state-in-effect for a fetch-on-mount-and-poll effect.
  useEffect(() => {
    let cancelled = false;

    async function load() {
      const res = await fetch("/api/batches");
      if (cancelled) return;
      if (res.status === 401) {
        router.push("/login");
        return;
      }
      if (res.ok) setBatches(await res.json());
    }

    load();
    const interval = setInterval(load, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [router, refreshTick]);

  async function handleLogout() {
    await fetch("/api/auth/logout", { method: "POST" });
    router.push("/login");
  }

  async function handleDelete(id: string) {
    const res = await fetch(`/api/batches/${id}`, { method: "DELETE" });
    if (res.status === 401) {
      router.push("/login");
      return;
    }
    setConfirmingDeleteId(null);
    setRefreshTick((t) => t + 1);
  }

  async function handleUpload(e: FormEvent) {
    e.preventDefault();
    if (!audioFiles || audioFiles.length === 0) {
      setUploadError("Choose at least one audio file (or a .zip of them).");
      return;
    }
    setUploading(true);
    setUploadError(null);

    const formData = new FormData();
    if (manifestFile) formData.append("manifest", manifestFile);

    const isSingleZip = audioFiles.length === 1 && audioFiles[0].name.toLowerCase().endsWith(".zip");
    if (isSingleZip) {
      formData.append("archive", audioFiles[0]);
    } else {
      Array.from(audioFiles).forEach((f) => formData.append("files", f));
    }

    try {
      const res = await fetch("/api/batches", { method: "POST", body: formData });
      if (res.status === 401) {
        router.push("/login");
        return;
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setUploadError(typeof body.detail === "string" ? body.detail : "Upload failed");
        return;
      }
      setManifestFile(null);
      setAudioFiles(null);
      if (manifestInputRef.current) manifestInputRef.current.value = "";
      if (audioInputRef.current) audioInputRef.current.value = "";
      setRefreshTick((t) => t + 1);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-zinc-500 dark:text-zinc-400">Batches</h2>
        <button
          onClick={handleLogout}
          className="text-sm text-zinc-500 underline hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
        >
          Sign out
        </button>
      </div>

      <form
        onSubmit={handleUpload}
        className="flex flex-col gap-3 rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950"
      >
        <h3 className="text-sm font-medium text-zinc-900 dark:text-zinc-50">New batch</h3>
        <label className="flex flex-col gap-1 text-sm text-zinc-600 dark:text-zinc-400">
          Call recordings — a .zip archive, or select the audio files
          directly (multi-select). If a manifest CSV is included among
          them, it is picked up automatically.
          <input
            ref={audioInputRef}
            type="file"
            multiple
            accept="audio/*,.zip,.csv"
            onChange={(e) => setAudioFiles(e.target.files)}
            className="text-sm text-zinc-700 dark:text-zinc-300"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm text-zinc-600 dark:text-zinc-400">
          Manifest CSV (optional, if not already included above) — a{" "}
          <code>name</code> column (exact filename), and optionally a{" "}
          <code>result_json</code> column with expected results for
          comparison. Omit entirely to process every file above.
          <input
            ref={manifestInputRef}
            type="file"
            accept=".csv"
            onChange={(e) => setManifestFile(e.target.files?.[0] ?? null)}
            className="text-sm text-zinc-700 dark:text-zinc-300"
          />
        </label>
        {uploadError && <p className="text-sm text-red-600 dark:text-red-400">{uploadError}</p>}
        <button
          type="submit"
          disabled={uploading}
          className="self-start rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-zinc-50 dark:text-zinc-900"
        >
          {uploading ? "Uploading…" : "Upload batch"}
        </button>
      </form>

      <ul className="flex flex-col gap-2">
        {batches === null && <li className="text-sm text-zinc-400">Loading…</li>}
        {batches?.length === 0 && <li className="text-sm text-zinc-400">No batches yet.</li>}
        {batches?.map((b) => (
          <li
            key={b.id}
            className="flex items-center gap-2 rounded-lg border border-zinc-200 bg-white px-4 py-3 hover:border-zinc-300 dark:border-zinc-800 dark:bg-zinc-950 dark:hover:border-zinc-700"
          >
            <Link href={`/batches/${b.id}`} className="flex flex-1 items-center justify-between gap-4">
              <div className="flex flex-col gap-0.5">
                <span className="text-sm font-medium text-zinc-900 dark:text-zinc-50">{b.manifest_name}</span>
                <span className="text-sm text-zinc-500 dark:text-zinc-400">
                  {b.completed_calls}/{b.total_calls} calls · {new Date(b.created_at).toLocaleString()}
                </span>
              </div>
              <span className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-medium ${STATUS_STYLES[b.status]}`}>
                {b.status}
              </span>
            </Link>
            {confirmingDeleteId === b.id ? (
              <div className="flex shrink-0 items-center gap-2 text-sm">
                <span className="text-zinc-500 dark:text-zinc-400">Delete?</span>
                <button
                  onClick={() => handleDelete(b.id)}
                  className="rounded-md bg-red-600 px-2 py-1 font-medium text-white hover:bg-red-700"
                >
                  Confirm
                </button>
                <button
                  onClick={() => setConfirmingDeleteId(null)}
                  className="rounded-md px-2 py-1 text-zinc-500 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setConfirmingDeleteId(b.id)}
                aria-label={`Delete batch ${b.manifest_name}`}
                className="shrink-0 rounded-md px-2 py-1 text-sm text-zinc-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-500/10 dark:hover:text-red-400"
              >
                Delete
              </button>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
