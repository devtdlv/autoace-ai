"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import type { Batch, Call } from "@/lib/api";

const POLL_INTERVAL_MS = 3000;

const CALL_STATUS_STYLES: Record<Call["status"], string> = {
  pending: "bg-zinc-100 text-zinc-600 dark:bg-zinc-500/10 dark:text-zinc-400",
  processing: "bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400",
  done: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400",
  failed: "bg-red-50 text-red-700 dark:bg-red-500/10 dark:text-red-400",
};

export default function BatchDetailPage(props: PageProps<"/batches/[id]">) {
  const { id } = use(props.params);
  const router = useRouter();
  const [batch, setBatch] = useState<Batch | null>(null);
  const [calls, setCalls] = useState<Call[] | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const res = await fetch(`/api/batches/${id}`);
      if (cancelled) return;
      if (res.status === 401) {
        router.push("/login");
        return;
      }
      if (res.status === 404) {
        setNotFound(true);
        return;
      }
      if (res.ok) {
        const body = await res.json();
        setBatch(body.batch);
        setCalls(body.calls);
      }
    }

    load();
    const interval = setInterval(load, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [id, router]);

  async function handleDelete() {
    const res = await fetch(`/api/batches/${id}`, { method: "DELETE" });
    if (res.status === 401) {
      router.push("/login");
      return;
    }
    router.push("/");
  }

  if (notFound) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 bg-zinc-50 dark:bg-black">
        <p className="text-sm text-zinc-500 dark:text-zinc-400">Batch not found.</p>
        <Link href="/" className="text-sm text-zinc-700 underline dark:text-zinc-300">
          Back to dashboard
        </Link>
      </div>
    );
  }

  return (
    <div className="flex flex-col flex-1 items-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex w-full max-w-4xl flex-col gap-6 px-6 py-12 sm:px-8">
        <div className="flex items-center justify-between">
          <Link href="/" className="text-sm text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200">
            ← Dashboard
          </Link>
          {batch && (
            <div className="flex gap-3">
              <a
                href={`/api/batches/${id}/export?format=csv`}
                className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-900"
              >
                Export CSV
              </a>
              <a
                href={`/api/batches/${id}/export?format=json`}
                className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-900"
              >
                Export JSON
              </a>
              {confirmingDelete ? (
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-zinc-500 dark:text-zinc-400">Delete?</span>
                  <button
                    onClick={handleDelete}
                    className="rounded-md bg-red-600 px-3 py-1.5 font-medium text-white hover:bg-red-700"
                  >
                    Confirm
                  </button>
                  <button
                    onClick={() => setConfirmingDelete(false)}
                    className="rounded-md border border-zinc-300 px-3 py-1.5 text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-900"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setConfirmingDelete(true)}
                  className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm text-red-600 hover:bg-red-50 dark:border-zinc-700 dark:text-red-400 dark:hover:bg-red-500/10"
                >
                  Delete
                </button>
              )}
            </div>
          )}
        </div>

        {batch && (
          <header className="flex flex-col gap-2">
            <h1 className="text-xl font-semibold text-zinc-900 dark:text-zinc-50">{batch.manifest_name}</h1>
            <div className="flex items-center gap-3 text-sm text-zinc-500 dark:text-zinc-400">
              <span>{new Date(batch.created_at).toLocaleString()}</span>
              <span>·</span>
              <span>
                {batch.completed_calls}/{batch.total_calls} processed
              </span>
              <span>·</span>
              <span className="capitalize">{batch.status}</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
              <div
                className="h-full bg-zinc-900 transition-all dark:bg-zinc-50"
                style={{
                  width: `${batch.total_calls > 0 ? (batch.completed_calls / batch.total_calls) * 100 : 0}%`,
                }}
              />
            </div>
          </header>
        )}

        <div className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
          <table className="w-full min-w-max text-left text-sm">
            <thead className="bg-zinc-100 text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
              <tr>
                {[
                  "File", "Status", "Tone", "Intensity", "Noise", "Noise type",
                  "Severity", "Quality", "Overlap", "Long silence", "Confidence",
                ].map((h) => (
                  <th key={h} className="px-3 py-2 font-medium whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
              {calls?.map((call) => (
                <tr key={call.id} className="bg-white dark:bg-zinc-950">
                  <td className="px-3 py-2 whitespace-nowrap text-zinc-900 dark:text-zinc-50">{call.filename}</td>
                  <td className="px-3 py-2 whitespace-nowrap">
                    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${CALL_STATUS_STYLES[call.status]}`}>
                      {call.status}
                    </span>
                  </td>
                  {call.result ? (
                    <>
                      <td className="px-3 py-2 whitespace-nowrap capitalize">{call.result.emotional_tone}</td>
                      <td className="px-3 py-2 whitespace-nowrap capitalize">{call.result.emotional_intensity}</td>
                      <td className="px-3 py-2 whitespace-nowrap">
                        {call.result.background_noise_present ? "Yes" : "No"}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap">{call.result.background_noise_type || "—"}</td>
                      <td className="px-3 py-2 whitespace-nowrap capitalize">
                        {call.result.background_noise_severity}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap capitalize">
                        {call.result.audio_quality.replace("_", " ")}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap">
                        {call.result.speaker_overlap_present ? "Yes" : "No"}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap">
                        {call.result.long_silence_present ? "Yes" : "No"}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap">{call.result.confidence.toFixed(2)}</td>
                    </>
                  ) : (
                    <td className="px-3 py-2 text-zinc-400" colSpan={9}>
                      {call.status === "failed" ? call.error ?? "Failed" : "—"}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}
