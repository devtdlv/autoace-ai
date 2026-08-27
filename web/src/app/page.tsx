type Milestone = {
  label: string;
  status: "done" | "in_progress" | "todo";
  detail: string;
};

const milestones: Milestone[] = [
  {
    label: "Audio preprocessing",
    status: "done",
    detail: "Normalizes any uploaded format to a consistent format for analysis.",
  },
  {
    label: "Speech detection & transcription",
    status: "done",
    detail: "Locates speech vs. silence and transcribes calls, fully on this server.",
  },
  {
    label: "Emotional tone analysis",
    status: "in_progress",
    detail: "Detects tone (neutral/satisfied/frustrated/upset/distressed) and intensity.",
  },
  {
    label: "Background noise & audio quality analysis",
    status: "todo",
    detail: "Noise presence/type/severity, technical quality, speaker overlap, long silence.",
  },
  {
    label: "Batch results & confidence scoring",
    status: "todo",
    detail: "Combines every signal into the final per-call result.",
  },
  {
    label: "Login & batch upload dashboard",
    status: "todo",
    detail: "Upload a folder/ZIP + CSV manifest, track progress, review and export results.",
  },
];

const statusStyles: Record<Milestone["status"], string> = {
  done: "bg-emerald-50 text-emerald-700 ring-emerald-600/20 dark:bg-emerald-500/10 dark:text-emerald-400 dark:ring-emerald-500/30",
  in_progress:
    "bg-amber-50 text-amber-700 ring-amber-600/20 dark:bg-amber-500/10 dark:text-amber-400 dark:ring-amber-500/30",
  todo: "bg-zinc-100 text-zinc-500 ring-zinc-500/20 dark:bg-zinc-500/10 dark:text-zinc-400 dark:ring-zinc-500/20",
};

const statusLabels: Record<Milestone["status"], string> = {
  done: "Done",
  in_progress: "In progress",
  todo: "Not started",
};

export default function Home() {
  return (
    <div className="flex flex-col flex-1 items-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex w-full max-w-2xl flex-col gap-10 px-6 py-20 sm:px-8">
        <header className="flex flex-col gap-3">
          <span className="text-sm font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
            AutoAce AI
          </span>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            Call Analysis Dashboard
          </h1>
          <p className="max-w-lg text-base leading-7 text-zinc-600 dark:text-zinc-400">
            Analyzes production call audio for emotional tone, background
            noise, and audio quality. Processing runs entirely on this
            server — audio is never sent to a third party.
          </p>
        </header>

        <section className="flex flex-col gap-3">
          <h2 className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
            Build status
          </h2>
          <ul className="flex flex-col gap-2">
            {milestones.map((m) => (
              <li
                key={m.label}
                className="flex items-start justify-between gap-4 rounded-lg border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-950"
              >
                <div className="flex flex-col gap-0.5">
                  <span className="text-sm font-medium text-zinc-900 dark:text-zinc-50">
                    {m.label}
                  </span>
                  <span className="text-sm text-zinc-500 dark:text-zinc-400">
                    {m.detail}
                  </span>
                </div>
                <span
                  className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${statusStyles[m.status]}`}
                >
                  {statusLabels[m.status]}
                </span>
              </li>
            ))}
          </ul>
        </section>

        <footer className="text-sm text-zinc-400 dark:text-zinc-600">
          Login and batch upload will appear here once that milestone is
          complete.
        </footer>
      </main>
    </div>
  );
}
