"use client";

import { PipecatClient, RTVIEvent, type BotOutputData, type TranscriptData } from "@pipecat-ai/client-js";
import {
  PipecatClientAudio,
  PipecatClientProvider,
  usePipecatClient,
  usePipecatClientMicControl,
  usePipecatClientTransportState,
  useRTVIClientEvent,
  VoiceVisualizer,
} from "@pipecat-ai/client-react";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { formatDuration } from "@/lib/api";

import { Button, Card } from "./ui";

interface Message {
  id: number;
  role: "user" | "bot";
  text: string;
}

interface Progress {
  question: number;
  total: number;
  phase: string;
  topic: string | null;
}

interface Latency {
  total_ms: number;
  turn: number;
  llm: number;
  tts: number;
  output: number;
}

/**
 * Rendered client-side only (see InterviewRoomLoader): the Pipecat client touches
 * navigator/WebRTC, so it must never be constructed during server rendering.
 */
export default function InterviewRoom({ sessionId }: { sessionId: string }) {
  const [client] = useState(
    () => new PipecatClient({ transport: new SmallWebRTCTransport(), enableMic: true, enableCam: false }),
  );

  useEffect(() => {
    return () => {
      void client.disconnect();
    };
  }, [client]);

  return (
    <PipecatClientProvider client={client}>
      <Room sessionId={sessionId} />
      <PipecatClientAudio />
    </PipecatClientProvider>
  );
}

function Room({ sessionId }: { sessionId: string }) {
  const client = usePipecatClient();
  const router = useRouter();
  const transportState = usePipecatClientTransportState();
  const { enableMic, isMicEnabled } = usePipecatClientMicControl();

  const [messages, setMessages] = useState<Message[]>([]);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [latency, setLatency] = useState<Latency[]>([]);
  const [botSpeaking, setBotSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const ended = useRef(false);
  const nextId = useRef(0);
  const scroller = useRef<HTMLDivElement>(null);

  const connected = transportState === "connected" || transportState === "ready";

  useEffect(() => {
    if (!startedAt) return;
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, [startedAt]);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const append = useCallback((role: Message["role"], text: string, newTurn: boolean) => {
    const clean = text.trim();
    if (!clean) return;
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === role && !newTurn) {
        return [...prev.slice(0, -1), { ...last, text: `${last.text} ${clean}` }];
      }
      return [...prev, { id: nextId.current++, role, text: clean }];
    });
  }, []);

  useRTVIClientEvent(
    RTVIEvent.BotStartedSpeaking,
    useCallback(() => {
      setBotSpeaking(true);
    }, []),
  );
  useRTVIClientEvent(
    RTVIEvent.BotStoppedSpeaking,
    useCallback(() => setBotSpeaking(false), []),
  );
  useRTVIClientEvent(
    RTVIEvent.UserTranscript,
    useCallback(
      (data: TranscriptData) => {
        if (data.final) append("user", data.text, false);
      },
      [append],
    ),
  );
  useRTVIClientEvent(
    RTVIEvent.BotOutput,
    useCallback(
      (data: BotOutputData) => {
        // Each sentence arrives as "new" (queued for synthesis), then "completed" (spoken).
        // Show only what was actually spoken, so an interrupted reply stays truncated.
        if (data.aggregated_by === "word") return;
        if (data.spoken_status ? data.spoken_status !== "completed" : data.spoken === false) return;
        append("bot", data.text, false);
      },
      [append],
    ),
  );
  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    useCallback((data: { type?: string } & Record<string, unknown>) => {
      if (data.type === "progress") setProgress(data as unknown as Progress);
      if (data.type === "latency") setLatency((prev) => [...prev.slice(-19), data as unknown as Latency]);
    }, []),
  );

  const goToReport = useCallback(() => {
    if (ended.current) return;
    ended.current = true;
    router.push(`/report/${sessionId}`);
  }, [router, sessionId]);

  useRTVIClientEvent(
    RTVIEvent.Disconnected,
    useCallback(() => {
      // The interviewer ends the call after the goodbye; head to the report.
      if (startedAt) setTimeout(goToReport, 800);
    }, [goToReport, startedAt]),
  );

  async function start() {
    setError(null);
    try {
      await client!.initDevices();
      await client!.connect({
        webrtcRequestParams: { endpoint: "/api/offer", requestData: { session_id: sessionId } },
      });
      setStartedAt(Date.now());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function end() {
    await client!.disconnect();
    goToReport();
  }

  const lastLatency = latency[latency.length - 1];
  const elapsed = startedAt ? (now - startedAt) / 1000 : 0;

  if (!startedAt) {
    return (
      <Card className="mx-auto max-w-xl text-center">
        <h1 className="text-xl font-semibold tracking-tight">Ready when you are</h1>
        <p className="mx-auto mt-2 max-w-md text-sm text-ink-2">
          Use headphones if you can. Speak naturally: pause to think, and interrupt the interviewer whenever you
          like. The interview ends on its own after the last question.
        </p>
        {error && (
          <p className="mt-4 rounded-lg bg-bad-bg px-3 py-2 text-sm text-bad">
            Couldn&apos;t connect: {error}
          </p>
        )}
        <Button className="mt-6" onClick={start} disabled={transportState === "connecting" || transportState === "authenticating"}>
          {transportState === "connecting" ? "Connecting…" : "Start interview"}
        </Button>
      </Card>
    );
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
      <Card className="flex h-[70vh] flex-col p-0">
        <div className="flex items-center justify-between border-b border-line px-5 py-3">
          <div className="text-sm">
            {progress && progress.question > 0 ? (
              <>
                <span className="font-medium">
                  Question {progress.question} of {progress.total}
                </span>
                {progress.topic && <span className="text-ink-2"> · {progress.topic}</span>}
                {progress.phase === "follow_up" && <span className="text-ink-2"> · follow-up</span>}
              </>
            ) : (
              <span className="font-medium">Introduction</span>
            )}
          </div>
          <span className="tabular text-sm text-ink-2" aria-label="Elapsed time">
            {formatDuration(elapsed)}
          </span>
        </div>
        <div ref={scroller} className="flex-1 space-y-3 overflow-y-auto px-5 py-4" aria-live="polite">
          {messages.length === 0 && <p className="text-sm text-muted">Connecting to your interviewer…</p>}
          {messages.map((m) => (
            <div key={m.id} className={m.role === "user" ? "flex justify-end" : "flex"}>
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-2 text-sm leading-relaxed ${
                  m.role === "user" ? "bg-accent text-accent-ink" : "bg-surface-2"
                }`}
              >
                <div className="mb-0.5 text-[11px] font-medium opacity-70">{m.role === "user" ? "You" : "Alex"}</div>
                {m.text}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <div className="space-y-4">
        <Card>
          <div className="mb-3 text-xs font-medium text-muted">Interviewer</div>
          <div className="flex h-12 items-center justify-center">
            <VoiceVisualizer participantType="bot" barColor="var(--series-1)" barCount={12} barWidth={4} barGap={3} barMaxHeight={44} />
          </div>
          <p className="mt-2 text-center text-xs text-ink-2">{botSpeaking ? "Speaking…" : "Listening"}</p>
          <div className="mt-4 mb-3 text-xs font-medium text-muted">You</div>
          <div className="flex h-12 items-center justify-center">
            <VoiceVisualizer participantType="local" barColor="var(--series-3)" barCount={12} barWidth={4} barGap={3} barMaxHeight={44} />
          </div>
          <div className="mt-4 flex gap-2">
            <Button variant="ghost" className="flex-1" onClick={() => enableMic(!isMicEnabled)}>
              {isMicEnabled ? "Mute" : "Unmute"}
            </Button>
            <Button variant="danger" className="flex-1" onClick={end} disabled={!connected}>
              End
            </Button>
          </div>
        </Card>

        <Card>
          <div className="text-xs font-medium text-muted">Response latency (last turn)</div>
          {lastLatency ? (
            <>
              <div className="tabular mt-1 text-2xl font-semibold tracking-tight">
                {(lastLatency.total_ms / 1000).toFixed(2)} s
              </div>
              <dl className="tabular mt-2 grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs text-ink-2">
                <dt>Turn detection</dt>
                <dd className="text-right">{Math.round(lastLatency.turn)} ms</dd>
                <dt>LLM</dt>
                <dd className="text-right">{Math.round(lastLatency.llm)} ms</dd>
                <dt>TTS</dt>
                <dd className="text-right">{Math.round(lastLatency.tts)} ms</dd>
              </dl>
            </>
          ) : (
            <p className="mt-1 text-sm text-muted">Appears after your first answer.</p>
          )}
        </Card>
      </div>
    </div>
  );
}
