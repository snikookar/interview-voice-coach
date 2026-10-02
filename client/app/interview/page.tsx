import Link from "next/link";

import { InterviewRoomLoader } from "@/components/InterviewRoomLoader";

export default async function InterviewPage({ searchParams }: PageProps<"/interview">) {
  const { session } = await searchParams;
  if (typeof session !== "string") {
    return (
      <p className="text-sm">
        No session selected. <Link href="/" className="text-accent underline">Start a new interview</Link>.
      </p>
    );
  }
  return <InterviewRoomLoader sessionId={session} />;
}
