import type { Metadata } from "next";
// Self-hosted (npm "geist") instead of next/font/google: builds need no network access.
import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Interview Voice Coach",
  description: "Practice technical interviews out loud with a real-time voice interviewer.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <header className="border-b border-line bg-surface">
          <nav className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
            <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
              <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-full bg-accent" />
              Interview Voice Coach
            </Link>
            <Link href="/#history" className="text-sm text-ink-2 hover:text-ink">
              History
            </Link>
          </nav>
        </header>
        <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
