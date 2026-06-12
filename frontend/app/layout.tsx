import type { Metadata } from "next";
import { Geist_Mono } from "next/font/google";
import Link from "next/link";
import { BrainCircuit } from "lucide-react";
import "./globals.css";

const mono = Geist_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "LLM Studio",
  description: "LLM fine-tuning dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={mono.variable}>
      <body className="bg-gray-50 text-gray-900 min-h-screen font-sans antialiased">
        <nav className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-8">
          <Link href="/" className="flex items-center gap-2 text-gray-900 font-semibold tracking-tight text-sm">
            <BrainCircuit size={18} className="text-blue-600" />
            LLM Studio
          </Link>
          <Link href="/" className="text-gray-500 hover:text-gray-900 text-sm transition-colors">
            Jobs
          </Link>
          <Link href="/experiments" className="text-gray-500 hover:text-gray-900 text-sm transition-colors">
            Experiments
          </Link>
          <Link href="/compute" className="text-gray-500 hover:text-gray-900 text-sm transition-colors">
            Compute
          </Link>
        </nav>
        <main className="max-w-6xl mx-auto px-6 py-8">{children}</main>
        <footer className="border-t border-gray-200 bg-white mt-auto">
          <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
            <p className="text-xs text-gray-400">
              LLM Studio — fine-tune and serve large language models with experiment tracking, remote GPU compute, and real-time inference.
            </p>
            <a
              href="https://github.com/bythebug"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-gray-400 hover:text-gray-700 transition-colors font-mono"
            >
              @bythebug
            </a>
          </div>
        </footer>
      </body>
    </html>
  );
}
