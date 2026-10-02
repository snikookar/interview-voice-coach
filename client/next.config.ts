import type { NextConfig } from "next";

// The browser only ever talks to the Next.js origin; /api/* (REST and the WebRTC
// SDP offer/ICE exchange) is proxied to the FastAPI server. One origin means no
// CORS setup and no server URL baked into client code.
const API_URL = process.env.API_URL ?? "http://127.0.0.1:7860";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};

export default nextConfig;
