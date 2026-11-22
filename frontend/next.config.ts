import type { NextConfig } from "next";

// The browser always calls same-origin /api/*; Next proxies to the FastAPI service.
// On Render, BACKEND_URL comes from the API service's host (no scheme), so add https://.
function backendUrl(): string {
  const raw = process.env.BACKEND_URL || "http://localhost:8000";
  return /^https?:\/\//.test(raw) ? raw.replace(/\/$/, "") : `https://${raw}`;
}

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backendUrl()}/api/:path*` }];
  },
};

export default nextConfig;
