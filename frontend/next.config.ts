import type { NextConfig } from "next";

// Two deployment modes:
// 1. STATIC_EXPORT=1 -> plain HTML/JS in out/, served by the FastAPI service itself
//    (single Render service; the browser calls same-origin /api/*).
// 2. Default -> Next.js server that proxies /api/* to BACKEND_URL (separate web service).
function backendUrl(): string {
  const raw = process.env.BACKEND_URL || "http://localhost:8000";
  return /^https?:\/\//.test(raw) ? raw.replace(/\/$/, "") : `https://${raw}`;
}

const isStatic = process.env.STATIC_EXPORT === "1";

const nextConfig: NextConfig = isStatic
  ? { output: "export", trailingSlash: true, images: { unoptimized: true } }
  : {
      async rewrites() {
        return [{ source: "/api/:path*", destination: `${backendUrl()}/api/:path*` }];
      },
    };

export default nextConfig;
