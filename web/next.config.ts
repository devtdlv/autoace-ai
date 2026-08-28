import type { NextConfig } from "next";

// In production, nginx routes /api/* directly to the FastAPI service
// (deploy/nginx.conf) — this rewrite never fires there. In dev, the
// dashboard (localhost:3001) and API (localhost:8001) are separate
// processes; this makes /api/* look same-origin to the browser so the
// session cookie round-trips without any CORS/credentials wrangling in
// fetch calls, exactly mirroring prod's same-origin setup.
const API_ORIGIN = process.env.AUTOACE_API_ORIGIN ?? "http://127.0.0.1:8001";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_ORIGIN}/:path*`,
      },
    ];
  },
};

export default nextConfig;
