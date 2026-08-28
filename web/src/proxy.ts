import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Next.js 16 renamed `middleware` to `proxy` (same mechanism, new name/file).
const SESSION_COOKIE_NAME = "autoace_session";
const PUBLIC_PATHS = ["/login"];

// This only checks whether the session cookie is *present*, not whether its
// signature/expiry is valid — verifying an itsdangerous-signed cookie would
// mean duplicating that scheme (and sharing the signing secret) in JS. The
// real check is the API's own `require_session` dependency on every
// request; a stale/tampered cookie gets past this guard but is rejected by
// the API, and the dashboard pages redirect to /login on a 401 from there.
export function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const isPublicPath = PUBLIC_PATHS.some((path) => pathname.startsWith(path));
  const hasSessionCookie = request.cookies.has(SESSION_COOKIE_NAME);

  if (!isPublicPath && !hasSessionCookie) {
    return NextResponse.redirect(new URL("/login", request.url));
  }
  if (pathname === "/login" && hasSessionCookie) {
    return NextResponse.redirect(new URL("/", request.url));
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};
