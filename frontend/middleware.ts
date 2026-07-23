import { NextRequest, NextResponse } from "next/server";

// Injects the shared backend API key server-side so it never reaches the
// browser bundle. Only runs on /api/* (proxied to the FastAPI backend via
// next.config.ts rewrites).
export function middleware(request: NextRequest) {
  const apiKey = process.env.API_KEY;
  if (!apiKey) return NextResponse.next();

  const headers = new Headers(request.headers);
  headers.set("x-api-key", apiKey);
  return NextResponse.next({ request: { headers } });
}

export const config = {
  matcher: ["/api/:path*"],
};
