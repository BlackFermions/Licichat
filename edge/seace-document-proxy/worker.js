const ALLOWED_HOSTS = new Set([
  "prod1.seace.gob.pe",
  "prod2.seace.gob.pe",
  "prod3.seace.gob.pe",
  "prod4.seace.gob.pe",
]);

const FILE_CODE_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const MAX_FILE_BYTES = 12 * 1024 * 1024;

async function secureEqual(left, right) {
  const encoder = new TextEncoder();
  const [leftHash, rightHash] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(left || "")),
    crypto.subtle.digest("SHA-256", encoder.encode(right || "")),
  ]);
  const leftBytes = new Uint8Array(leftHash);
  const rightBytes = new Uint8Array(rightHash);
  let difference = 0;
  for (let index = 0; index < leftBytes.length; index += 1) {
    difference |= leftBytes[index] ^ rightBytes[index];
  }
  return difference === 0;
}

function normalizeSeaceUrl(rawUrl) {
  const url = new URL(String(rawUrl || ""));
  if (
    url.protocol !== "https:" ||
    !ALLOWED_HOSTS.has(url.hostname.toLowerCase()) ||
    url.username ||
    url.password ||
    url.port
  ) {
    throw new Error("invalid_host");
  }
  if (!url.pathname.endsWith("/SdescargarArchivoAlfresco")) {
    throw new Error("invalid_path");
  }
  const fileCode = url.searchParams.get("fileCode");
  if (!FILE_CODE_PATTERN.test(fileCode || "")) {
    throw new Error("invalid_file_code");
  }
  url.search = new URLSearchParams({ fileCode }).toString();
  url.hash = "";
  return url;
}

export default {
  async fetch(request, env) {
    if (request.method === "GET") {
      return Response.json({ status: "ok", service: "licigob-seace-edge-proxy" });
    }
    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    const authorized = await secureEqual(
      request.headers.get("X-Proxy-Key"),
      env.PROXY_KEY,
    );
    if (!env.PROXY_KEY || !authorized) {
      return new Response("Unauthorized", { status: 401 });
    }

    let sourceUrl;
    try {
      const payload = await request.json();
      sourceUrl = normalizeSeaceUrl(payload.url);
    } catch {
      return new Response("Invalid document", { status: 400 });
    }

    const upstream = await fetch(sourceUrl, {
      headers: {
        Accept: "application/pdf,application/octet-stream;q=0.9,*/*;q=0.5",
        "Accept-Language": "es-PE,es;q=0.9",
        Referer: `https://${sourceUrl.hostname}/portal/`,
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " +
          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
      },
      redirect: "manual",
    });

    if (!upstream.ok || !upstream.body) {
      return new Response("Document unavailable", { status: 502 });
    }

    const declaredLength = Number(upstream.headers.get("Content-Length") || 0);
    if (declaredLength > MAX_FILE_BYTES) {
      return new Response("Document too large", { status: 413 });
    }
    const document = await upstream.arrayBuffer();
    if (document.byteLength === 0 || document.byteLength > MAX_FILE_BYTES) {
      return new Response("Invalid document size", { status: 413 });
    }
    const signature = new TextDecoder().decode(document.slice(0, 5));
    if (signature !== "%PDF-") {
      return new Response("Unexpected document type", { status: 422 });
    }

    const headers = new Headers({
      "Cache-Control": "private, no-store",
      "Content-Length": String(document.byteLength),
      "Content-Type": "application/pdf",
      "X-Content-Type-Options": "nosniff",
    });

    return new Response(document, { status: 200, headers });
  },
};
