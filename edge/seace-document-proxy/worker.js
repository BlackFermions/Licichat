const ALLOWED_HOSTS = new Set([
  "prod1.seace.gob.pe",
  "prod2.seace.gob.pe",
  "prod3.seace.gob.pe",
  "prod4.seace.gob.pe",
]);
const ALLOWED_ORIGINS = new Set([
  "https://www.licigob.pe",
  "https://licigob.pe",
  "http://localhost:5173",
]);
const FILE_CODE_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const TENDER_ID_PATTERN = /^[A-Za-z0-9._-]{1,80}$/;
const MAX_FILE_BYTES = 50 * 1024 * 1024;
const MAX_DOCUMENTS = 2;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

async function secureEqual(left, right) {
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

function decodeBase64Url(value) {
  const normalized = value.replace(/-/g, "+").replace(/_/g, "/");
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
  const binary = atob(padded);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

async function verifyTicket(ticket, secret) {
  if (!secret || typeof ticket !== "string") return null;
  const parts = ticket.split(".");
  if (parts.length !== 2) return null;
  try {
    const key = await crypto.subtle.importKey(
      "raw",
      encoder.encode(secret),
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["verify"],
    );
    const valid = await crypto.subtle.verify(
      "HMAC",
      key,
      decodeBase64Url(parts[1]),
      encoder.encode(parts[0]),
    );
    if (!valid) return null;
    const payload = JSON.parse(decoder.decode(decodeBase64Url(parts[0])));
    if (
      !TENDER_ID_PATTERN.test(String(payload.tender_id || "")) ||
      !Number.isFinite(payload.exp) ||
      payload.exp < Math.floor(Date.now() / 1000)
    ) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
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

function corsHeaders(origin) {
  return {
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

function jsonResponse(payload, status, origin) {
  return Response.json(payload, {
    status,
    headers: origin ? corsHeaders(origin) : {},
  });
}

function documentRank(document) {
  const title = String(document.title || "").toLowerCase();
  if (title.includes("bases integradas")) return 0;
  if (title.includes("bases administrativas")) return 1;
  if (document.document_type === "biddingDocuments") return 2;
  return 3;
}

async function fetchSeaceDocument(sourceUrl) {
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
  const declaredLength = Number(upstream.headers.get("Content-Length") || 0);
  if (!upstream.ok || !upstream.body) throw new Error("document_unavailable");
  if (declaredLength > MAX_FILE_BYTES) throw new Error("document_too_large");
  return upstream;
}

function detectedContentType(bytes) {
  const startsWith = (...expected) =>
    expected.every((value, index) => bytes[index] === value);
  if (startsWith(0x25, 0x50, 0x44, 0x46, 0x2d)) return "application/pdf";
  if (startsWith(0x50, 0x4b, 0x03, 0x04) || startsWith(0x50, 0x4b, 0x05, 0x06)) {
    return "application/zip";
  }
  if (startsWith(0x52, 0x61, 0x72, 0x21, 0x1a, 0x07)) return "application/vnd.rar";
  if (startsWith(0x37, 0x7a, 0xbc, 0xaf, 0x27, 0x1c)) return "application/x-7z-compressed";
  return null;
}

function serviceUrl(base, path) {
  const url = new URL(base);
  if (url.protocol !== "https:") throw new Error("invalid_service_url");
  url.pathname = path;
  url.search = "";
  url.hash = "";
  return url.toString();
}

function publicPreparationFailureCode(failures) {
  const reasons = failures.map((failure) => String(failure || ""));
  if (reasons.some((reason) => reason.includes("scanned_pdf") || reason.includes("ocr_required"))) {
    return "ocr_required";
  }
  if (reasons.some((reason) => reason.includes("file_too_large") || reason.includes("document_too_large"))) {
    return "file_too_large";
  }
  return "documents_unavailable";
}

async function prepareDocuments(request, env) {
  const origin = request.headers.get("Origin") || "";
  if (!ALLOWED_ORIGINS.has(origin)) {
    return jsonResponse({ status: "error", code: "origin_forbidden" }, 403, "");
  }
  const body = await request.json().catch(() => ({}));
  const ticket = await verifyTicket(body.ticket, env.EDGE_TICKET_KEY);
  if (!ticket) {
    return jsonResponse({ status: "error", code: "invalid_ticket" }, 401, origin);
  }

  const statusUrl = serviceUrl(env.AI_SERVICE_URL, "/api/v1/edge-status");
  const ingestUrl = serviceUrl(env.AI_SERVICE_URL, "/api/v1/edge-ingest");
  const privateHeaders = {
    "Content-Type": "application/json",
    "X-Edge-Ingest-Key": env.EDGE_INGEST_KEY,
  };
  const cachedResponse = await fetch(statusUrl, {
    method: "POST",
    headers: privateHeaders,
    body: JSON.stringify({ tender_id: ticket.tender_id }),
  });
  if (cachedResponse.ok) {
    const cached = await cachedResponse.json();
    if (cached.ready) {
      return jsonResponse({ status: "ready", source: "cache" }, 200, origin);
    }
  }

  const detailUrl = new URL(`/api/tenders/${encodeURIComponent(ticket.tender_id)}`, env.BACKEND_URL);
  const detailResponse = await fetch(detailUrl, { headers: { Accept: "application/json" } });
  if (!detailResponse.ok) {
    return jsonResponse({ status: "error", code: "tender_unavailable" }, 502, origin);
  }
  const detail = await detailResponse.json();
  const documents = (Array.isArray(detail.documents) ? detail.documents : [])
    .filter((document) => {
      const format = String(document.format || "").toLowerCase();
      return document.url && (format === "pdf" || format === "application/pdf");
    })
    .sort((left, right) => documentRank(left) - documentRank(right))
    .slice(0, MAX_DOCUMENTS);
  if (!documents.length) {
    return jsonResponse({ status: "error", code: "no_pdf" }, 422, origin);
  }

  let prepared = 0;
  let skipped = 0;
  const failures = [];
  for (const document of documents) {
    let stage = "normalize";
    try {
      const sourceUrl = normalizeSeaceUrl(document.url);
      stage = "download";
      const pdf = await fetchSeaceDocument(sourceUrl);
      stage = "ingest";
      const ingestResponse = await fetch(ingestUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/pdf",
          "X-Document-Url": sourceUrl.toString(),
          "X-Edge-Ingest-Key": env.EDGE_INGEST_KEY,
          "X-Tender-Id": String(ticket.tender_id),
        },
        body: pdf.body,
      });
      if (ingestResponse.ok) {
        prepared += 1;
      } else if (ingestResponse.status === 413 || ingestResponse.status === 422) {
        skipped += 1;
        const payload = await ingestResponse.json().catch(() => ({}));
        failures.push(`ingest_${ingestResponse.status}_${payload.code || "rejected"}`);
      } else {
        throw new Error(`status_${ingestResponse.status}`);
      }
    } catch (error) {
      skipped += 1;
      const reason = String(error?.message || "failed").replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 80);
      failures.push(`${stage}_${reason}`);
      console.warn(
        `edge_prepare_failed tender=${ticket.tender_id} stage=${stage} reason=${reason}`,
      );
    }
  }

  if (!prepared) {
    return jsonResponse(
      { status: "error", code: publicPreparationFailureCode(failures), skipped, failures },
      422,
      origin,
    );
  }
  return jsonResponse({ status: "ready", source: "edge", prepared, skipped }, 200, origin);
}

async function legacyProxy(request, env) {
  const authorized = await secureEqual(request.headers.get("X-Proxy-Key"), env.PROXY_KEY);
  if (!env.PROXY_KEY || !authorized) return new Response("Unauthorized", { status: 401 });
  let sourceUrl;
  try {
    const payload = await request.json();
    sourceUrl = normalizeSeaceUrl(payload.url);
  } catch {
    return new Response("Invalid document", { status: 400 });
  }
  try {
    const upstream = await fetchSeaceDocument(sourceUrl);
    const document = await upstream.arrayBuffer();
    if (document.byteLength === 0 || document.byteLength > MAX_FILE_BYTES) {
      return new Response("Invalid document size", { status: 413 });
    }
    const contentType = detectedContentType(new Uint8Array(document.slice(0, 8)));
    if (!contentType) {
      return new Response("Unexpected document type", { status: 422 });
    }
    return new Response(document, {
      status: 200,
      headers: {
        "Cache-Control": "private, no-store",
        "Content-Length": String(document.byteLength),
        "Content-Type": contentType,
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch {
    return new Response("Document unavailable", { status: 502 });
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "GET") {
      return Response.json({ status: "ok", service: "licigob-seace-edge-proxy" });
    }
    if (request.method === "OPTIONS" && url.pathname === "/prepare") {
      const origin = request.headers.get("Origin") || "";
      if (!ALLOWED_ORIGINS.has(origin)) return new Response(null, { status: 403 });
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }
    if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
    if (url.pathname === "/prepare") return prepareDocuments(request, env);
    if (url.pathname === "/") return legacyProxy(request, env);
    return new Response("Not found", { status: 404 });
  },
};
