# LiciGob SEACE edge proxy

Authenticated Cloudflare Worker used when SEACE rejects an Azure egress
address. The browser starts `/prepare` with a short-lived backend ticket, so
the download runs near the Peruvian user. The Worker validates the SEACE host,
download route and `fileCode`, then streams the PDF directly to AI Lite.

```powershell
npx wrangler login
npx wrangler secret put PROXY_KEY
npx wrangler secret put EDGE_TICKET_KEY
npx wrangler secret put EDGE_INGEST_KEY
npx wrangler deploy
```

`EDGE_TICKET_KEY` is shared only with the backend. `EDGE_INGEST_KEY` is shared
only with AI Lite. The legacy `PROXY_KEY` keeps the server-side fallback
available without exposing any secret to the browser.

The Worker intentionally uses Cloudflare's default edge placement. SEACE may
reject requests executed from a data-center location even when the same route
works from the Peru edge. The batch ingestion pilot therefore stages originals
in Blob before starting Azure compute instead of forcing a Worker region.
