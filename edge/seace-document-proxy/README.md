# LiciGob SEACE edge proxy

Small authenticated Cloudflare Worker used only when SEACE rejects an Azure
egress address. It validates the SEACE host, download route and `fileCode`
before forwarding a PDF response.

```powershell
npx wrangler login
npx wrangler secret put PROXY_KEY
npx wrangler deploy
```

Configure the resulting HTTPS URL and the same random secret as
`SEACE_DOCUMENT_PROXY_URL` and `SEACE_DOCUMENT_PROXY_KEY` in
`licigob-ai-lite`.
