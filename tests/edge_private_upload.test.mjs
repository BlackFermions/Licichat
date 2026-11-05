import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHmac } from 'node:crypto';

const source = await readFile(new URL('../edge/seace-document-proxy/worker.js', import.meta.url), 'utf8');
const { default: worker } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const env = {EDGE_TICKET_KEY:'test-ticket', EDGE_INGEST_KEY:'test-ingest', AI_SERVICE_URL:'https://ai.invalid', BACKEND_URL:'https://api.invalid'};
const token = (user=11, exp=Math.floor(Date.now()/1000)+60) => {
  const data=Buffer.from(JSON.stringify({tender_id:'123', user_id:user, exp})).toString('base64url');
  return `${data}.${createHmac('sha256',env.EDGE_TICKET_KEY).update(data).digest('base64url')}`;
};
const upload = (ticket=token(), origin='https://www.licigob.pe') => new Request('https://worker.invalid/upload', {
  method:'POST', headers:{Origin:origin,'Content-Type':'application/pdf','X-Preparation-Ticket':ticket}, body:'%PDF-test',
});

test('private upload requires allowed origin, valid ticket, positive owner', async () => {
  assert.equal((await worker.fetch(upload(token(),'https://attacker.invalid'),env)).status,403);
  assert.equal((await worker.fetch(upload('forged'),env)).status,401);
  assert.equal((await worker.fetch(upload(token(0)),env)).status,401);
  assert.equal((await worker.fetch(upload(token(11,1)),env)).status,401);
});

test('preflight allows private ticket header only to approved origin', async () => {
  const response=await worker.fetch(new Request('https://worker.invalid/upload',{method:'OPTIONS',headers:{Origin:'https://www.licigob.pe'}}),env);
  assert.equal(response.status,204);
  assert.match(response.headers.get('Access-Control-Allow-Headers'),/X-Preparation-Ticket/);
});

test('upload streams to private endpoint, binds signed owner, never fetches SEACE', async () => {
  const original=globalThis.fetch;
  const calls=[];
  globalThis.fetch=async (url, options) => {
    calls.push(String(url));
    assert.equal(options.headers['X-User-Id'],'11');
    assert.equal(options.headers['X-Tender-Id'],'123');
    assert.equal(options.headers['X-Edge-Ingest-Key'],'test-ingest');
    assert.ok(options.body instanceof ReadableStream);
    return Response.json({status:'ready',source:'private_upload',document_session_id:'test-session'});
  };
  try {
    const response=await worker.fetch(upload(),env);
    assert.equal(response.status,200);
    assert.equal((await response.json()).source,'private_upload');
    assert.deepEqual(calls,['https://ai.invalid/api/v1/private-document']);
  } finally {globalThis.fetch=original;}
});

test('read-only status retries a cold start and reuses existing text', async () => {
  const original=globalThis.fetch;
  let count=0;
  globalThis.fetch=async (url) => {
    assert.equal(String(url),'https://ai.invalid/api/v1/edge-status');
    count++;
    return count===1 ? new Response('cold',{status:503}) : Response.json({ready:true});
  };
  try {
    const response=await worker.fetch(new Request('https://worker.invalid/prepare',{
      method:'POST',headers:{Origin:'https://www.licigob.pe','Content-Type':'application/json'},body:JSON.stringify({ticket:token()}),
    }),env);
    assert.equal(response.status,200);
    assert.equal((await response.json()).source,'cache');
    assert.equal(count,2);
  } finally {globalThis.fetch=original;}
});

test('unavailable cache status never falls through to a SEACE download', async () => {
  const original=globalThis.fetch;
  let count=0;
  globalThis.fetch=async (url) => {
    assert.equal(String(url),'https://ai.invalid/api/v1/edge-status'); count++;
    return new Response('cold',{status:503});
  };
  try {
    const response=await worker.fetch(new Request('https://worker.invalid/prepare',{
      method:'POST',headers:{Origin:'https://www.licigob.pe'},body:JSON.stringify({ticket:token()}),
    }),env);
    assert.equal(response.status,503);
    assert.equal((await response.json()).code,'ai_temporarily_unavailable');
    assert.equal(count,2);
  } finally {globalThis.fetch=original;}
});
