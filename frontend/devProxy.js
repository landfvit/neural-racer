export function rewriteLocalOrigin(proxyRequest, incomingRequest, backend) {
  const host = incomingRequest.headers.host;
  const origin = incomingRequest.headers.origin;
  // keep foreign origins intact so the backend can reject them
  if (/^(127\.0\.0\.1|localhost):\d+$/.test(host || '') && origin === `http://${host}`) {
    proxyRequest.setHeader('Origin', backend.origin);
  }
}
