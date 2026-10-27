import { describe, expect, it, vi } from 'vitest';
import { launchRun, validateSettings, workbenchUrl } from './launch.js';
import { rewriteLocalOrigin } from '../devProxy.js';

const settings = { circuit: 'spa', architecture: 'deep', generations: '75', seed: '2026', file: null };
const success = () => ({ ok: true, json: async () => ({}) });

function requestBodies(fetcher) {
  return fetcher.mock.calls.map(([path, options]) => [path, options.body ? JSON.parse(options.body) : undefined]);
}

describe('starting a run', () => {
  it('stops, resets the circuit, applies settings, then starts training in order', async () => {
    const fetcher = vi.fn(async () => success());
    await launchRun(settings, { fetcher });
    expect(requestBodies(fetcher)).toEqual([
      ['/api/train', { running: false }], ['/api/circuit', { id: 'spa' }],
      ['/api/architecture', { id: 'deep' }], ['/api/seed', { seed: 2026 }],
      ['/api/generations', { generations: 75 }], ['/api/train', { running: true }],
    ]);
  });

  it('starts the four-network comparison only after applying the chosen settings', async () => {
    const fetcher = vi.fn(async () => success());
    await launchRun(settings, { compare: true, fetcher });
    expect(requestBodies(fetcher).slice(-2)).toEqual([
      ['/api/generations', { generations: 75 }], ['/api/comparison', undefined],
    ]);
    expect(fetcher).toHaveBeenCalledTimes(6);
  });

  it('applies the pending image instead of a preset circuit', async () => {
    const file = new File(['image'], 'my-track.png', { type: 'image/png' });
    const fetcher = vi.fn(async () => success());
    await launchRun({ ...settings, file }, { fetcher });
    expect(fetcher.mock.calls[1][0]).toBe('/api/track');
    expect(fetcher.mock.calls[1][1].body.get('file')).toBe(file);
    expect(fetcher.mock.calls.some(([path]) => path === '/api/circuit')).toBe(false);
  });

  it('stops immediately and shows the server message when a setting fails', async () => {
    const fetcher = vi.fn(async (path) => path === '/api/architecture'
      ? { ok: false, json: async () => ({ error: { message: 'Network not available.' } }) }
      : success());
    await expect(launchRun(settings, { fetcher })).rejects.toThrow('Network not available.');
    expect(fetcher).toHaveBeenCalledTimes(3);
  });

  it('rejects invalid settings before changing the simulation', async () => {
    const fetcher = vi.fn();
    await expect(launchRun({ ...settings, seed: '-1' }, { fetcher })).rejects.toThrow('seed');
    await expect(launchRun({ ...settings, generations: '1.5' }, { fetcher })).rejects.toThrow('generation');
    expect(fetcher).not.toHaveBeenCalled();
    expect(() => validateSettings({ ...settings, generations: '10000', seed: '4294967295' })).not.toThrow();
  });
});

describe('local development routing', () => {
  it('routes built pages to the same-origin workbench and development to its backend', () => {
    expect(workbenchUrl({ dev: false })).toBe('/workbench');
    expect(workbenchUrl({ dev: true, backend: 'http://127.0.0.1:8004' })).toBe('http://127.0.0.1:8004/workbench');
  });

  it('rewrites only an exact matching local development origin', () => {
    const proxyRequest = { setHeader: vi.fn() };
    const backend = new URL('http://127.0.0.1:8004');
    rewriteLocalOrigin(proxyRequest, { headers: { host: '127.0.0.1:5174', origin: 'http://127.0.0.1:5174' } }, backend);
    expect(proxyRequest.setHeader).toHaveBeenCalledWith('Origin', backend.origin);
    proxyRequest.setHeader.mockClear();
    for (const origin of ['https://other.example', 'http://localhost:5174', 'http://127.0.0.1:5174.attacker.example']) {
      rewriteLocalOrigin(proxyRequest, { headers: { host: '127.0.0.1:5174', origin } }, backend);
    }
    expect(proxyRequest.setHeader).not.toHaveBeenCalled();
  });
});
