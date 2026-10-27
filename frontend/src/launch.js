export const MAX_UPLOAD_BYTES = 8 * 1024 * 1024;

export function validateSettings({ circuit, architecture, generations, seed, file }) {
  if (!file && !circuit) throw new Error('Choose a circuit for your run.');
  if (!architecture) throw new Error('Choose a neural network for your run.');
  if (!/^\d+$/.test(String(generations)) || Number(generations) < 1 || Number(generations) > 10000) {
    throw new Error('Choose a generation budget between 1 and 10,000.');
  }
  if (!/^\d+$/.test(String(seed)) || Number(seed) > 4294967295) {
    throw new Error('Choose a seed between 0 and 4,294,967,295.');
  }
  if (file) {
    if (!['image/png', 'image/jpeg'].includes(file.type)) throw new Error('Upload a PNG or JPEG track image.');
    if (file.size > MAX_UPLOAD_BYTES) throw new Error('Upload an image smaller than 8 MB.');
  }
}

export async function request(path, options = {}, fetcher = fetch) {
  let response;
  try {
    response = await fetcher(path, options);
  } catch {
    throw new Error('Could not reach the simulation. Check that the local server is running and try again.');
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error('The simulation returned an unexpected response. Please try again.');
  }
  if (!response.ok) throw new Error(data?.error?.message || 'The simulation could not start this run. Please try again.');
  return data;
}

export function workbenchUrl({ dev = import.meta.env.DEV, backend = __NEURAL_RACER_BACKEND__ } = {}) {
  // the vite development server only serves the landing page
  return dev ? `${backend}/workbench` : '/workbench';
}

export async function launchRun(settings, { compare = false, fetcher = fetch } = {}) {
  validateSettings(settings);
  const post = (path, data) => request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  }, fetcher);

  // applying a fresh track first keeps the generation budget independent of earlier runs
  await post('/api/train', { running: false });
  if (settings.file) {
    const body = new FormData();
    body.append('file', settings.file);
    await request('/api/track', { method: 'POST', body }, fetcher);
  } else {
    await post('/api/circuit', { id: settings.circuit });
  }
  await post('/api/architecture', { id: settings.architecture });
  await post('/api/seed', { seed: Number(settings.seed) });
  await post('/api/generations', { generations: Number(settings.generations) });
  if (compare) {
    await request('/api/comparison', { method: 'POST' }, fetcher);
  } else {
    await post('/api/train', { running: true });
  }
}
