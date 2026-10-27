import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App.jsx';

const circuits = [{ id: 'monza', name: 'Monza' }, { id: 'spa', name: 'Spa-Francorchamps' }];
const architectures = [{ id: 'compact', name: 'Compact', sizes: [5, 12, 3] }, { id: 'deep', name: 'Deep', sizes: [5, 16, 12, 3] }];

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  vi.stubGlobal('fetch', vi.fn(async (path) => {
    if (path === '/api/circuits') return { ok: true, json: async () => ({ circuits }) };
    if (path === '/api/architectures') return { ok: true, json: async () => ({ architectures }) };
    if (path.endsWith('circuits.json')) return { ok: true, json: async () => ({ circuits: [] }) };
    return { ok: true, json: async () => ({}) };
  }));
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it('updates the launch circuit when a circuit card is selected', async () => {
  render(<App navigate={vi.fn()} />);
  const card = await screen.findByRole('button', { name: 'Select Spa-Francorchamps' });
  fireEvent.click(card);
  expect(screen.getByLabelText('Circuit').value).toBe('spa');
  expect(card.getAttribute('aria-pressed')).toBe('true');
});

it('keeps user settings visible when launch fails and does not navigate', async () => {
  const navigate = vi.fn();
  render(<App navigate={navigate} />);
  await screen.findByRole('button', { name: 'Select Monza' });
  fireEvent.change(screen.getByLabelText('Seed'), { target: { value: '21' } });
  fireEvent.change(screen.getByLabelText('Neural network'), { target: { value: 'deep' } });
  const originalFetch = fetch;
  vi.stubGlobal('fetch', vi.fn(async (path, options) => path === '/api/circuit'
    ? { ok: false, json: async () => ({ error: { message: 'Circuit could not be loaded.' } }) }
    : originalFetch(path, options)));
  fireEvent.click(screen.getByRole('button', { name: 'Start training' }));
  expect((await screen.findByRole('alert')).textContent).toContain('Circuit could not be loaded.');
  expect(screen.getByLabelText('Seed').value).toBe('21');
  expect(screen.getByLabelText('Neural network').value).toBe('deep');
  expect(navigate).not.toHaveBeenCalled();
  await waitFor(() => expect(screen.getByRole('button', { name: 'Start training' }).disabled).toBe(false));
});

it('holds an uploaded file until launch and clears it when a preset is chosen', async () => {
  render(<App navigate={vi.fn()} />);
  await screen.findByRole('button', { name: 'Select Monza' });
  const file = new File(['track'], 'practice.png', { type: 'image/png' });
  fireEvent.change(screen.getByLabelText('Upload a custom track'), { target: { files: [file] } });
  expect(screen.getByLabelText('Circuit').value).toBe('custom');
  expect(screen.getByText('Selected: practice.png')).toBeTruthy();
  expect(fetch.mock.calls.some(([, options]) => options?.method === 'POST')).toBe(false);
  fireEvent.click(screen.getByRole('button', { name: 'Select Spa-Francorchamps' }));
  expect(screen.getByLabelText('Circuit').value).toBe('spa');
  expect(screen.queryByText('Selected: practice.png')).toBeNull();
});
