import { useEffect, useRef, useState } from 'react';
import { launchRun, MAX_UPLOAD_BYTES, request, workbenchUrl } from './launch.js';

const githubUrl = 'https://github.com/landfvit/neural-racer';
const circuitNotes = {
  monza: 'Long straights, decisive turns.',
  silverstone: 'Fast bends that flow together.',
  spa: 'A varied, winding route.',
  monaco: 'Tight turns. Little room to spare.',
  interlagos: 'A compact rhythm of changing turns.',
};

export default function App({ navigate = (url) => window.location.assign(url) }) {
  const [circuits, setCircuits] = useState([]);
  const [architectures, setArchitectures] = useState([]);
  const [outlines, setOutlines] = useState({});
  const [circuit, setCircuit] = useState('monza');
  const [architecture, setArchitecture] = useState('compact');
  const [generations, setGenerations] = useState('100');
  const [seed, setSeed] = useState('42');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(null);
  const [error, setError] = useState('');
  const [catalogAttempt, setCatalogAttempt] = useState(0);
  const fileInput = useRef(null);
  const circuitSelect = useRef(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError('');
    Promise.all([request('/api/circuits'), request('/api/architectures')])
      .then(([circuitData, architectureData]) => {
        if (!Array.isArray(circuitData.circuits) || !circuitData.circuits.length ||
            !Array.isArray(architectureData.architectures) || !architectureData.architectures.length) {
          throw new Error('The circuit and network catalog could not be loaded. Please try again.');
        }
        if (!active) return;
        setCircuits(circuitData.circuits);
        setArchitectures(architectureData.architectures);
        setCircuit((selected) => circuitData.circuits.some((item) => item.id === selected) ? selected : circuitData.circuits[0].id);
        setArchitecture((selected) => architectureData.architectures.some((item) => item.id === selected) ? selected : architectureData.architectures[0].id);
      })
      .catch((failure) => { if (active) setError(failure.message); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [catalogAttempt]);

  useEffect(() => {
    let active = true;
    fetch(`${import.meta.env.BASE_URL}circuits.json`)
      .then((response) => response.ok ? response.json() : Promise.reject())
      .then((data) => {
        if (active) setOutlines(Object.fromEntries(data.circuits.map((item) => [item.id, item])));
      })
      .catch(() => {});
    return () => { active = false; };
  }, []);

  const busy = pending !== null;
  const ready = !loading && circuits.length > 0 && architectures.length > 0;

  function chooseCircuit(id, scroll = false) {
    setCircuit(id);
    setFile(null);
    setError('');
    if (fileInput.current) fileInput.current.value = '';
    if (scroll) {
      const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;
      document.getElementById('start')?.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' });
      circuitSelect.current?.focus({ preventScroll: true });
    }
  }

  function chooseFile(event) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    if (!['image/png', 'image/jpeg'].includes(selected.type)) {
      setError('Upload a PNG or JPEG track image.');
      event.target.value = '';
      return;
    }
    if (selected.size > MAX_UPLOAD_BYTES) {
      setError('Upload an image smaller than 8 MB.');
      event.target.value = '';
      return;
    }
    setFile(selected);
    setError('');
  }

  async function start(compare = false) {
    if (busy || !ready) return;
    setError('');
    setPending(compare ? 'compare' : 'train');
    try {
      await launchRun({ circuit, architecture, generations, seed, file }, { compare });
      navigate(workbenchUrl());
    } catch (failure) {
      setError(failure.message);
      setPending(null);
    }
  }

  return (
    <>
      <header className="site-header">
        <a className="wordmark" href="#" aria-label="NeuralRacer home">NeuralRacer</a>
        <nav aria-label="Main navigation">
          <a className="section-link" href="#circuits">Circuits</a>
          <a className="section-link" href="#how-it-learns">How it learns</a>
          <a className="github-nav" href={githubUrl} target="_blank" rel="noreferrer">GitHub</a>
          <a className="workbench-link" href={workbenchUrl()}>Open workbench</a>
        </nav>
      </header>

      <main>
        <section className="hero" aria-labelledby="hero-title">
          <div className="hero-glow" aria-hidden="true" />
          <div className="hero-content">
            <h1 id="hero-title">Teach a car to race.</h1>
            <p className="hero-description">A little neural network. A lot of trial and error.<br className="desktop-break" /> Watch a better driver emerge, generation by generation.</p>

            <form id="start" className="run-composer" onSubmit={(event) => { event.preventDefault(); start(); }} aria-label="Configure a training run" aria-busy={busy}>
              <div className="primary-fields">
                <label className="field" htmlFor="circuit">
                  <span>Circuit</span>
                  <select id="circuit" ref={circuitSelect} value={file ? 'custom' : circuit} onChange={(event) => chooseCircuit(event.target.value)} disabled={busy || loading}>
                    {loading && <option value={circuit}>Loading circuits…</option>}
                    {file && <option value="custom" disabled>Custom track</option>}
                    {circuits.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
                  </select>
                </label>
                <label className="field" htmlFor="network">
                  <span>Neural network</span>
                  <select id="network" value={architecture} onChange={(event) => setArchitecture(event.target.value)} disabled={busy || loading}>
                    {loading && <option value={architecture}>Loading networks…</option>}
                    {architectures.map((item) => <option key={item.id} value={item.id}>{item.name} ({item.sizes.join('–')})</option>)}
                  </select>
                </label>
              </div>
              <div className="run-fields">
                <label className="field number-field" htmlFor="generations"><span>Generations</span><input id="generations" type="number" inputMode="numeric" min="1" max="10000" step="1" required value={generations} onChange={(event) => setGenerations(event.target.value)} disabled={busy} /></label>
                <label className="field number-field" htmlFor="seed"><span>Seed</span><input id="seed" type="number" inputMode="numeric" min="0" max="4294967295" step="1" required value={seed} onChange={(event) => setSeed(event.target.value)} disabled={busy} /></label>
                <button className="start-button" type="submit" disabled={!ready || busy}>{pending === 'train' ? 'Starting training…' : pending === 'compare' ? 'Starting comparison…' : 'Start training'}</button>
              </div>
              <div className="composer-actions">
                <input className="file-input" ref={fileInput} type="file" accept="image/png,image/jpeg" onChange={chooseFile} tabIndex={-1} aria-label="Upload a custom track" disabled={busy} />
                <button className="text-button upload-button" type="button" onClick={() => fileInput.current?.click()} disabled={busy}>{file ? 'Change track' : 'Upload a track'}</button>
                <button className="text-button compare-button" type="button" onClick={() => start(true)} disabled={!ready || busy}>Compare all four networks</button>
              </div>
              {file && <p className="file-selection" title={file.name}>Selected: {file.name}</p>}
              {error && <div className="form-error" role="alert"><span>{error}</span>{!ready && !loading && <button type="button" onClick={() => setCatalogAttempt((value) => value + 1)}>Try again</button>}</div>}
            </form>
            <p className="hero-footnote">Runs locally. Opens live in your workbench.</p>
          </div>
        </section>

        <section className="circuits-section content-width" id="circuits" aria-labelledby="circuits-title">
          <div className="section-intro">
            <div><p className="eyebrow">Pick your starting line</p><h2 id="circuits-title">Every circuit teaches<br className="desktop-break" /> something different.</h2></div>
            <p>Five circuit-inspired layouts, adapted for training.<br className="desktop-break" /> Find a rhythm on the straights or work on the turns.</p>
          </div>
          <div className="circuit-grid">
            {circuits.map((item, index) => {
              const outline = outlines[item.id];
              const selected = !file && circuit === item.id;
              return <button key={item.id} type="button" className={`circuit-card${selected ? ' is-selected' : ''}`} onClick={() => chooseCircuit(item.id, true)} disabled={busy} aria-pressed={selected} aria-label={`Select ${item.name}`}>
                <div className="circuit-art">
                  <span className="circuit-number">0{index + 1}</span>
                  {outline && <svg viewBox={`0 0 ${outline.width} ${outline.height}`} role="img" aria-label={`${item.name} training layout`}><polygon points={outline.route.map((point) => point.join(',')).join(' ')} fill="none" stroke="currentColor" strokeWidth="18" strokeLinejoin="round" strokeLinecap="round" /></svg>}
                  <span className="selection-text">{selected ? 'Selected' : 'Choose circuit'}</span>
                </div>
                <h3>{item.name}</h3>
                <p>{circuitNotes[item.id] || item.difficulty}</p>
              </button>;
            })}
          </div>
          <p className="circuit-caption">Or bring your own: upload a PNG or JPEG track from the starting panel.</p>
        </section>

        <section className="learning-section content-width" id="how-it-learns" aria-labelledby="learning-title">
          <div className="section-intro"><div><p className="eyebrow">From zero to a better lap</p><h2 id="learning-title">Small networks.<br />Visible progress.</h2></div><p>No pretrained driver. The first generation starts<br className="desktop-break" /> with zero weights, then learns through evolution.</p></div>
          <div className="learning-grid">
            <article><span className="step-number">01</span><h3>Sense the road.</h3><p>Five rays measure the distance to the walls. The network turns those readings into steering, target speed, and braking.</p></article>
            <article><span className="step-number">02</span><h3>Try. Select. Mutate.</h3><p>Cars drive, the best performers are selected, and their networks are mutated. Each new generation gets another chance.</p></article>
            <article><span className="step-number">03</span><h3>Find a faster lap.</h3><p>Watch the cars and inspect their networks live. For drivers that finish a lap, a shorter lap time wins.</p></article>
          </div>
        </section>
      </main>

      <footer className="site-footer content-width"><a className="wordmark" href="#">NeuralRacer</a><p>Python simulation. Local training.</p><a href={githubUrl} target="_blank" rel="noreferrer">View on GitHub</a></footer>
    </>
  );
}
