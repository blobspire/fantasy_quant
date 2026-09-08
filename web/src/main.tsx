import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './theme.css';

/**
 * Entry point. Nothing but mounting happens here.
 *
 * StrictMode is on: it double-invokes effects in development, which is exactly
 * the pressure `useResource` needs to be correct about aborting a request whose
 * component went away mid-simulation. It has no effect on the production build.
 */
const container = document.getElementById('root');
if (!container) throw new Error('#root is missing from index.html');

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
