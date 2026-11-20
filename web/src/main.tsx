import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/geist'
import '@fontsource-variable/inter'
import '@fontsource-variable/newsreader'
import { App } from './App'
import './styles/tokens.css'
import './styles/global.css'

const root = document.getElementById('root')
if (!root) throw new Error('Elemento #root ausente.')

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
