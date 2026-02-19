import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { useEffect, useState } from 'react'
import './index.css'
import App from './App.tsx'
import FlowLabPage from './FlowLabPage.tsx'

const getHashRoute = () => window.location.hash || '#/'

function Root() {
  const [route, setRoute] = useState<string>(getHashRoute())

  useEffect(() => {
    const onHashChange = () => setRoute(getHashRoute())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  if (route.startsWith('#/flow-lab')) {
    return <FlowLabPage />
  }
  return <App />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
)
