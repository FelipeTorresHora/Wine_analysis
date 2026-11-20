import { ClosingGrid } from './components/ClosingGrid'
import { Footer } from './components/Footer'
import { Hero } from './components/Hero'
import { Index } from './components/Index'
import { Nav } from './components/Nav'
import { NumberBand } from './components/NumberBand'
import { useShellScroll } from './lib/useShellScroll'
import { Desequilibrio } from './sections/Desequilibrio'
import { Parceiros } from './sections/Parceiros'
import { Producao } from './sections/Producao'
import { Preco } from './sections/Preco'
import { Metodo } from './sections/Metodo'

export function App() {
  const { activeId, indexVisible, navTone } = useShellScroll()

  return (
    <>
      <Nav activeId={activeId} tone={navTone} />
      <div id="faixa-escura">
        <Hero />
        <NumberBand />
      </div>
      <Index activeId={activeId} visible={indexVisible} />
      <main id="artigo">
        <article className="article">
          <Desequilibrio />
          <Parceiros />
          <Producao />
          <Preco />
          <Metodo />
        </article>
      </main>
      <ClosingGrid />
      <Footer />
    </>
  )
}
