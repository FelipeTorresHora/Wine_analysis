import { Hero } from './components/Hero'
import { Index } from './components/Index'
import { Nav } from './components/Nav'
import { NumberBand } from './components/NumberBand'
import { useShellScroll } from './lib/useShellScroll'
import { Section01 } from './sections/Section01'
import { Section02 } from './sections/Section02'
import { Section03 } from './sections/Section03'
import { Section04 } from './sections/Section04'
import { Section05 } from './sections/Section05'

export function App() {
  const { activeId, indexVisible, navOnPaper } = useShellScroll()

  return (
    <>
      <Nav activeId={activeId} onPaper={navOnPaper} />
      <div id="faixa-escura">
        <Hero />
        <NumberBand />
      </div>
      <Index activeId={activeId} visible={indexVisible} />
      <main id="artigo">
        <article className="article">
          <Section01 />
          <Section02 />
          <Section03 />
          <Section04 />
          <Section05 />
        </article>
      </main>
    </>
  )
}
