import { Button } from './Button'
import { sections } from '../sections/manifest'

type Props = {
  activeId: string | null
  onPaper: boolean
}

const dadosHref = `${import.meta.env.BASE_URL}atlas.json`

export function Nav({ activeId, onPaper }: Props) {
  return (
    <header id="navegacao" className={onPaper ? 'nav-bar is-paper' : 'nav-bar'}>
      <a className="brand" href="#topo">
        <span className="brand-mark">Atlas</span>
        <span className="brand-sub">do vinho</span>
      </a>
      <nav className="nav-links" aria-label="Principal">
        <ul>
          {sections.map((section) => {
            const current = activeId === section.id
            return (
              <li key={section.id} className="nav-item">
                <a
                  href={`#${section.id}`}
                  aria-current={current ? 'location' : undefined}
                  aria-describedby={`${section.id}-resumo`}
                >
                  {section.nav}
                </a>
                <div className="nav-panel">
                  <a href={`#${section.id}`} tabIndex={-1}>
                    <span className="nav-panel-number">{section.number}</span>
                    <span className="nav-panel-title">{section.title}</span>
                  </a>
                  <p id={`${section.id}-resumo`}>{section.answer}</p>
                </div>
              </li>
            )
          })}
        </ul>
      </nav>
      <div className="nav-actions">
        <Button href={dadosHref} variant={onPaper ? 'outline' : 'outline-light'}>
          Dados
        </Button>
        <Button href="#desequilibrio" variant="solid">
          Começar
        </Button>
      </div>
    </header>
  )
}
