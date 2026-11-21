import { sections } from '../sections/manifest'

const REPO = 'https://github.com/FelipeTorresHora/Wine_analysis'

type Quadrado = {
  left: string
  size: number
  duration: string
  delay: string
}

function montarQuadrados(total: number): Quadrado[] {
  const lista: Quadrado[] = []
  for (let i = 0; i < total; i += 1) {
    const left = ((i * 47) % 1000) / 10
    const size = 12 + (i % 3) * 6
    const duration = 5.2 + (i % 5) * 0.9
    const delay = -((i * 1.17) % duration)
    lista.push({
      left: `${left}%`,
      size,
      duration: `${duration.toFixed(2)}s`,
      delay: `${delay.toFixed(2)}s`,
    })
  }
  return lista
}

const QUADRADOS = montarQuadrados(36)

export function Footer() {
  return (
    <footer className="site-footer" id="rodape">
      <div className="footer-inner">
        <div className="footer-col">
          <p className="footer-label">O atlas</p>
          <p className="footer-mark">Atlas do vinho</p>
          <p className="footer-note">O que o Brasil bebe, vende e produz, de 1970 a 2025.</p>
        </div>
        <nav className="footer-col" aria-label="Seções">
          <p className="footer-label">Seções</p>
          <ul>
            {sections.map((section) => (
              <li key={section.id}>
                <a href={`#${section.id}`}>
                  <span className="footer-num">{section.number}</span>
                  {section.nav}
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <div className="footer-col">
          <p className="footer-label">Fonte</p>
          <p>Embrapa Uva e Vinho</p>
          <p className="footer-note">Quilogramas, litros e US$ por quilograma.</p>
        </div>
        <div className="footer-col">
          <p className="footer-label">Repositório</p>
          <p>
            <a href={REPO}>FelipeTorresHora/Wine_analysis</a>
          </p>
          <p className="footer-note">O JSON agregado fica em web/public/atlas.json.</p>
        </div>
      </div>
      <div className="pixel-band" aria-hidden="true">
        {QUADRADOS.map((quadrado, index) => (
          <span
            key={index}
            className="pixel"
            style={{
              left: quadrado.left,
              width: quadrado.size,
              height: quadrado.size,
              animationDuration: quadrado.duration,
              animationDelay: quadrado.delay,
            }}
          />
        ))}
      </div>
    </footer>
  )
}
