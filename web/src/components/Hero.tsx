import { Button } from './Button'

export function Hero() {
  return (
    <section className="hero" id="topo" aria-labelledby="hero-titulo">
      <div className="hero-inner">
        <p className="hero-kicker">1970–2024</p>
        <h1 id="hero-titulo" className="hero-title">
          O vinho que o Brasil bebe
        </h1>
        <p className="hero-sub">
          De onde vem, para onde vai, o que o país passou a produzir.
        </p>
        <div className="hero-actions">
          <Button href="#desequilibrio" variant="solid">
            Ler o panorama
          </Button>
          <Button href="#metodo" variant="outline-light">
            Ver o método
          </Button>
        </div>
      </div>
    </section>
  )
}
