const dadosHref = `${import.meta.env.BASE_URL}atlas.json`

const cards = [
  {
    id: 'parceiros',
    title: 'Parceiros',
    text: 'Quais países lideram cada fluxo.',
    href: '#parceiros',
  },
  {
    id: 'producao',
    title: 'Produção',
    text: 'Litros de mesa, fino, suco e espumante.',
    href: '#producao',
  },
  {
    id: 'dados',
    title: 'Baixar os dados',
    text: 'O JSON agregado da página.',
    href: dadosHref,
    download: 'atlas.json',
  },
] as const

export function ClosingGrid() {
  return (
    <section className="closing" id="continuar" aria-labelledby="continuar-titulo">
      <div className="closing-inner">
        <h2 id="continuar-titulo" className="closing-title">
          Continuar a partir daqui
        </h2>
        <ul className="closing-grid">
          {cards.map((card) => (
            <li key={card.id}>
              <a
                className="closing-card"
                href={card.href}
                {...('download' in card ? { download: card.download } : {})}
              >
                <h3 className="closing-card-title">{card.title}</h3>
                <p className="closing-card-text">{card.text}</p>
              </a>
            </li>
          ))}
          <li>
            <div className="closing-card">
              <h3 className="closing-card-title">Laboratório</h3>
              <p className="closing-soon">em seguida</p>
            </div>
          </li>
        </ul>
      </div>
    </section>
  )
}
