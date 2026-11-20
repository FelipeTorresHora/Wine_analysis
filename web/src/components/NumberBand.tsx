import { useAtlas } from '../lib/useAtlas'

export function NumberBand() {
  const { status, destaques } = useAtlas()

  return (
    <section className="band" aria-label="Destaques" aria-busy={status === 'loading'}>
      {status === 'ready' ? (
        <ol className="band-list">
          {destaques.map((item, index) => (
            <li key={item.id || `${item.valor}-${index}`} className="band-item">
              <p className="band-value">{item.valor}</p>
              <p className="band-frase">{item.frase}</p>
            </li>
          ))}
        </ol>
      ) : (
        <p className="band-status" role={status === 'error' ? 'alert' : 'status'}>
          {status === 'error' ? 'Não foi possível ler os destaques.' : 'Carregando os números.'}
        </p>
      )}
    </section>
  )
}
