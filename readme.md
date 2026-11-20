# Atlas da vitivinicultura brasileira

Página única sobre o vinho que o Brasil bebe, de 1970 a 2024: de onde vem, para onde vai e o que o país passou a produzir. O texto e os gráficos ficam em `web/` e leem um JSON já agregado.

## De onde vêm os dados

A fonte é a Embrapa Uva e Vinho. Os arquivos estão em `data/`.

| Arquivo | O que entra no JSON |
|---|---|
| `ImpVinhosMesa.csv`, `ExpVinhoMesa.csv` | Vinho de mesa, importação e exportação. Separador tab. Cada ano aparece duas vezes: a primeira coluna é kg, a segunda é US$. |
| `ImpEspumantes.csv`, `ExpEspumantes.csv` | O mesmo par, para espumante. |
| `Producao.csv` | Litros por família. Separador `;`. Em 2024 o ponto é separador de milhar. Entram só os totais de mesa, fino, suco e a soma dos espumantes. |
| `Comercio.csv` | Comércio interno, separador `;`. O script precisa abri-lo. O conteúdo não entra no JSON. |

`WineQT.csv` fica em `data/` e não entra nesta página. O laboratório de qualidade usa outro conjunto e não é vinho brasileiro.

## Gerar o JSON

Na raiz do repositório, com Python 3, sem dependências além da biblioteca padrão:

```bash
python scripts/build_atlas.py
```

O script lê `data/` e grava `web/public/atlas.json`. O arquivo traz `meta`, `destaques`, `comercio`, `producao` e `preco`. Quantidade e valor ficam em campos separados. O preço é US$ por quilograma, `usd / kg`, e só entra quando há quilogramas.

Para conferir os totais de 2024 e os litros de 1970 e 2024:

```bash
python -m unittest tests/test_build_atlas.py
```

## Abrir a página

```bash
cd web
npm install
npm run dev
```

A página abre em `http://127.0.0.1:5173`.

`npm run build` gera a versão estática em `web/dist`. `npm run preview` serve essa versão em `http://127.0.0.1:4173`.

## app_streamlit/

A pasta `streamlit/` passou a se chamar `app_streamlit/`, para não encobrir a biblioteca Streamlit. Ela está vazia: os arquivos são só marcação, não entram nesta versão e a página não os serve.
