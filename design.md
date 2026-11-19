# Design

Referência visual: a página “Introducing Griffin” da Tavus (tavus.io), gravada em 19 de novembro de 2026. Este documento traduz o que aparece na gravação para o atlas da vitivinicultura brasileira. A estrutura, o ritmo e o sistema visual vêm de lá. Marca, textos e números da Tavus ficam de fora.

O produto é uma página só, lida de cima a baixo, no lugar de um dashboard com várias rotas. Panorama, parceiros, produção e preço são seções numeradas dessa página.

## Princípios

- A página parece um relatório de pesquisa: título em serif, corpo em sans, diagramas retos, muito ar entre blocos.
- Cada seção responde uma pergunta e termina num gráfico ou num número grande.
- O acento rosa marca ação e a série em destaque. O verde marca quem lidera um comparativo. Os demais ficam em cinza.
- Unidade sempre visível: kg, litro, US$/kg, ano.
- Movimento curto. Quem prefere menos animação não vê a faixa de pixels nem o halo dos cards.

## Tom de tela

Uma coluna de leitura no centro, índice fixo à esquerda a partir da seção 01, hero em tela cheia com textura escura, faixa de três estatísticas, blocos 01–05, grade final escura e um rodapé com a faixa rosa animada.

Largura do texto corrido: 720px. Diagramas e gráficos podem ir até 880px. O hero e a faixa de números usam a largura da janela, com o conteúdo limitado a 1120px.

## Cor

Valores propostos a partir do que a gravação mostra. São o sistema do atlas, não uma amostra com conta-gotas do vídeo.

| Token | Valor | Uso |
|---|---|---|
| `--paper` | `#ffffff` | Fundo do artigo |
| `--ink` | `#161411` | Texto principal |
| `--ink-soft` | `#5e584f` | Apoio, legendas, índice inativo |
| `--line` | `#e7e2da` | Divisores e trilho de gráfico |
| `--hero` | `#2c1618` | Fundo do hero, com textura |
| `--hero-deep` | `#1a0d10` | Sombra e vinheta do hero |
| `--accent` | `#ff4d8d` | Botão sólido, série destacada, hover, faixa final |
| `--accent-deep` | `#e11d68` | Pixels escuros da faixa |
| `--win` | `#3dde7a` | Barra do item que lidera o comparativo |
| `--bar` | `#d8d3cb` | Barras dos demais |
| `--cta` | `#0c0c0c` | Fundo da grade final |
| `--grid` | `rgba(176, 48, 48, 0.28)` | Grade fina sobre o fundo da grade final |

Texto branco sobre `--hero`, `--cta` e `--accent`. Texto `--ink` sobre `--paper`. O verde não entra em botão.

## Tipo

- Títulos: serif de alto contraste, peso regular, tracking levemente fechado. Implementar com Newsreader. Na referência, o título do hero é grande, centralizado, em caixa alta só na linha de cima (“introduzindo”); a frase principal vai em caixa de frase.
- Interface, navegação, botões, números e legendas: sans geométrica. Implementar com Geist, com Inter se a primeira não carregar.
- Corpo: 18px, entrelinha 1.6, medida de 720px.
- Rótulo de seção: `01` em sans, pequeno, cor `--ink-soft`, seguido do título em serif.
- Número de destaque: sans, tabular, peso 500, grande o bastante para ocupar a coluna.

## Forma

- Botões da navegação: pílula. “Entrar” é contorno; “Começar” é sólido `--accent`.
- Botões do hero: o primário é sólido rosa; o secundário é contorno claro sobre o hero.
- Cards, diagramas e gráficos: canto reto, sem sombra. Separação por fio de 1px `--line` ou pelo próprio fundo.
- Índice lateral: sem caixa. O item da seção visível fica em `--ink` e peso 560. Os outros ficam em `--ink-soft`.

## Peças

### Navegação

Fixa no topo, sobre o hero e depois sobre o papel. Esquerda: marca do atlas. Centro: Panorama, Parceiros, Produção, Preço, Método. Direita: contorno “Dados” e sólido “Começar”, este último levando à seção 01. Hover de um item do centro abre um painel curto, com fade e um leve deslocamento para baixo, no mesmo gesto da referência. No atlas cada painel lista só a âncora da seção e uma linha do que ela responde.

### Hero

Fundo `--hero` com textura (grão ou trama), vinheta `--hero-deep`. Conteúdo centrado.

- Linha de cima, sans, caixa alta, espaçada: o período `1970–2024`.
- Título em serif: “O vinho que o Brasil bebe”.
- Subtítulo em sans, uma ou duas linhas: de onde vem, para onde vai, o que o país passou a produzir.
- Dois botões: “Ler o panorama” (sólido) e “Ver o método” (contorno).

### Faixa de números

Três colunas, logo abaixo do hero, ainda em fundo escuro, separadas por fios finos. Cada coluna tem um número grande e uma frase curta.

- `48%` — do vinho de mesa importado em 2024 veio do Chile.
- `US$ 1,64` — o quilo que o Brasil vendeu, contra `US$ 3,14` o quilo que comprou.
- `−53%` — a produção de vinho de mesa desde 1970, enquanto o suco de uva multiplicou.

### Artigo com índice

A partir daqui o fundo é `--paper`. À esquerda, um índice fixo com as cinco seções. À direita, a coluna de leitura. O índice marca a seção cujo título cruzou o terço superior da janela.

1. **O tamanho do desequilíbrio.** Texto curto e dois números lado a lado: 153 milhões de kg importados e 5,3 milhões exportados em 2024. Gráfico de barras horizontais, importação em `--accent`, exportação em `--bar`.
2. **Com quem o Brasil negocia.** Ranking horizontal. O país líder da série selecionada vai em `--win`; os outros em `--bar`. Um controle em pílula troca o fluxo: importação ou exportação, mesa ou espumante. Ao lado, um waffle de quadradinhos no estilo da referência: fatia do Paraguai nas exportações de vinho de mesa em 2024 (cerca de 70%) contra o restante, rosa contra cinza escuro.
3. **O que o país passou a produzir.** Duas séries na mesma escala de litros, só os totais de família (mesa, fino, suco, espumante). Mesa desce, suco sobe. Barras ou uma área simples, cantos retos, sem grade pesada.
4. **O preço por quilo.** US$/kg de 1970 a 2024 para o fluxo escolhido, e um recorte de 2024: espumante da França perto de US$ 16/kg contra o volume da Espanha. A série em foco usa `--accent`. A legenda segue o modelo da referência: rótulo curto ao lado de uma amostra de cor, fora do gráfico.
5. **Como estes números foram lidos.** Origem Embrapa, par kg e US$, totais separados das subcategorias, correção de nomes de país, e o aviso de que o laboratório de qualidade usa o WineQT, química de vinho tinto de outro conjunto, não prova de vinho brasileiro.

### Grade final

Fundo `--cta` com a grade fina `--grid`. Título em serif, branco: “Continuar a partir daqui”. Quatro cards em duas colunas, canto reto, fio sutil. Hover revela uma trama de pontos rosa sobre o card, como na referência.

- Parceiros
- Produção
- Baixar os dados (o JSON agregado)
- Laboratório, com o rótulo “em seguida”, levando a uma segunda página só quando ela existir

### Rodapé

Colunas enxutas: o atlas, as seções, a fonte dos dados, o repositório. Abaixo, uma faixa rosa clara com quadrados `--accent-deep` caindo em loop. A faixa para quando o sistema pede menos movimento.

## Gráficos

Todos no corpo do artigo, alinhados à coluna, com título em serif e legenda em sans.

- Barra horizontal. Eixo com o nome. Valor no fim da barra. Sem eixo numérico duplicado se o valor já está escrito.
- Waffle. Grade de células iguais. A parte em `--accent` é a fatia; o resto é cinza escuro. A porcentagem fica ao lado, em número grande.
- Série anual. Linha de 2px, pontos pequenos, anos no eixo, unidade no título. Uma série em `--accent`, a comparação em `--bar`.
- Verde `--win` só no item que lidera um ranking, nunca como cor de marca.

Hover de barra ou célula mostra o valor, a unidade e o ano. Não abre painel.

## Movimento

- Painel da navegação: surge em cerca de 160ms, com um deslocamento curto para baixo.
- Índice: troca de peso e cor sem deslizar a página.
- Card da grade final: a trama de pontos aparece no hover.
- Faixa do rodapé: loop contínuo de quadrados. Desligada em `prefers-reduced-motion`.

## Escrita na interface

Português. Frases curtas. Número e unidade na mesma linha. Sem sigla solta: na primeira vez, “US$ por quilograma”, depois “US$/kg”. O hero não promete previsão de qualidade.

## Responsivo

A gravação mostra só o desktop. No estreito, o índice sai da lateral e vira uma lista horizontal fixa sob a navegação. A faixa de três números e a grade final passam a uma coluna. O hero reduz o título e mantém os dois botões empilhados.
